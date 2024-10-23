use crate::{Operators, RandomSample, Weights};
use gguf::{GGufMetaMapExt, GGufModel};
use llama::{ext::f16, LlamaArgs, LlamaMeta, LlamaRequest, LlamaStorage, LlamaWorker, Tensor};
use operators::{
    cuda::{self, memcpy_d2h, Device, NoDevice},
    nvidia_gpu::{Config, Gpu},
    random_sample::{KVPair, SampleArgs},
};
use std::{slice::from_raw_parts_mut, usize};

type Worker<'w> = LlamaWorker<Operators, Weights<'w>>;

#[test]
fn test_infer() {
    let Some(shards) = test_utils::map_gguf_files() else {
        return;
    };
    let gguf = GGufModel::read(shards.iter().map(|s| &**s));

    let model = LlamaStorage::from_gguf(&gguf);
    let meta = &model.meta;
    println!("{meta:?}");

    let eos = gguf.tokenizer_ggml_eos_token_id().unwrap();
    let tokenizer = gguf.tokenizer();

    let gpu = match cuda::init() {
        Ok(()) => Device::new(0),
        Err(NoDevice) => return,
    };
    let gpu = Gpu::new(gpu.context(), Config::default());
    let gpu = &gpu;

    std::thread::scope(|s| {
        let sample = s.spawn(move || {
            let mut sample = RandomSample::new(gpu);
            sample.scheme(meta.dt_embd, meta.nvoc).unwrap();
            sample
        });
        gpu.apply(|ctx| {
            let stream = ctx.stream();
            let weights = Weights::new(&model, .., 1, usize::MAX, &stream);
            let token_embd = stream.from_host(model.token_embd);
            let mut worker = Worker::new(&gpu, meta.clone(), weights, true);
            let mut cache = meta
                .kv_cache(meta.nctx)
                .map(|size| stream.malloc::<u8>(size));
            let sample = sample.join().unwrap();

            let &LlamaMeta {
                dt_embd,
                nctx,
                nvoc,
                dh,
                ..
            } = meta;
            let sin_cos =
                <Operators as llama::Operators>::build_sin_cos(dt_embd, nctx, dh, &stream);
            let indices = RandomSample::build_indices(nvoc, &stream);

            test_utils::test_infer(eos, tokenizer, "Once upon a time,", |input, pos| {
                let mut embd = meta.embd(input.len()).map(|len| stream.malloc::<u8>(len));
                let mut logits = meta.logits(1).map(|len| stream.malloc::<u8>(len));

                let d = embd.get().len() / input.len();
                for (i, &tok) in input.iter().enumerate() {
                    stream.memcpy_d2d(
                        &mut embd.get_mut()[i * d..][..d],
                        &token_embd[tok as usize * d..][..d],
                    )
                }

                worker
                    .launch(
                        LlamaArgs {
                            embd: embd.map_slice_mut(),
                            logits: logits.map_slice_mut(),
                            sin_cos: sin_cos.map_slice(),
                            requests: vec![LlamaRequest {
                                cache: cache.map_slice_mut(),
                                seq_len: input.len(),
                                out_len: 1,
                                pos,
                            }],
                            num_tokens: input.len(),
                            max_seq_len: input.len(),
                            max_att_len: pos + input.len(),
                        },
                        &mut [],
                        &stream,
                    )
                    .unwrap();

                let mut pairs = Tensor::kv_pair_vec(1, |size| stream.malloc::<u8>(size));

                sample
                    .launch(
                        &mut pairs,
                        &logits,
                        &indices,
                        SampleArgs::ARG_MAX,
                        &mut [],
                        &stream,
                    )
                    .unwrap();

                let mut pair = KVPair::new(0, f16::ZERO);
                memcpy_d2h(
                    unsafe {
                        from_raw_parts_mut(&mut pair as *mut _ as *mut u8, size_of_val(&pair))
                    },
                    pairs.get(),
                );

                pair.idx() as _
            });
        });
    });
}
