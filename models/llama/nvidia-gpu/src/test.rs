use crate::{Operators, RandomSample, Weights};
use gguf::{GGufMetaMapExt, GGufModel};
use llama::{ext::f16, LlamaArgs, LlamaMeta, LlamaRequest, LlamaStorage, LlamaWorker, Tensor};
use operators::{
    cuda::{self, memcpy_d2h, DevByte, DevMem, Device, NoDevice, Stream},
    nvidia_gpu::{Config, Gpu},
    random_sample::{KVPair, SampleArgs},
};
use std::{slice::from_raw_parts_mut, usize};
use test_utils::CausalLM;

#[test]
fn test_infer() {
    let shards = match test_utils::map_gguf_files() {
        Some(shards) => shards,
        None => return,
    };
    let gpu = match cuda::init() {
        Ok(()) => Device::new(0),
        Err(NoDevice) => return,
    };

    let gguf = GGufModel::read(shards.iter().map(|s| &**s));
    let model = LlamaStorage::from_gguf(&gguf);
    let gpu = Gpu::new(gpu.context(), Config::default());
    let gpu = &gpu;

    std::thread::scope(|s| {
        let LlamaStorage {
            meta, token_embed, ..
        } = &model;

        let sample = s.spawn(move || {
            let mut sample = RandomSample::new(gpu);
            sample.scheme(meta.dt_embd, meta.nvoc).unwrap();
            sample
        });
        gpu.apply(|ctx| {
            let stream = ctx.stream();
            let weights = Weights::new(&model, usize::MAX, &stream);

            println!("{meta:?}");
            let &LlamaMeta {
                dt_embd,
                nctx,
                nvoc,
                dh,
                ..
            } = meta;

            let eos = gguf.tokenizer_ggml_eos_token_id().unwrap();
            let tokenizer = gguf.tokenizer();

            let mut cache: DevMem<'_> = meta
                .kv_cache(nctx)
                .map(|size| stream.malloc::<u8>(size))
                .take();

            test_utils::test_infer(
                Llama {
                    token_embed: stream.from_host(token_embed),
                    worker: LlamaWorker::new(gpu, meta.clone(), weights),
                    sample: sample.join().unwrap(),

                    sin_cos: <Operators as llama::Operators>::build_sin_cos(
                        dt_embd, nctx, dh, &stream,
                    ),
                    indices: RandomSample::build_indices(nvoc, &stream),
                    stream,
                },
                &mut cache,
                eos,
                tokenizer,
                "Once upon a time,",
            );
        });
    });
}

struct Llama<'ctx> {
    token_embed: DevMem<'ctx>,
    worker: LlamaWorker<Operators, Weights<'ctx>>,
    sample: RandomSample,

    sin_cos: Tensor<DevMem<'ctx>>,
    indices: Tensor<DevMem<'ctx>>,
    stream: Stream<'ctx>,
}

impl CausalLM<DevByte> for Llama<'_> {
    fn infer(&mut self, input: &[u32], cache: &mut [DevByte], pos: usize) -> u32 {
        let meta = self.worker.meta();

        let mut embd = meta
            .embd(input.len())
            .map(|len| self.stream.malloc::<u8>(len));
        let mut logits = meta.logits(1).map(|len| self.stream.malloc::<u8>(len));

        let d = embd.get().len() / input.len();
        for (i, &tok) in input.iter().enumerate() {
            self.stream.memcpy_d2d(
                &mut embd.get_mut()[i * d..][..d],
                &self.token_embed[tok as usize * d..][..d],
            )
        }

        self.worker
            .launch(
                LlamaArgs {
                    embd: embd.map_slice_mut(),
                    logits: logits.map_slice_mut(),
                    sin_cos: self.sin_cos.map_slice(),
                    requests: vec![LlamaRequest {
                        cache: meta.kv_cache(meta.nctx).map(|_| cache),
                        seq_len: input.len(),
                        out_len: 1,
                        pos,
                    }],
                    num_tokens: input.len(),
                    max_seq_len: input.len(),
                    max_att_len: pos + input.len(),
                    mlp_alpha: 1.,
                },
                &mut [],
                &self.stream,
            )
            .unwrap();

        let mut pairs = Tensor::kv_pair_vec(1, |size| self.stream.malloc::<u8>(size));

        self.sample
            .launch(
                &mut pairs,
                &logits,
                &self.indices,
                SampleArgs::ARG_MAX,
                &mut [],
                &self.stream,
            )
            .unwrap();

        let mut pair = KVPair::new(0, f16::ZERO);
        memcpy_d2h(
            unsafe { from_raw_parts_mut(&mut pair as *mut _ as *mut u8, size_of_val(&pair)) },
            pairs.get(),
        );

        pair.idx() as u32
    }
}
