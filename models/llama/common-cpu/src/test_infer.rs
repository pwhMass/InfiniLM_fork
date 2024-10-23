use crate::{Operators, RandomSample, Weights};
use gguf::{GGufMetaMapExt, GGufModel};
use llama::{ext::f16, LlamaArgs, LlamaMeta, LlamaRequest, LlamaStorage, LlamaWorker, Tensor};
use operators::{
    common_cpu::{Blob, Cpu, ThisThread},
    random_sample::{KVPair, SampleArgs},
};
use std::slice::from_raw_parts_mut;

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

    let weights = Weights::new(&model, .., 1);
    let mut worker = Worker::new(&Cpu, meta.clone(), weights, true);
    let mut cache = meta.kv_cache(meta.nctx).map(Blob::new);
    let sample = RandomSample::new(&Cpu);

    test_utils::test_infer(eos, tokenizer, "Once upon a time,", |input, pos| {
        let &LlamaMeta {
            dt_embd,
            nctx,
            nvoc,
            dh,
            ..
        } = meta;

        let sin_cos =
            <Operators as llama::Operators>::build_sin_cos(dt_embd, nctx, dh, &ThisThread);
        let indices = RandomSample::build_indices(nvoc, &ThisThread);

        let mut embd = meta.embd(input.len()).map(Blob::new);
        let mut logits = meta.logits(1).map(Blob::new);

        let d = embd.get().len() / input.len();
        for (i, &tok) in input.iter().enumerate() {
            embd.get_mut()[i * d..][..d]
                .copy_from_slice(&model.token_embd[tok as usize * d..][..d]);
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
                &ThisThread,
            )
            .unwrap();

        let mut pair = KVPair::new(0, f16::ZERO);
        let mut pairs = Tensor::kv_pair_vec(1, |_| unsafe {
            from_raw_parts_mut(&mut pair as *mut _ as _, size_of_val(&pair))
        });

        sample
            .launch(
                &mut pairs,
                &logits,
                &indices,
                SampleArgs::ARG_MAX,
                &mut [],
                &ThisThread,
            )
            .unwrap();

        pair.idx() as _
    });
}
