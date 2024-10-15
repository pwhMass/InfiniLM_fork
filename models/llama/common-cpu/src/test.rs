use crate::{Operators, RandomSample, Weights};
use gguf::{GGufMetaMapExt, GGufModel};
use llama::{ext::f16, LlamaArgs, LlamaMeta, LlamaRequest, LlamaStorage, LlamaWorker, Tensor};
use operators::{
    common_cpu::{Cpu, ThisThread},
    random_sample::{KVPair, SampleArgs},
};
use std::slice::from_raw_parts_mut;
use test_utils::CausalLM;

#[test]
fn test_infer() {
    let Some(shards) = test_utils::map_gguf_files() else {
        return;
    };
    let gguf = GGufModel::read(shards.iter().map(|s| &**s));

    let model = LlamaStorage::from_gguf(&gguf);
    let weights = Weights::new(&model);
    let LlamaStorage {
        meta, token_embed, ..
    } = model;
    println!("{meta:?}");

    let eos = gguf.tokenizer_ggml_eos_token_id().unwrap();
    let tokenizer = gguf.tokenizer();

    let mut cache = meta.kv_cache(meta.nctx).map(|size| vec![0u8; size]).take();

    test_utils::test_infer(
        Llama {
            token_embed,
            worker: LlamaWorker::new(&Cpu, meta, weights),
            sample: RandomSample::new(&Cpu),
        },
        &mut cache,
        eos,
        tokenizer,
        "Once upon a time,",
    );
}

struct Llama<'w> {
    token_embed: &'w [u8],
    worker: LlamaWorker<Operators, Weights<'w>>,
    sample: RandomSample,
}

impl CausalLM<u8> for Llama<'_> {
    fn infer(&mut self, input: &[u32], cache: &mut [u8], pos: usize) -> u32 {
        let meta = self.worker.meta();
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

        let mut embd = meta.embd(input.len()).map(|size| vec![0u8; size]);
        let mut logits = meta.logits(1).map(|size| vec![0u8; size]);

        let d = embd.get().len() / input.len();
        for (i, &tok) in input.iter().enumerate() {
            embd.get_mut()[i * d..][..d]
                .copy_from_slice(&self.token_embed[tok as usize * d..][..d]);
        }

        self.worker
            .launch(
                LlamaArgs {
                    embd: embd.map_slice_mut(),
                    logits: logits.map_slice_mut(),
                    sin_cos: sin_cos.map_slice(),
                    requests: vec![LlamaRequest {
                        cache: meta.kv_cache(nctx).map(|_| cache),
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
                &ThisThread,
            )
            .unwrap();

        let mut pair = KVPair::new(0, f16::ZERO);
        let mut pairs = Tensor::kv_pair_vec(1, |_| unsafe {
            from_raw_parts_mut(&mut pair as *mut _ as _, size_of_val(&pair))
        });

        self.sample
            .launch(
                &mut pairs,
                &logits,
                &indices,
                SampleArgs::ARG_MAX,
                &mut [],
                &ThisThread,
            )
            .unwrap();

        pair.idx() as u32
    }
}
