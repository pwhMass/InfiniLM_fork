use crate::{Operators, RandomSample, Weights};
use gguf::{GGufMetaMapExt, GGufModel};
use llama::{ext::f16, LlamaArgs, LlamaMeta, LlamaRequest, LlamaStorage, LlamaWorker, Tensor};
use operators::{
    cuda::{self, memcpy_d2h, DevByte, DevMem, Device, NoDevice, Stream},
    nvidia_gpu::{Config, Gpu},
    random_sample::{KVPair, SampleArgs},
};
use std::{
    slice::from_raw_parts_mut,
    time::{Duration, Instant},
    usize,
};
use test_utils::print_now;

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
    assert_eq!(model.meta.distribute, 1);

    let gpu = Gpu::new(gpu.context(), Config::default());
    gpu.apply(|ctx| {
        let stream = ctx.stream();
        let weights = Weights::new(&model, 0, 1, usize::MAX, &stream);

        let LlamaStorage {
            meta, token_embed, ..
        } = model;
        println!("{meta:?}");
        let mut llama = Llama {
            token_embed: stream.from_host(token_embed),
            worker: LlamaWorker::new(&gpu, meta.clone(), weights),
            sample: RandomSample::new(&gpu),
        };
        llama.sample.scheme(meta.dt_embd, meta.nvoc).unwrap();

        let eos = gguf.tokenizer_ggml_eos_token_id().unwrap();
        let tokenizer = gguf.tokenizer();

        let meta = llama.worker.meta();
        let mut cache = meta
            .kv_cache(meta.nctx)
            .map(|size| stream.malloc::<u8>(size))
            .take();

        let mut prompt = "Once upon a time,".to_string();
        print_now!("{prompt}");

        let mut tokens = tokenizer.encode(&prompt);
        let num_prompt_tokens = tokens.len();

        let mut prefill = Duration::ZERO;
        let mut decode = Duration::ZERO;

        let mut pos = 0;
        loop {
            let time = Instant::now();
            let next = llama.infer(&tokens, &mut cache, pos, &stream);
            let time = time.elapsed();

            if prefill.is_zero() {
                prefill = time;
            } else {
                decode += time;
            }

            pos += tokens.len();
            if next == eos {
                break;
            }

            let piece = tokenizer.decode(next);
            print_now!("{piece}");
            prompt.push_str(&piece);
            tokens = vec![next];
        }

        println!();
        println!();
        print_time("total", prefill + decode, pos);
        print_time("prefill", prefill, num_prompt_tokens);
        print_time("decode", decode, pos - num_prompt_tokens);
    });

    fn print_time(name: &str, time: Duration, n: usize) {
        println!(
            "{name}: {time:?} for {n} tokens, avg: {:?} per token",
            time.div_f64(n as _)
        )
    }
}

struct Llama<'ctx> {
    token_embed: DevMem<'ctx>,
    worker: LlamaWorker<Operators, Weights<'ctx>>,
    sample: RandomSample,
}

impl Llama<'_> {
    pub fn infer(
        &mut self,
        input: &[u32],
        cache: &mut [DevByte],
        pos: usize,
        stream: &Stream,
    ) -> u32 {
        let meta = self.worker.meta();
        let &LlamaMeta {
            dt_embd,
            nctx,
            nvoc,
            dh,
            ..
        } = meta;

        let sin_cos = <Operators as llama::Operators>::build_sin_cos(dt_embd, nctx, dh, stream);
        let indices = RandomSample::build_indices(nvoc, stream);

        let cache = meta.kv_cache(nctx).map(|_| cache);
        let mut embd = meta.embd(input.len()).map(|size| stream.malloc::<u8>(size));
        let mut logits = meta.logits(1).map(|size| stream.malloc::<u8>(size));

        let d = embd.get().len() / input.len();
        for (i, &tok) in input.iter().enumerate() {
            stream.memcpy_d2d(
                &mut embd.get_mut()[i * d..][..d],
                &self.token_embed[tok as usize * d..][..d],
            );
        }

        self.worker
            .launch(
                LlamaArgs {
                    embd: embd.map_slice_mut(),
                    logits: logits.map_slice_mut(),
                    sin_cos: sin_cos.map_slice(),
                    requests: vec![LlamaRequest {
                        cache,
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
                stream,
            )
            .unwrap();

        let mut pairs = Tensor::kv_pair_vec(1, |size| stream.malloc::<u8>(size));

        self.sample
            .launch(
                &mut pairs,
                &logits,
                &indices,
                SampleArgs::ARG_MAX,
                &mut [],
                stream,
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
