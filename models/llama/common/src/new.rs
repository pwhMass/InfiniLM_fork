use common::{upos, GGufModel};
use digit_layout::{types::F16, DigitLayout};
use ggus::GGufMetaDataValueType;
use tensor::{slice, Tensor};

pub struct LlamaModel<T> {
    pub meta: LlamaMeta,
    pub token_embed: T,
    pub output_norm: T,
    pub output: T,
    pub blocks: Box<[LlamaBlk<T>]>,
}

impl<T> LlamaModel<T> {
    pub fn from_gguf<'a>(gguf: &GGufModel<'a>, mut f: impl FnMut(&'a [u8]) -> T) -> LlamaModel<T> {
        use GGufMetaDataValueType as T;

        let mut meta = LlamaMeta {
            dt_norm: F16,
            dt_mat: F16,
            nblk: 0,
            nh: 0,
            nkvh: 0,
            dh: 0,
            di: 0,
            dctx: 0,
            dvoc: 0,
            epsilon: 1e-5,
            theta: 1e4,
        };
        meta.dt_norm = gguf.tensors["output_norm.weight"].ty;
        meta.dt_norm = gguf.tensors["token_embd.weight"].ty;
        meta.nblk = {
            let kv = &gguf.meta_kvs["llama.block_count"];
            assert_eq!(kv.ty(), T::U32);
            kv.value_reader().read::<u32>().unwrap() as _
        };
        meta.nh = {
            let kv = &gguf.meta_kvs["llama.attention.head_count"];
            assert_eq!(kv.ty(), T::U32);
            kv.value_reader().read::<u32>().unwrap() as _
        };
        meta.nkvh = match &gguf.meta_kvs.get("llama.attention.head_count_kv") {
            Some(kv) => {
                assert_eq!(kv.ty(), T::U32);
                kv.value_reader().read::<u32>().unwrap() as _
            }
            None => meta.nh,
        };
        meta.dh = {
            let kv = &gguf.meta_kvs["llama.rope.dimension_count"];
            assert_eq!(kv.ty(), T::U32);
            kv.value_reader().read::<u32>().unwrap() as _
        };
        {
            let kv = &gguf.meta_kvs["llama.embedding_length"];
            assert_eq!(kv.ty(), T::U32);
            let d = kv.value_reader().read::<u32>().unwrap() as usize;
            assert_eq!(meta.nh * meta.dh, d);
        }
        meta.di = {
            let kv = &gguf.meta_kvs["llama.feed_forward_length"];
            assert_eq!(kv.ty(), T::U32);
            kv.value_reader().read::<u32>().unwrap() as _
        };
        meta.dctx = {
            let kv = &gguf.meta_kvs["llama.context_length"];
            assert_eq!(kv.ty(), T::U32);
            kv.value_reader().read::<u32>().unwrap() as _
        };
        meta.dvoc = {
            let kv = &gguf.meta_kvs["llama.vocab_size"];
            assert_eq!(kv.ty(), T::U32);
            kv.value_reader().read::<u32>().unwrap() as _
        };
        if let Some(kv) = gguf.meta_kvs.get("llama.attention.layer_norm_rms_epsilon") {
            assert_eq!(kv.ty(), T::F32);
            meta.epsilon = kv.value_reader().read().unwrap();
        };
        if let Some(kv) = gguf.meta_kvs.get("llama.rope.freq_base") {
            assert_eq!(kv.ty(), T::F32);
            meta.theta = kv.value_reader().read().unwrap();
        };

        let mut data = |name: &str| f(gguf.tensors[&*name].data);
        #[rustfmt::skip]
        let blocks = (0..meta.nblk)
            .map(|i| LlamaBlk {
                attn_norm:   data(&format!("blk.{i}.attn_norm.weight")),
                attn_qkv:    data(&format!("blk.{i}.attn_qkv.weight")),
                attn_o:      data(&format!("blk.{i}.attn_output.weight")),
                ffn_norm:    data(&format!("blk.{i}.ffn_norm.weight")),
                ffn_gate_up: data(&format!("blk.{i}.ffn_gate_up.weight")),
                ffn_down:    data(&format!("blk.{i}.ffn_down.weight")),
            })
            .collect();

        Self {
            meta,
            token_embed: data("token_embd.weight"),
            output_norm: data("output_norm.weight"),
            output: data("output.weight"),
            blocks,
        }
    }
}

pub struct LlamaMeta {
    pub dt_norm: DigitLayout,
    pub dt_mat: DigitLayout,
    pub nblk: usize,
    pub nh: usize,
    pub nkvh: usize,
    pub dh: usize,
    pub di: usize,
    pub dctx: usize,
    pub dvoc: usize,
    pub epsilon: f32,
    pub theta: f32,
}

impl LlamaMeta {
    pub fn new_cache<S>(&self, f: impl FnOnce(usize) -> S) -> Tensor<S> {
        Tensor::alloc(
            self.dt_mat,
            &[
                self.nblk as _,
                2,
                self.nkvh as _,
                self.dctx as _,
                self.dh as _,
            ],
            f,
        )
    }
}

pub fn duplicate_cache<S>(
    cache: &Tensor<S>,
    pos: upos,
    malloc: impl FnOnce(usize) -> S,
    reform: impl FnOnce(Tensor<&mut S>, Tensor<&S>),
) -> Tensor<S> {
    let mut ans = Tensor::alloc(cache.data_layout(), cache.shape(), malloc);
    if pos > 0 {
        let &[_nlayers, 2, _nkvh, max_seq_len, _dh] = cache.shape() else {
            panic!()
        };
        assert!(pos <= max_seq_len);
        let slice = [
            slice![=>],
            slice![=>],
            slice![=>],
            slice![=>pos],
            slice![=>],
        ];
        reform(ans.as_mut().slice(&slice), cache.as_ref().slice(&slice));
    }
    ans
}

pub struct LlamaBlk<T> {
    pub attn_norm: T,
    pub attn_qkv: T,
    pub attn_o: T,
    pub ffn_norm: T,
    pub ffn_gate_up: T,
    pub ffn_down: T,
}

impl<T> LlamaBlk<T> {
    pub fn as_ref<U>(&self) -> LlamaBlk<&T> {
        macro_rules! map {
            ($($ident:ident)+) => {
                LlamaBlk {$(
                    $ident: &self.$ident,
                )+}
            };
        }
        map! {
            attn_norm
            attn_qkv
            attn_o
            ffn_norm
            ffn_gate_up
            ffn_down
        }
    }

    pub fn map<U>(self, mut f: impl FnMut(T) -> U) -> LlamaBlk<U> {
        macro_rules! map {
            ($($ident:ident)+) => {
                LlamaBlk {$(
                    $ident: f(self.$ident),
                )+}
            };
        }
        map! {
            attn_norm
            attn_qkv
            attn_o
            ffn_norm
            ffn_gate_up
            ffn_down
        }
    }
}
