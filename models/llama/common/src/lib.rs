mod args;
mod compute;

use gguf::GGufModel;
use ggus::{ggml_quants::digit_layout::DigitLayout, GGufMetaError, GGufMetaMapExt};

pub use compute::LlamaBlks;

#[derive(Clone)]
pub struct LlamaModel<T> {
    pub meta: LlamaMeta,
    pub token_embed: T,
    pub output_norm: T,
    pub output: T,
    pub blocks: Box<[LlamaBlk<T>]>,
}

#[derive(Clone, Debug)]
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
    pub distribute: usize,
}

#[derive(Clone)]
pub struct LlamaBlk<T> {
    pub attn_norm: T,
    pub attn_qkv: T,
    pub attn_o: T,
    pub ffn_norm: T,
    pub ffn_gate_up: T,
    pub ffn_down: T,
}

impl<'a> LlamaModel<&'a [u8]> {
    pub fn from_gguf(gguf: &GGufModel<'a>) -> Self {
        #[rustfmt::skip]
        let meta = LlamaMeta {
            dt_norm: gguf.tensors["output_norm.weight"].ty,
            dt_mat : gguf.tensors[ "token_embd.weight"].ty,
            nblk   : gguf.llm_block_count().unwrap   (),
            nh     : gguf.llm_attention_head_count   ().unwrap(),
            nkvh   : gguf.llm_attention_head_count_kv().unwrap(),
            dh     : gguf.llm_rope_dimension_count   ().unwrap(),
            di     : gguf.llm_feed_forward_length    ().unwrap(),
            dctx   : gguf.llm_context_length         ().unwrap(),
            dvoc   : gguf.tokenizer_ggml_tokens      ().unwrap().len(),
            epsilon: match gguf.llm_attention_layer_norm_rms_epsilon() {
                Ok(val) => val,
                Err(GGufMetaError::NotExist) => 1e-5,
                Err(e) => panic!("failed to read meta: {e:?}"),
            },
            theta  : match gguf.llm_rope_freq_base() {
                Ok(val) => val,
                Err(GGufMetaError::NotExist) => 1e4,
                Err(e) => panic!("failed to read meta: {e:?}"),
            },
            distribute: match gguf.get_usize("llama.distrubute") {
                Ok(val) => val,
                Err(GGufMetaError::NotExist) => 1,
                Err(e) => panic!("failed to read meta: {e:?}"),
            },
        };
        assert_eq!(meta.nh * meta.dh, gguf.llm_embedding_length().unwrap());

        #[rustfmt::skip]
        let blocks = (0..meta.nblk)
            .map(|i| LlamaBlk {
                attn_norm:   gguf.tensors[&*format!("blk.{i}.attn_norm.weight"  )].data,
                attn_qkv:    gguf.tensors[&*format!("blk.{i}.attn_qkv.weight"   )].data,
                attn_o:      gguf.tensors[&*format!("blk.{i}.attn_output.weight")].data,
                ffn_norm:    gguf.tensors[&*format!("blk.{i}.ffn_norm.weight"   )].data,
                ffn_gate_up: gguf.tensors[&*format!("blk.{i}.ffn_gate_up.weight")].data,
                ffn_down:    gguf.tensors[&*format!("blk.{i}.ffn_down.weight"   )].data,
            })
            .collect();

        Self {
            meta,
            token_embed: gguf.tensors["token_embd.weight"].data,
            output_norm: gguf.tensors["output_norm.weight"].data,
            output: gguf.tensors["output.weight"].data,
            blocks,
        }
    }
}

#[test]
fn test_load() {
    let Some(shards) = test_utils::map_gguf_files() else {
        return;
    };
    let gguf = GGufModel::read(shards.iter().map(|s| &**s));
    let llama = LlamaModel::from_gguf(&gguf);
    println!("{:?}", llama.meta);
}
