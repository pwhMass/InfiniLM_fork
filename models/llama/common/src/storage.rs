use crate::LlamaMeta;
use gguf::{GGufMetaError, GGufMetaMapExt, GGufModel};

#[derive(Clone)]
pub struct Storage<T> {
    pub meta: LlamaMeta,
    pub token_embed: T,
    pub output_norm: T,
    pub output: T,
    pub blocks: Box<[BlkStorage<T>]>,
}

#[derive(Clone, Copy)]
pub struct BlkStorage<T> {
    pub attn_norm: T,
    pub attn_qkv: T,
    pub attn_o: T,
    pub ffn_norm: T,
    pub ffn_gate_up: T,
    pub ffn_down: T,
}

impl<'a> Storage<&'a [u8]> {
    pub fn from_gguf(gguf: &GGufModel<'a>) -> Self {
        #[rustfmt::skip]
        let meta = LlamaMeta {
            dt_embd: gguf.tensors[ "token_embd.weight"].ty,
            dt_norm: gguf.tensors["output_norm.weight"].ty,
            dt_mat : gguf.tensors[     "output.weight"].ty,

            nblk: gguf.llm_block_count            ().unwrap(),
            nctx: gguf.llm_context_length         ().unwrap(),
            nvoc: gguf.tokenizer_ggml_tokens      ().unwrap().len(),
            nh  : gguf.llm_attention_head_count   ().unwrap(),
            nkvh: gguf.llm_attention_head_count_kv().unwrap(),
            d   : gguf.llm_embedding_length       ().unwrap(),
            dh  : gguf.llm_rope_dimension_count   ().unwrap(),
            di  : gguf.llm_feed_forward_length    ().unwrap(),

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
        };

        #[rustfmt::skip]
        let blocks = (0..meta.nblk)
            .map(|i| BlkStorage {
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

impl<T> BlkStorage<T> {
    pub fn map<U>(self, mut f: impl FnMut(T) -> U) -> BlkStorage<U> {
        BlkStorage {
            attn_norm: f(self.attn_norm),
            attn_qkv: f(self.attn_qkv),
            attn_o: f(self.attn_o),
            ffn_norm: f(self.ffn_norm),
            ffn_gate_up: f(self.ffn_gate_up),
            ffn_down: f(self.ffn_down),
        }
    }

    pub fn as_ref(&self) -> BlkStorage<&T> {
        BlkStorage {
            attn_norm: &self.attn_norm,
            attn_qkv: &self.attn_qkv,
            attn_o: &self.attn_o,
            ffn_norm: &self.ffn_norm,
            ffn_gate_up: &self.ffn_gate_up,
            ffn_down: &self.ffn_down,
        }
    }
}

#[test]
fn test_load() {
    let Some(shards) = test_utils::map_gguf_files() else {
        return;
    };
    let gguf = GGufModel::read(shards.iter().map(|s| &**s));
    let llama = Storage::from_gguf(&gguf);
    println!("{:?}", llama.meta);
}
