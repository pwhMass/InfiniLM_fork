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

#[derive(Clone)]
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

impl<T> Storage<T> {
    pub fn map<U>(self, mut f: impl FnMut(T) -> U) -> Storage<U> {
        Storage {
            meta: self.meta,
            token_embed: f(self.token_embed),
            output_norm: f(self.output_norm),
            output: f(self.output),
            blocks: self
                .blocks
                .into_vec()
                .into_iter()
                .map(|blk| blk.map(&mut f))
                .collect(),
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
