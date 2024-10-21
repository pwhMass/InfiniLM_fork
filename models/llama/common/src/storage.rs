use crate::{normalize, LlamaMeta};
use common::{borrow, own, Contiguous};
use gguf::{GGufMetaError, GGufMetaMapExt, GGufModel};
use std::ops::{DerefMut, RangeBounds};
use tensor::{rearrange, split, Tensor};

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

impl<'w> BlkStorage<&'w [u8]> {
    pub fn distribute<U>(
        &self,
        meta: &LlamaMeta,
        range: impl RangeBounds<usize>,
        count: usize,
        mut f: impl FnMut(usize) -> U,
    ) -> BlkStorage<Contiguous<'w, U>>
    where
        U: DerefMut<Target = [u8]>,
    {
        let range = normalize(range, count);
        let start = range.start;
        let len = range.len();
        assert!(0 < len && len <= count);

        let &LlamaMeta {
            nh, nkvh, dh, di, ..
        } = meta;
        assert_eq!(nkvh % count, 0);
        assert_eq!(di % count, 0);

        use crate::TensorUsage::Storage;
        BlkStorage {
            attn_norm: borrow(self.attn_norm),
            attn_qkv: if len == count {
                borrow(self.attn_qkv)
            } else {
                let t = meta
                    .attn_qkv(Storage)
                    .map(|_| self.attn_qkv)
                    .tile(0, &[(nh + nkvh + nkvh), dh]);
                split!(t => q, k, v; [nh, nkvh, nkvh] @ 0);

                let p = nh / count;
                let q = q.slice(0, p * start, 1, p * len);
                let p = nkvh / count;
                let k = k.slice(0, p * start, 1, p * len);
                let v = v.slice(0, p * start, 1, p * len);
                assert!(q.is_contiguous() && k.is_contiguous() && v.is_contiguous());

                let q_ = Tensor::new(q.dt(), q.shape());
                let k_ = Tensor::new(k.dt(), k.shape());
                let v_ = Tensor::new(v.dt(), v.shape());
                let mut ans = f(q_.get() + k_.get() + v_.get());

                let qs = 0;
                let ks = qs + k_.get();
                let vs = ks + k_.get();
                rearrange(&mut q_.map(|len| &mut ans[qs..len]), &q);
                rearrange(&mut k_.map(|len| &mut ans[ks..len]), &k);
                rearrange(&mut v_.map(|len| &mut ans[vs..len]), &v);

                own(ans)
            },
            attn_o: if len == count {
                borrow(self.attn_o)
            } else {
                let t = meta.attn_o(Storage).map(|_| self.attn_o).tile(1, &[nh, dh]);

                let p = nh / count;
                let t = t.slice(1, p * start, 1, p * len);

                let mut t_ = Tensor::new(t.dt(), t.shape()).map(&mut f);
                rearrange(&mut t_, &t);

                own(t_.take())
            },
            ffn_norm: borrow(self.ffn_norm),
            ffn_gate_up: if len == count {
                borrow(self.ffn_gate_up)
            } else {
                let t = meta.ffn_gate_up(Storage).map(|_| self.ffn_gate_up);
                split!(t => g, u; [di, di] @ 0);

                let p = di / count;
                let g = g.slice(0, p * start, 1, p * len);
                let u = u.slice(0, p * start, 1, p * len);
                assert!(g.is_contiguous() && u.is_contiguous());

                let g_ = Tensor::new(g.dt(), g.shape());
                let u_ = Tensor::new(u.dt(), u.shape());
                let mut ans = f(g_.get() + u_.get());

                rearrange(&mut g_.map(|len| &mut ans[..len]), &g);
                rearrange(&mut u_.map(|len| &mut ans[len..]), &u);

                own(ans)
            },
            ffn_down: if len == count {
                borrow(self.ffn_down)
            } else {
                let t = meta.ffn_down(Storage).map(|_| self.ffn_down);

                let p = di / count;
                let t = t.slice(1, p * start, 1, p * len);

                let mut t_ = Tensor::new(t.dt(), t.shape()).map(&mut f);
                rearrange(&mut t_, &t);

                own(t_.take())
            },
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
