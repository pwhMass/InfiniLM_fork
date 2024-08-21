use causal_lm::{CausalLM, DecodingMeta, Model, QueryContext, SampleMeta};
use common::{f16, map_files, upos, utok, Blob, FileLoadError, GGufModel};
use common_cpu::{
    tensor::{reslice, slice, udim, Tensor},
    CpuKernels, Kernels, KernelsA, KernelsB, ThisThread,
};
use digit_layout::types::{F16, F32};
use llama::{
    new::{duplicate_cache, LlamaMeta, LlamaModel},
    ComputeConst, ComputeStream, Handle, QueueOf, SliceOn,
};
use memmap2::Mmap;
use std::{iter::repeat, ops::Deref, path::Path, slice::from_raw_parts, sync::Arc};

pub struct Transformer {
    model: LlamaModel<OwnedSlice>,
    kernels: CpuKernels,
}

#[derive(Clone)]
struct OwnedSlice {
    _map: Arc<[Mmap]>,
    slice: &'static [u8],
}

impl Deref for OwnedSlice {
    type Target = [u8];
    #[inline]
    fn deref(&self) -> &Self::Target {
        self.slice
    }
}

impl Model for Transformer {
    type Meta = ();
    type Error = FileLoadError;

    #[inline]
    fn load(gguf: impl AsRef<Path>, _meta: Self::Meta) -> Result<Self, Self::Error> {
        let files: Arc<[Mmap]> = Arc::from(map_files(gguf));
        let gguf = GGufModel::read(files.iter().map(|f| &**f));

        Ok(Self {
            model: LlamaModel::from_gguf(&gguf, |s| OwnedSlice {
                _map: files.clone(),
                slice: unsafe { from_raw_parts(s.as_ptr(), s.len()) },
            }),
            kernels: Default::default(),
        })
    }
}

impl ComputeStream for Transformer {
    type Handle = common_cpu::Cpu;
    type Storage = Blob;
    type Buf<'m> = Blob;
    type Pos<'m> = &'m [u8];

    #[inline]
    fn malloc(&self, len: usize) -> Self::Buf<'_> {
        Blob::new(len)
    }
    #[inline]
    fn map_pos<'p>(&self, pos: &'p [u32]) -> Self::Pos<'p>
    where
        Self: 'p,
    {
        reslice(pos)
    }
    #[inline]
    fn map_storage<'a>(&'a self, storage: &'a mut Self::Storage) -> &'a mut SliceOn<Self::Handle> {
        storage
    }
    #[inline]
    fn kernels(&self) -> &impl Kernels<Self::Handle> {
        &self.kernels
    }
    #[inline]
    fn queue(&self) -> &QueueOf<Self::Handle> {
        &ThisThread
    }
    #[inline]
    fn constant(&self) -> ComputeConst {
        let meta = &self.model.meta;
        ComputeConst {
            nh: meta.nh as _,
            nkvh: meta.nkvh as _,
            di: meta.di as _,
            epsilon: meta.epsilon,
            theta: meta.theta,
        }
    }

    fn debug<T>(tensor: &Tensor<T>)
    where
        T: Deref<Target = SliceOn<Self::Handle>>,
    {
        println!("{tensor}");
    }

    #[inline]
    fn layers(
        &self,
    ) -> impl Iterator<Item = impl llama::LLamaLayer<Byte = <Self::Handle as Handle>::Byte>> {
        (0..self.model.meta.nblk).map(|i| LlamaLayer(&self.model, i))
    }
}

struct LlamaLayer<'a>(&'a LlamaModel<OwnedSlice>, usize);

impl<'a> llama::LLamaLayer for LlamaLayer<'a> {
    type Byte = u8;
    type Storage<'m> = &'m [u8] where Self: 'm;

    #[inline]
    fn att_layernorm(&self) -> Tensor<Self::Storage<'_>> {
        let LlamaMeta { nh, dh, .. } = self.0.meta;
        Tensor::new(F32, &[(nh * dh) as _], &self.0.blocks[self.1].attn_norm)
    }
    #[inline]
    fn att_qkv(&self) -> Tensor<Self::Storage<'_>> {
        let LlamaMeta { nh, nkvh, dh, .. } = self.0.meta;
        Tensor::new(
            F16,
            &[((nh + nkvh + nkvh) * dh) as _, (nh * dh) as _],
            &*self.0.blocks[self.1].attn_qkv,
        )
        .transpose(&[1, 0])
    }
    #[inline]
    fn att_o(&self) -> Tensor<Self::Storage<'_>> {
        let LlamaMeta { nh, dh, .. } = self.0.meta;
        Tensor::new(
            F16,
            &[(nh * dh) as _, (nh * dh) as _],
            &*self.0.blocks[self.1].attn_o,
        )
        .transpose(&[1, 0])
    }
    #[inline]
    fn mlp_layernorm(&self) -> Tensor<Self::Storage<'_>> {
        let LlamaMeta { nh, dh, .. } = self.0.meta;
        Tensor::new(F32, &[(nh * dh) as _], &self.0.blocks[self.1].ffn_norm)
    }
    #[inline]
    fn mlp_gate_up(&self) -> Tensor<Self::Storage<'_>> {
        let LlamaMeta { nh, dh, di, .. } = self.0.meta;
        Tensor::new(
            F16,
            &[(di + di) as _, (nh * dh) as _],
            &*self.0.blocks[self.1].ffn_gate_up,
        )
        .transpose(&[1, 0])
    }
    #[inline]
    fn mlp_down(&self) -> Tensor<Self::Storage<'_>> {
        let LlamaMeta { nh, dh, di, .. } = self.0.meta;
        Tensor::new(
            F16,
            &[(nh * dh) as _, di as _],
            &*self.0.blocks[self.1].ffn_down,
        )
        .transpose(&[1, 0])
    }
}

impl CausalLM for Transformer {
    type Storage = Blob;

    #[inline]
    fn max_seq_len(&self) -> upos {
        self.model.meta.dctx as _
    }
    #[inline]
    fn bos_token(&self) -> utok {
        1
    }
    #[inline]
    fn eos_token(&self) -> utok {
        2
    }
    #[inline]
    fn new_cache(&self) -> Tensor<Self::Storage> {
        self.model.meta.new_cache(Blob::new)
    }
    #[inline]
    fn duplicate_cache(&self, cache: &Tensor<Self::Storage>, pos: upos) -> Tensor<Self::Storage> {
        duplicate_cache(cache, pos, Blob::new, |dst, src| {
            src.map_physical(|u| &**u)
                .reform_to(&mut dst.map_physical(|u| &mut **u))
        })
    }

    fn token_embed(&self, queries: impl IntoIterator<Item = utok>) -> Tensor<Self::Storage> {
        let LlamaMeta { nh, dh, dvoc, .. } = self.model.meta;
        let d = (nh * dh) as udim;

        let tokens = queries.into_iter().collect::<Vec<_>>();
        let nt = tokens.len() as udim;

        let mut x = Tensor::alloc(F16, &[nt, d], Blob::new);
        let token_embed = Tensor::new(F16, &[dvoc as _, d], &*self.model.token_embed);
        self.kernels
            .gather(&mut x, &token_embed, tokens, &ThisThread);
        x
    }

    fn forward<'a>(
        &self,
        queries: impl IntoIterator<Item = QueryContext<'a, Self::Storage>>,
        token_embedded: Tensor<Self::Storage>,
    ) -> Tensor<Self::Storage> {
        <Self as ComputeStream>::forward(self, queries, token_embedded)
    }

    fn decode(
        &self,
        decoding: impl IntoIterator<Item = DecodingMeta>,
        hidden_state: Tensor<Self::Storage>,
    ) -> Tensor<Self::Storage> {
        let LlamaMeta {
            nh,
            dh,
            dvoc,
            epsilon,
            ..
        } = self.model.meta;
        let d = (nh * dh) as udim;

        let mut x = hidden_state;
        let range = DecodingMeta::select(&mut x, decoding, |dst, src| dst.copy_from_slice(src));

        if range.is_empty() {
            return Tensor::alloc(F16, &[0, d as _], Blob::new);
        }

        let lm_layernorm = Tensor::new(F32, &[d], &*self.model.output_norm);
        let lm_head = Tensor::new(F16, &[dvoc as _, d], &*self.model.output).transpose(&[1, 0]);
        let mut x = x.slice(&[slice![range.start => range.end], slice![=>]]);
        let mut logits = Tensor::alloc(F16, &[x.shape()[0], lm_head.shape()[1]], Blob::new);

        // 复制一个 x 以实现原地归一化
        let x_ = x
            .as_ref()
            .map_physical(|u| unsafe { from_raw_parts(u.as_ptr(), u.len()) });
        self.kernels()
            .rms_norm(&mut x, &x_, &lm_layernorm, epsilon, self.queue());
        self.kernels()
            .mat_mul(&mut logits, 0., &x, &lm_head, 1., self.queue());

        logits
    }

    fn sample(
        &self,
        args: impl IntoIterator<Item = SampleMeta>,
        logits: Tensor<Self::Storage>,
    ) -> Vec<utok> {
        let &[_, voc] = logits.shape() else { panic!() };
        let logits: &[f16] = reslice(logits.as_slice());
        args.into_iter()
            .flat_map(|meta| repeat(meta.args).take(meta.num_decode))
            .enumerate()
            .map(|(i, args)| {
                self.kernels.sample(
                    args.temperature,
                    args.top_p,
                    args.top_k,
                    &common_cpu::slice!(logits; voc; [i]),
                )
            })
            .collect()
    }
}

#[test]
fn test_infer() {
    causal_lm::test_impl::<Transformer>(
        (),
        &[
            29966, 29989, 1792, 29989, 29958, 13, 29903, 388, 376, 18567, 29908, 304, 592, 21106,
            29879, 5299, 29989, 465, 22137, 29989, 29958, 13,
        ],
    );
}
