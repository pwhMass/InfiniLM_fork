#![cfg(detected_cuda)]

#[macro_use]
extern crate log;
pub extern crate cuda;

mod fused_softmax;
mod gather;
mod mat_mul;
mod reform;
mod rms_norm;
mod rotary_embedding;
mod swiglu;
mod paged_attention;

pub use common::utok;
pub use tensor::{slice, udim,reslice, DataType, LocalSplitable, Tensor};

use cublas::{Cublas, CublasSpore};
use cuda::{
    memcpy_d2h, ContextGuard, ContextResource, ContextSpore, CudaDataType::f16, CudaDataType::u16, DevByte,
    ModuleSpore, Ptx, Stream,
};
use fused_softmax::FusedSoftmax;
use reform::Reform;
use rms_norm::RmsNormalization;
use rotary_embedding::Rope;
use paged_attention::PagedAttention;
use std::{
    ops::{Deref, DerefMut},
    sync::Arc,
};
use swiglu::Swiglu;
use transformer::{Kernels, Llama2};

pub struct NvidiaKernelsPtx {
    epsilon: f32,
    theta: f32,
    rms_norm: Arc<RmsNormalization>,
    rotary_embedding: Arc<Rope>,
    reform: Arc<Reform>,
    softmax: Arc<FusedSoftmax>,
    swiglu: Arc<Swiglu>,
    paged_attention: Arc<PagedAttention>,
}

impl NvidiaKernelsPtx {
    pub fn new(host: &dyn Llama2, block_size: usize) -> Self {
        Self {
            epsilon: host.rms_norm_eps(),
            theta: host.rope_theta(),
            rms_norm: Arc::new(RmsNormalization::new(f16, host.hidden_size(), block_size)),
            rotary_embedding: Arc::new(Rope::new(block_size)),
            reform: Arc::new(Reform::new(block_size, 32)),
            softmax: Arc::new(FusedSoftmax::new(
                f16,
                host.max_position_embeddings(),
                block_size,
            )),
            swiglu: Arc::new(Swiglu::new(f16, block_size)),
            paged_attention: Arc::new(PagedAttention::new(host.num_attention_heads() as u32, 1, 128, 32, u16, u16)),
        }
    }
}

trait PtxWapper: Sized {
    fn ptx(&self) -> &Ptx;
    #[inline]
    fn load(self: Arc<Self>, ctx: &ContextGuard) -> ModuleWapper<Self> {
        ModuleWapper {
            module: ctx.load(self.ptx()).sporulate(),
            kernel: self,
        }
    }
}

struct ModuleWapper<T> {
    module: ModuleSpore,
    kernel: Arc<T>,
}

pub struct NvidiaKernels {
    epsilon: f32,
    theta: f32,
    cublas: CublasSpore,
    rms_norm: ModuleWapper<RmsNormalization>,
    rotary_embedding: ModuleWapper<Rope>,
    reform: ModuleWapper<Reform>,
    softmax: ModuleWapper<FusedSoftmax>,
    swiglu: ModuleWapper<Swiglu>,
    paged_attention: ModuleWapper<PagedAttention>,
}

impl NvidiaKernelsPtx {
    pub fn load(&self, ctx: &ContextGuard) -> NvidiaKernels {
        NvidiaKernels {
            epsilon: self.epsilon,
            theta: self.theta,
            cublas: Cublas::new(ctx).sporulate(),
            rms_norm: self.rms_norm.clone().load(ctx),
            rotary_embedding: self.rotary_embedding.clone().load(ctx),
            reform: self.reform.clone().load(ctx),
            softmax: self.softmax.clone().load(ctx),
            swiglu: self.swiglu.clone().load(ctx),
            paged_attention: self.paged_attention.clone().load(ctx),
        }
    }
}

impl NvidiaKernels {
    pub fn kill(&mut self, ctx: &ContextGuard) {
        unsafe {
            self.cublas.kill(ctx);
            self.rms_norm.module.kill(ctx);
            self.rotary_embedding.module.kill(ctx);
            self.reform.module.kill(ctx);
            self.softmax.module.kill(ctx);
            self.swiglu.module.kill(ctx);
            self.paged_attention.module.kill(ctx);
        }
    }
}

pub struct KernelRuntime<'a> {
    pub kernels: &'a NvidiaKernels,
    pub stream: &'a Stream<'a>,
}

impl NvidiaKernels {
    #[inline]
    pub fn on<'a>(&'a self, stream: &'a Stream) -> KernelRuntime<'a> {
        KernelRuntime {
            kernels: self,
            stream,
        }
    }
}

impl KernelRuntime<'_>{
    #[inline]
   pub fn paged_attention<OutT, QT, KT, VT, BlockTablesT, SeqLensT>(
        &self,
        out: &mut Tensor<OutT>,
        query: &Tensor<QT>,
        key_cache: &Tensor<KT>,
        value_cache: &Tensor<VT>,
        num_kv_heads: u32,
        scale: f32,
        block_tables: &Tensor<BlockTablesT>,
        seq_lens: &Tensor<SeqLensT>,
        max_seq_len: u32,
    ) where
    OutT: DerefMut<Target = [DevByte]>,
    QT: Deref<Target = [DevByte]>,
    KT: Deref<Target = [DevByte]>,
    VT: Deref<Target = [DevByte]>,
    BlockTablesT: Deref<Target = [DevByte]>,
    SeqLensT: Deref<Target = [DevByte]>,
    {
        let ModuleWapper { module, kernel } = &self.kernels.paged_attention;
        kernel.launch(module, out, query, key_cache, value_cache, num_kv_heads, scale, block_tables, seq_lens, max_seq_len, self.stream);
    }    
}

impl Kernels for KernelRuntime<'_> {
    type Storage = [DevByte];

    #[inline]
    fn gather<T, U, I>(&self, x: &mut Tensor<T>, table: &Tensor<U>, tokens: I)
    where
        T: DerefMut<Target = Self::Storage>,
        U: Deref<Target = [u8]>,
        I: IntoIterator<Item = utok>,
    {
        gather::gather(x, table, tokens, self.stream);
    }

    #[inline]
    fn rms_norm<T, U, V>(&self, y: &mut Tensor<T>, x: &Tensor<U>, w: &Tensor<V>)
    where
        T: DerefMut<Target = Self::Storage>,
        U: Deref<Target = Self::Storage>,
        V: Deref<Target = Self::Storage>,
    {
        let ModuleWapper { module, kernel } = &self.kernels.rms_norm;
        kernel.launch(module, y, x, w, self.kernels.epsilon, self.stream);
    }

    #[inline]
    fn mat_mul<T, U, V>(
        &self,
        c: &mut Tensor<T>,
        beta: f32,
        a: &Tensor<U>,
        b: &Tensor<V>,
        alpha: f32,
    ) where
        T: DerefMut<Target = Self::Storage>,
        U: Deref<Target = Self::Storage>,
        V: Deref<Target = Self::Storage>,
    {
        let cublas = unsafe { self.kernels.cublas.sprout(self.stream.ctx()) };
        cublas.set_stream(self.stream);
        mat_mul::mat_mul(&cublas, c, beta, a, b, alpha)
    }

    #[inline]
    fn rotary_embedding<T, U>(&self, t: &mut Tensor<T>, pos: &Tensor<U>)
    where
        T: DerefMut<Target = Self::Storage>,
        U: Deref<Target = Self::Storage>,
    {
        let ModuleWapper { module, kernel } = &self.kernels.rotary_embedding;
        kernel.launch(module, t, pos, self.kernels.theta, self.stream);
    }

    #[inline]
    fn reform<T, U>(&self, dst: &mut Tensor<T>, src: &Tensor<U>)
    where
        T: DerefMut<Target = Self::Storage>,
        U: Deref<Target = Self::Storage>,
    {
        let ModuleWapper { module, kernel } = &self.kernels.reform;
        kernel.launch(&module, dst, src, self.stream);
    }

    #[inline]
    fn softmax<T>(&self, att: &mut Tensor<T>)
    where
        T: DerefMut<Target = Self::Storage>,
    {
        let ModuleWapper { module, kernel } = &self.kernels.softmax;
        kernel.launch(&module, att, self.stream);
    }

    #[inline]
    fn swiglu<T, U>(&self, gate: &mut Tensor<T>, up: &Tensor<U>)
    where
        T: DerefMut<Target = Self::Storage>,
        U: Deref<Target = Self::Storage>,
    {
        let ModuleWapper { module, kernel } = &self.kernels.swiglu;
        kernel.launch(module, gate, up, self.stream);
    }

    
}

#[allow(unused)]
pub fn map_tensor<T>(tensor: &Tensor<T>) -> Tensor<Vec<u8>>
where
    T: Deref<Target = [DevByte]>,
{
    unsafe {
        tensor.as_ref().map_physical(|dev| {
            let mut buf = vec![0; dev.len()];
            memcpy_d2h(&mut buf, dev);
            buf
        })
    }
}
