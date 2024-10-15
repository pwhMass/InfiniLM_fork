#![cfg(hw_detected)]

use llama::{BlkWeight, LlamaBlkStorage, LlamaStorage, Tensor, WeightLoader};
use operators::{
    all_reduce::{AllReduce, NonAllReduce},
    cuda::{memcpy_d2h, DevByte, DevMem, Event, HostMem, Stream},
    nvidia_gpu::Gpu,
    random_sample::nvidia_gpu::Operator as RandomSampleGpu,
    ByteOf, QueueOf, TopoNode,
};
use std::{marker::PhantomData, mem::replace, ops::Deref};

pub struct Operators<N = Gpu, R = NonAllReduce<Gpu>>(PhantomData<(N, R)>);

pub type RandomSample = llama::RandomSample<Gpu, RandomSampleGpu>;

pub struct Weights<'ctx> {
    blks: Box<[LlamaBlkStorage<DevMem<'ctx>>]>,
    #[allow(dead_code)]
    blk_source: Box<[LlamaStorage<HostMem<'ctx>>]>,
    output_norm: DevMem<'ctx>,
    output: DevMem<'ctx>,
}

macro_rules! op {
    ($name:ident) => {
        operators::$name::nvidia_gpu::Operator
    };
}

impl<N, R> llama::Operators for Operators<N, R>
where
    N: TopoNode<Gpu>,
    R: AllReduce<Gpu, N>,
{
    type Hardware = Gpu;
    type TopoNode = N;
    type RmsNorm = op!(rms_norm);
    type MatMul = op!(mat_mul);
    type Rope = op!(rope);
    type AttnKVCached = op!(attention_kv_cached);
    type Mlp = op!(mlp);
    type Rearrange = op!(rearrange);
    type AllReduce = R;

    fn debug<T>(tensor: &Tensor<T>)
    where
        T: Deref<Target = [ByteOf<Self::Hardware>]>,
    {
        let tensor = tensor.as_ref().map(|s| {
            let mut host = vec![0u8; s.len()];
            memcpy_d2h(&mut host, s);
            host
        });
        println!("{tensor}");
    }
}

impl<'blk> Weights<'blk> {
    pub fn new(model: &LlamaStorage<&'_ [u8]>, pool_size: usize, stream: &Stream<'blk>) -> Self {
        assert!(pool_size > 0);
        if pool_size < model.meta.nblk {
            todo!()
        } else {
            let mut loader = model.blocks[0].map(|s| H2DLoader::new(s.len(), stream));

            Self {
                blks: model
                    .blocks
                    .iter()
                    .map(|blk| LlamaBlkStorage {
                        attn_norm: loader.attn_norm.load(blk.attn_norm, stream),
                        attn_qkv: loader.attn_qkv.load(blk.attn_qkv, stream),
                        attn_o: loader.attn_o.load(blk.attn_o, stream),
                        ffn_norm: loader.ffn_norm.load(blk.ffn_norm, stream),
                        ffn_gate_up: loader.ffn_gate_up.load(blk.ffn_gate_up, stream),
                        ffn_down: loader.ffn_down.load(blk.ffn_down, stream),
                    })
                    .collect(),
                blk_source: Box::new([]),
                output_norm: stream.from_host(model.output_norm),
                output: stream.from_host(model.output),
            }
        }
    }
}

struct H2DLoader<'ctx> {
    event: Event<'ctx>,
    host: HostMem<'ctx>,
    dev: DevMem<'ctx>,
}

impl<'ctx> H2DLoader<'ctx> {
    fn new(size: usize, stream: &Stream<'ctx>) -> Self {
        Self {
            event: stream.record(),
            host: stream.ctx().malloc_host::<u8>(size),
            dev: stream.malloc::<u8>(size),
        }
    }

    fn load(&mut self, host: &[u8], stream: &Stream<'ctx>) -> DevMem<'ctx> {
        self.event.synchronize();
        self.host.copy_from_slice(host);
        stream.memcpy_h2d(&mut self.dev, &self.host);
        replace(&mut self.dev, stream.malloc::<u8>(self.host.len()))
    }
}

impl WeightLoader for Weights<'_> {
    type Hardware = Gpu;
    type Memory<'s>
        = &'s [DevByte]
    where
        Self: 's;

    #[inline]
    fn load_blk(
        &self,
        which: BlkWeight,
        iblk: usize,
        _queue: &QueueOf<Self::Hardware>,
    ) -> Self::Memory<'_> {
        let blk = &self.blks[iblk];
        match which {
            BlkWeight::AttnNorm => &blk.attn_norm,
            BlkWeight::AttnQKV => &blk.attn_qkv,
            BlkWeight::AttnO => &blk.attn_o,
            BlkWeight::FfnNorm => &blk.ffn_norm,
            BlkWeight::FfnGateUp => &blk.ffn_gate_up,
            BlkWeight::FfnDown => &blk.ffn_down,
        }
    }

    #[inline]
    fn output_norm(&self, _queue: &QueueOf<Self::Hardware>) -> Self::Memory<'_> {
        &self.output_norm
    }

    #[inline]
    fn output(&self, _queue: &QueueOf<Self::Hardware>) -> Self::Memory<'_> {
        &self.output
    }
}

#[cfg(test)]
mod test;