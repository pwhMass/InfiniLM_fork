#![cfg(hw_detected)]

use llama::{BlkWeight, LlamaBlkStorage, LlamaStorage, Tensor, WeightLoader};
use operators::{
    all_reduce::{AllReduce, NonAllReduce},
    cuda::{memcpy_d2h, DevByte, DevMem, HostMem, Stream},
    nvidia_gpu::Gpu,
    random_sample::nvidia_gpu::Operator as RandomSampleGpu,
    ByteOf, QueueOf, TopoNode,
};
use std::{marker::PhantomData, ops::Deref};

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
    pub fn new(
        model: &LlamaStorage<&'_ [u8]>,
        rank: usize,
        distribute: usize,
        pool_size: usize,
        stream: &Stream<'blk>,
    ) -> Self {
        assert!(pool_size > 0);
        if pool_size < model.meta.nblk {
            todo!()
        } else {
            assert_eq!(rank, 0);
            assert_eq!(distribute, 1);
            Self {
                blks: model
                    .blocks
                    .iter()
                    .map(|blk| blk.clone().map(|s| stream.from_host(s)))
                    .collect(),
                blk_source: Box::new([]),
                output_norm: stream.from_host(model.output_norm),
                output: stream.from_host(model.output),
            }
        }
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
