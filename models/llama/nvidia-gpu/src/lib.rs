#![cfg(hw_detected)]

use llama::{BlkWeight, LlamaStorage, Tensor, WeightLoader};
use operators::{
    all_reduce::{AllReduce, NonAllReduce},
    cuda::{memcpy_d2h, DevByte},
    nvidia_gpu::Gpu,
    random_sample::nvidia_gpu::Operator as RandomSampleGpu,
    ByteOf, QueueOf, TopoNode,
};
use std::{marker::PhantomData, ops::Deref};

pub struct Operators<N = Gpu, R = NonAllReduce<Gpu>>(PhantomData<(N, R)>);

pub type RandomSample = llama::RandomSample<Gpu, RandomSampleGpu>;

pub struct Weights<'w>(PhantomData<&'w ()>);

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

impl<'w> Weights<'w> {
    pub fn new(_model: &LlamaStorage<&'w [u8]>, _rank: usize, _distribute: usize) -> Self {
        todo!()
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
        _which: BlkWeight,
        _iblk: usize,
        _queue: &QueueOf<Self::Hardware>,
    ) -> Self::Memory<'_> {
        todo!()
    }

    #[inline]
    fn output_norm(&self, _queue: &QueueOf<Self::Hardware>) -> Self::Memory<'_> {
        todo!()
    }

    #[inline]
    fn output(&self, _queue: &QueueOf<Self::Hardware>) -> Self::Memory<'_> {
        todo!()
    }
}

#[cfg(test)]
mod test;
