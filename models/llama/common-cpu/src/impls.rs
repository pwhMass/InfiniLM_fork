use llama::{BlkWeight, LlamaBlkStorage, LlamaStorage, Tensor, WeightLoader};
use operators::{
    all_reduce::{AllReduce, NonAllReduce},
    common_cpu::Cpu,
    random_sample::common_cpu::Operator as RandomSampleCpu,
    ByteOf, QueueOf, TopoNode,
};
use std::{marker::PhantomData, ops::Deref};

pub struct Operators<N = Cpu, R = NonAllReduce<Cpu>>(PhantomData<(N, R)>);

pub type RandomSample = llama::RandomSample<Cpu, RandomSampleCpu>;

pub struct Weights<'w> {
    blks: Box<[LlamaBlkStorage<&'w [u8]>]>,
    output_norm: &'w [u8],
    output: &'w [u8],
}

macro_rules! op {
    ($name:ident) => {
        operators::$name::common_cpu::Operator
    };
}

impl<N, R> llama::Operators for Operators<N, R>
where
    N: TopoNode<Cpu>,
    R: AllReduce<Cpu, N>,
{
    type Hardware = Cpu;
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
        println!("{tensor}");
    }
}

impl<'w> Weights<'w> {
    pub fn new(model: &LlamaStorage<&'w [u8]>, rank: usize, distribute: usize) -> Self {
        let LlamaStorage {
            output_norm,
            output,
            blocks,
            ..
        } = model;
        Self {
            blks: blocks
                .iter()
                .map(|blk| {
                    blk.clone().map(|s| {
                        let len = s.len() / distribute;
                        &s[rank * len..][..len]
                    })
                })
                .collect(),
            output_norm,
            output,
        }
    }
}

impl WeightLoader for Weights<'_> {
    type Hardware = Cpu;
    type Memory<'s>
        = &'s [u8]
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
            BlkWeight::AttnNorm => blk.attn_norm,
            BlkWeight::AttnQKV => blk.attn_qkv,
            BlkWeight::AttnO => blk.attn_o,
            BlkWeight::FfnNorm => blk.ffn_norm,
            BlkWeight::FfnGateUp => blk.ffn_gate_up,
            BlkWeight::FfnDown => blk.ffn_down,
        }
    }

    #[inline]
    fn output_norm(&self, _queue: &QueueOf<Self::Hardware>) -> Self::Memory<'_> {
        self.output_norm
    }

    #[inline]
    fn output(&self, _queue: &QueueOf<Self::Hardware>) -> Self::Memory<'_> {
        self.output
    }
}
