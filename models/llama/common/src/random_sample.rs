use operators::{
    random_sample::{self, RandomSample as Trait, SampleArgs},
    Hardware, LaunchError, QueueAlloc,
};
use std::{
    marker::PhantomData,
    ops::{Deref, DerefMut},
    ptr::{null, null_mut},
};
use tensor::Tensor;

pub struct RandomSample<H, Op> {
    op: Op,
    _hardware: PhantomData<H>,
}

impl<H, Op> RandomSample<H, Op>
where
    H: Hardware,
    Op: Trait<H>,
{
    pub fn new(processor: &H) -> Self {
        Self {
            op: Op::new(processor),
            _hardware: PhantomData,
        }
    }

    pub fn launch<Pair, L, I, QA>(
        &self,
        pairs: &mut Tensor<Pair>,
        logits: &Tensor<L>,
        indices: &Tensor<I>,
        config: SampleArgs,
        workspace: &mut [H::Byte],
        queue_alloc: &QA,
    ) -> Result<(), LaunchError>
    where
        Pair: DerefMut<Target = [H::Byte]>,
        L: Deref<Target = [H::Byte]>,
        I: Deref<Target = [H::Byte]>,
        QA: QueueAlloc<Hardware = H>,
    {
        let layout = indices.layout();
        let mut args = random_sample::Args {
            kv_pair: layout.clone(),
            kv_pair_base: null_mut(),
            logits: layout.clone(),
            logits_base: null(),
            indices: layout.clone(),
            indices_base: indices.base(),
            config,
            seed: 0.,
        };

        for i in 0..logits.shape()[0] {
            let mut pair = pairs.map_slice_mut().index(0, i);
            let logits = logits.map_slice().index(0, i);

            args.kv_pair = pair.layout();
            args.kv_pair_base = pair.base_mut();
            args.logits = logits.layout();
            args.logits_base = logits.base();
            args.seed = rand::random();

            self.op.launch(&args, workspace, queue_alloc)?;
        }

        Ok(())
    }
}
