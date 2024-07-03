#![cfg(detected_cuda)]

mod gather;
mod sample;

use common::utok;
use common_devices::{layout, Operators, SliceOn};
use cuda::{AsRaw, Device};
use digit_layout::types::{F16, U32};
use operators::{
    dyn_, fuesd_softmax::nvidia_gpu as softmax, mat_mul::nvidia_gpu as mat_mul,
    reform::nvidia_gpu as reform, rms_norm::nvidia_gpu as rms_norm, rope::nvidia_gpu as rope,
    swiglu::nvidia_gpu as swiglu, Operator, QueueOf, TensorLayout,
};
use std::{
    collections::HashMap,
    ops::{Deref, DerefMut},
    ptr::{null, null_mut},
};

pub use common_devices::{Kernels, KernelsA, KernelsB};
pub use operators::{cuda, nvidia_gpu::Handle as Gpu};
pub use sample::{sample_cpu, sample_nv};
pub use tensor::{reslice, reslice_mut, slice, split, udim, LocalSplitable, Tensor};

pub struct NvidiaKernels(HashMap<i32, Internal>);

struct Internal {
    mat_mul: mat_mul::Operator,
    rms_norm: rms_norm::Operator,
    rope: rope::Operator,
    reform: reform::Operator,
    softmax: softmax::Operator,
    swiglu: swiglu::Operator,
}

impl Internal {
    pub fn new(handle: &Gpu, d: usize) -> Self {
        let mat_mul = mat_mul::Operator::new(handle);

        let mut rms_norm = rms_norm::Operator::new(handle);
        rms_norm
            .scheme(&operators::rms_norm::Args {
                y_layout: TensorLayout::new(F16, [dyn_(), d.into()], [dyn_(); 2]),
                y_base: null_mut(),
                x_layout: TensorLayout::new(F16, [dyn_(), d.into()], [dyn_(); 2]),
                x_base: null(),
                w_layout: TensorLayout::new(F16, [d.into()], [dyn_()]),
                w_base: null(),
                epsilon: 0.,
            })
            .unwrap();

        let mut rope = rope::Operator::new(handle);
        rope.scheme(&operators::rope::Args {
            t_layout: TensorLayout::new(F16, [dyn_(); 3], [dyn_(); 3]),
            t_base: null_mut(),
            p_layout: TensorLayout::new(U32, [dyn_()], [dyn_()]),
            p_base: null(),
            theta: 0.,
        })
        .unwrap();

        let mut reform = reform::Operator::new(handle);
        reform
            .scheme(&operators::reform::Args {
                dst_layout: TensorLayout::new(F16, [dyn_(); 2], [dyn_(); 2]),
                dst_base: null_mut(),
                src_layout: TensorLayout::new(F16, [dyn_(); 2], [dyn_(); 2]),
                src_base: null(),
            })
            .unwrap();

        let mut softmax = softmax::Operator::new(handle);
        softmax
            .scheme(&operators::fuesd_softmax::Args {
                att_layout: TensorLayout::new(F16, [dyn_(); 3], [dyn_(); 3]),
                att_base: null_mut(),
            })
            .unwrap();

        let mut swiglu = swiglu::Operator::new(handle);
        swiglu
            .scheme(&operators::swiglu::Args {
                gate_layout: TensorLayout::new(F16, [dyn_(); 2], [dyn_(); 2]),
                gate_base: null_mut(),
                up_layout: TensorLayout::new(F16, [dyn_(); 2], [dyn_(); 2]),
                up_base: null(),
            })
            .unwrap();

        Self {
            mat_mul,
            rms_norm,
            rope,
            reform,
            softmax,
            swiglu,
        }
    }
}

impl NvidiaKernels {
    pub fn new(devices: &[Device], rms_norm_size: usize) -> Self {
        Self(
            devices
                .iter()
                .map(|d| {
                    (
                        unsafe { d.as_raw() },
                        Internal::new(&Gpu::new(d.retain_primary()), rms_norm_size),
                    )
                })
                .collect(),
        )
    }
}

impl NvidiaKernels {
    fn get(&self, queue: &QueueOf<Gpu>) -> &Internal {
        self.0.get(&unsafe { queue.ctx().dev().as_raw() }).unwrap()
    }
}

impl Kernels<Gpu> for NvidiaKernels {}

impl Operators for NvidiaKernels {
    type Handle = Gpu;

    fn rms_norm_op(
        &self,
        queue: &QueueOf<Self::Handle>,
    ) -> &impl operators::rms_norm::RmsNorm<Self::Handle> {
        &self.get(queue).rms_norm
    }

    fn mat_mul_op(
        &self,
        queue: &QueueOf<Self::Handle>,
    ) -> &impl operators::mat_mul::MatMul<Self::Handle> {
        &self.get(queue).mat_mul
    }

    fn rope_op(&self, queue: &QueueOf<Self::Handle>) -> &impl operators::rope::Rope<Self::Handle> {
        &self.get(queue).rope
    }

    fn softmax_op(
        &self,
        queue: &QueueOf<Self::Handle>,
    ) -> &impl operators::fuesd_softmax::FusedSoftmax<Self::Handle> {
        &self.get(queue).softmax
    }

    fn swiglu_op(
        &self,
        queue: &QueueOf<Self::Handle>,
    ) -> &impl operators::swiglu::Swiglu<Self::Handle> {
        &self.get(queue).swiglu
    }
}

impl KernelsB for NvidiaKernels {
    type Handle = Gpu;

    fn gather<T, U, I>(
        &self,
        x: &mut Tensor<T>,
        table: &Tensor<U>,
        tokens: I,
        queue: &QueueOf<Self::Handle>,
    ) where
        T: DerefMut<Target = SliceOn<Self::Handle>>,
        U: Deref<Target = [u8]>,
        I: IntoIterator<Item = utok>,
    {
        gather::gather(x, table, tokens, queue);
    }

    fn reform<T, U>(&self, dst: &mut Tensor<T>, src: &Tensor<U>, queue: &QueueOf<Self::Handle>)
    where
        T: DerefMut<Target = SliceOn<Self::Handle>>,
        U: Deref<Target = SliceOn<Self::Handle>>,
    {
        self.get(queue)
            .reform
            .launch(
                &operators::reform::Args {
                    dst_layout: layout(dst),
                    dst_base: dst.base_mut(),
                    src_layout: layout(src),
                    src_base: src.base(),
                },
                queue,
            )
            .unwrap();
    }
}

pub fn synchronize() {
    cuda::init();
    for i in 0..cuda::Device::count() {
        cuda::Device::new(i as _)
            .retain_primary()
            .apply(|ctx| ctx.synchronize());
    }
}
