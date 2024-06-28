#![cfg(detected_cuda)]

mod gather;
mod sample;

use common::utok;
use common_devices::{layout, SliceOn};
use cuda::{AsRaw, ContextSpore, Device};
use digit_layout::types::{F16, U32};
use operators::{
    cuda::CurrentCtx, dyn_, fuesd_softmax::nvidia_gpu as softmax, mat_mul::nvidia_gpu as mat_mul,
    reform::nvidia_gpu as reform, rms_norm::nvidia_gpu as rms_norm, rope::nvidia_gpu as rope,
    swiglu::nvidia_gpu as swiglu, Operator, QueueOf, TensorLayout,
};
use std::{
    collections::HashMap,
    ops::{Deref, DerefMut},
    ptr::{null, null_mut},
};

pub use common_devices::Kernels;
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

impl Kernels for NvidiaKernels {
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

    fn rms_norm<T, U, V>(
        &self,
        y: &mut Tensor<T>,
        x: &Tensor<U>,
        w: &Tensor<V>,
        epsilon: f32,
        queue: &QueueOf<Self::Handle>,
    ) where
        T: DerefMut<Target = SliceOn<Self::Handle>>,
        U: Deref<Target = SliceOn<Self::Handle>>,
        V: Deref<Target = SliceOn<Self::Handle>>,
    {
        self.get(queue)
            .rms_norm
            .launch(
                &operators::rms_norm::Args {
                    y_layout: layout(y),
                    y_base: y.base_mut(),
                    x_layout: layout(x),
                    x_base: x.base(),
                    w_layout: layout(w),
                    w_base: w.base(),
                    epsilon,
                },
                queue,
            )
            .unwrap();
    }

    fn rope<T, U>(
        &self,
        t: &mut Tensor<T>,
        pos: &Tensor<U>,
        theta: f32,
        queue: &QueueOf<Self::Handle>,
    ) where
        T: DerefMut<Target = SliceOn<Self::Handle>>,
        U: Deref<Target = SliceOn<Self::Handle>>,
    {
        self.get(queue)
            .rope
            .launch(
                &operators::rope::Args {
                    t_layout: layout(t),
                    t_base: t.base_mut(),
                    p_layout: layout(pos),
                    p_base: pos.base(),
                    theta,
                },
                queue,
            )
            .unwrap();
    }

    fn mat_mul<T, U, V>(
        &self,
        c: &mut Tensor<T>,
        beta: f32,
        a: &Tensor<U>,
        b: &Tensor<V>,
        alpha: f32,
        queue: &QueueOf<Self::Handle>,
    ) where
        T: DerefMut<Target = SliceOn<Self::Handle>>,
        U: Deref<Target = SliceOn<Self::Handle>>,
        V: Deref<Target = SliceOn<Self::Handle>>,
    {
        self.get(queue)
            .mat_mul
            .launch(
                &operators::mat_mul::Args {
                    c_layout: layout(c),
                    c_base: c.base_mut(),
                    beta,
                    a_layout: layout(a),
                    a_base: a.base(),
                    b_layout: layout(b),
                    b_base: b.base(),
                    alpha,
                },
                queue,
            )
            .unwrap();
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

    fn softmax<T>(&self, att: &mut Tensor<T>, queue: &QueueOf<Self::Handle>)
    where
        T: DerefMut<Target = SliceOn<Self::Handle>>,
    {
        self.get(queue)
            .softmax
            .launch(
                &operators::fuesd_softmax::Args {
                    att_layout: layout(att),
                    att_base: att.base_mut(),
                },
                queue,
            )
            .unwrap();
    }

    fn swiglu<T, U>(&self, gate: &mut Tensor<T>, up: &Tensor<U>, queue: &QueueOf<Self::Handle>)
    where
        T: DerefMut<Target = SliceOn<Self::Handle>>,
        U: Deref<Target = SliceOn<Self::Handle>>,
    {
        self.get(queue)
            .swiglu
            .launch(
                &operators::swiglu::Args {
                    gate_layout: layout(gate),
                    gate_base: gate.base_mut(),
                    up_layout: layout(up),
                    up_base: up.base(),
                },
                queue,
            )
            .unwrap();
    }
}

pub struct DropOption<T>(Option<T>);

impl<T> From<T> for DropOption<T> {
    #[inline]
    fn from(value: T) -> Self {
        Self(Some(value))
    }
}

impl<T> AsRef<T> for DropOption<T> {
    #[inline]
    fn as_ref(&self) -> &T {
        self.0.as_ref().unwrap()
    }
}

impl<T> AsMut<T> for DropOption<T> {
    #[inline]
    fn as_mut(&mut self) -> &mut T {
        self.0.as_mut().unwrap()
    }
}

impl<T: ContextSpore> DropOption<T> {
    #[inline]
    pub fn sprout<'ctx>(&mut self, ctx: &'ctx CurrentCtx) -> <T as ContextSpore>::Resource<'ctx> {
        self.0.take().unwrap().sprout(ctx)
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
