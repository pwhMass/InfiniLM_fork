#[macro_export]
macro_rules! slice {
    ($blob:expr; $width:expr; [$line:expr]) => {
        $blob[$line as usize * $width as usize..][..$width as usize]
    };
}

mod gather;

use common::utok;
use common_devices::{layout, SliceOn};
use operators::{
    fuesd_softmax::common_cpu as softmax, mat_mul::common_cpu as mat_mul,
    rms_norm::common_cpu as rms_norm, rope::common_cpu as rope, swiglu::common_cpu as swiglu,
    Operator, QueueOf,
};
use std::ops::{Deref, DerefMut};
use tensor::Tensor;

pub extern crate tensor;

pub use common_devices::Kernels;
pub use operators::common_cpu::{Handle as Cpu, ThisThread};

pub struct CpuKernels {
    mat_mul: mat_mul::Operator,
    rms_norm: rms_norm::Operator,
    rope: rope::Operator,
    softmax: softmax::Operator,
    swiglu: swiglu::Operator,
}

impl Default for CpuKernels {
    fn default() -> Self {
        Self {
            mat_mul: mat_mul::Operator::new(&Cpu),
            rms_norm: rms_norm::Operator::new(&Cpu),
            rope: rope::Operator::new(&Cpu),
            softmax: softmax::Operator::new(&Cpu),
            swiglu: swiglu::Operator::new(&Cpu),
        }
    }
}

impl Kernels for CpuKernels {
    type Handle = Cpu;

    fn gather<T, U, I>(
        &self,
        x: &mut Tensor<T>,
        table: &Tensor<U>,
        tokens: I,
        _queue: &QueueOf<Self::Handle>,
    ) where
        T: DerefMut<Target = SliceOn<Self::Handle>>,
        U: Deref<Target = SliceOn<Self::Handle>>,
        I: IntoIterator<Item = utok>,
    {
        gather::gather(x, table, tokens);
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
        self.rms_norm
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
        self.rope
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
        self.mat_mul
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

    fn reform<T, U>(&self, dst: &mut Tensor<T>, src: &Tensor<U>, _queue: &QueueOf<Self::Handle>)
    where
        T: DerefMut<Target = SliceOn<Self::Handle>>,
        U: Deref<Target = SliceOn<Self::Handle>>,
    {
        src.reform_to(dst);
    }

    fn softmax<T>(&self, att: &mut Tensor<T>, queue: &QueueOf<Self::Handle>)
    where
        T: DerefMut<Target = SliceOn<Self::Handle>>,
    {
        self.softmax
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
        self.swiglu
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
