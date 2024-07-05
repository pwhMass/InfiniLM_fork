use common::utok;
use operators::{
    fuesd_softmax, mat_mul, reform, rms_norm, rope, swiglu, Argument, Handle, Operator, QueueOf,
    TensorLayout,
};
use std::ops::{Deref, DerefMut};
use tensor::Tensor;

pub fn layout<T>(t: &Tensor<T>) -> TensorLayout {
    let dt = t.data_layout();
    let shape = t
        .shape()
        .iter()
        .map(|&x| Argument::new(x as usize))
        .collect::<Vec<_>>();
    let strides = t
        .strides()
        .iter()
        .map(|&x| Argument::new(x as isize * dt.nbytes() as isize))
        .collect::<Vec<_>>();
    TensorLayout::new(dt, shape, strides)
}

pub type SliceOn<H> = [<H as Handle>::Byte];

pub trait Operators {
    type Handle: Handle;

    fn reform_op(&self, queue: &QueueOf<Self::Handle>) -> &impl reform::Reform<Self::Handle>;
    fn rms_norm_op(&self, queue: &QueueOf<Self::Handle>) -> &impl rms_norm::RmsNorm<Self::Handle>;
    fn mat_mul_op(&self, queue: &QueueOf<Self::Handle>) -> &impl mat_mul::MatMul<Self::Handle>;
    fn rope_op(&self, queue: &QueueOf<Self::Handle>) -> &impl rope::Rope<Self::Handle>;
    fn softmax_op(
        &self,
        queue: &QueueOf<Self::Handle>,
    ) -> &impl fuesd_softmax::FusedSoftmax<Self::Handle>;
    fn swiglu_op(&self, queue: &QueueOf<Self::Handle>) -> &impl swiglu::Swiglu<Self::Handle>;
}

pub trait KernelsA {
    type Handle: Handle;

    fn reform<T, U>(&self, dst: &mut Tensor<T>, src: &Tensor<U>, queue: &QueueOf<Self::Handle>)
    where
        T: DerefMut<Target = SliceOn<Self::Handle>>,
        U: Deref<Target = SliceOn<Self::Handle>>;

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
        V: Deref<Target = SliceOn<Self::Handle>>;

    fn rope<T, U>(
        &self,
        t: &mut Tensor<T>,
        pos: &Tensor<U>,
        theta: f32,
        queue: &QueueOf<Self::Handle>,
    ) where
        T: DerefMut<Target = SliceOn<Self::Handle>>,
        U: Deref<Target = SliceOn<Self::Handle>>;

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
        V: Deref<Target = SliceOn<Self::Handle>>;

    fn softmax<T>(&self, att: &mut Tensor<T>, queue: &QueueOf<Self::Handle>)
    where
        T: DerefMut<Target = SliceOn<Self::Handle>>;

    fn swiglu<T, U>(&self, gate: &mut Tensor<T>, up: &Tensor<U>, queue: &QueueOf<Self::Handle>)
    where
        T: DerefMut<Target = SliceOn<Self::Handle>>,
        U: Deref<Target = SliceOn<Self::Handle>>;
}

pub trait KernelsB {
    type Handle: Handle;

    fn gather<T, U, I>(
        &self,
        x: &mut Tensor<T>,
        table: &Tensor<U>,
        tokens: I,
        queue: &QueueOf<Self::Handle>,
    ) where
        T: DerefMut<Target = SliceOn<Self::Handle>>,
        U: Deref<Target = [u8]>,
        I: IntoIterator<Item = utok>;
}

pub trait Kernels<H: Handle>: KernelsA<Handle = H> + KernelsB<Handle = H> {}

impl<Ops: Operators> KernelsA for Ops {
    type Handle = <Ops as Operators>::Handle;

    fn reform<T, U>(&self, dst: &mut Tensor<T>, src: &Tensor<U>, queue: &QueueOf<Self::Handle>)
    where
        T: DerefMut<Target = SliceOn<Self::Handle>>,
        U: Deref<Target = SliceOn<Self::Handle>>,
    {
        self.reform_op(queue)
            .launch(
                &reform::Args {
                    dst_layout: layout(dst),
                    dst_base: dst.base_mut(),
                    src_layout: layout(src),
                    src_base: src.base(),
                },
                queue,
            )
            .unwrap();
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
        self.rms_norm_op(queue)
            .launch(
                &rms_norm::Args {
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
        self.rope_op(queue)
            .launch(
                &rope::Args {
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
        self.mat_mul_op(queue)
            .launch(
                &mat_mul::Args {
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

    fn softmax<T>(&self, att: &mut Tensor<T>, queue: &QueueOf<Self::Handle>)
    where
        T: DerefMut<Target = SliceOn<Self::Handle>>,
    {
        self.softmax_op(queue)
            .launch(
                &fuesd_softmax::Args {
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
        self.swiglu_op(queue)
            .launch(
                &swiglu::Args {
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
