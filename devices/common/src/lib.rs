use common::utok;
use operators::{Argument, Handle, QueueOf, TensorLayout};
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

pub trait Kernels {
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

    fn reform<T, U>(&self, dst: &mut Tensor<T>, src: &Tensor<U>, queue: &QueueOf<Self::Handle>)
    where
        T: DerefMut<Target = SliceOn<Self::Handle>>,
        U: Deref<Target = SliceOn<Self::Handle>>;

    fn softmax<T>(&self, att: &mut Tensor<T>, queue: &QueueOf<Self::Handle>)
    where
        T: DerefMut<Target = SliceOn<Self::Handle>>;

    fn swiglu<T, U>(&self, gate: &mut Tensor<T>, up: &Tensor<U>, queue: &QueueOf<Self::Handle>)
    where
        T: DerefMut<Target = SliceOn<Self::Handle>>,
        U: Deref<Target = SliceOn<Self::Handle>>;
}

// pub fn rms_norm<S, D, Y, X, W>(
//     _: PhantomData<S>,
//     op: &S::Operator,
//     y: &mut Tensor<Y>,
//     x: &Tensor<X>,
//     w: &Tensor<W>,
//     epsilon: f32,
//     queue: &QueueOf<D>,
// ) where
//     D: Device,
//     S: RmsNorm<D>,
//     S::Error: fmt::Debug,
//     Y: DerefMut<Target = [D::Byte]>,
//     X: Deref<Target = [D::Byte]>,
//     W: Deref<Target = [D::Byte]>,
// {
//     S::new(
//         op,
//         rms_norm::LayoutAttrs {
//             y: layout(y),
//             x: layout(x),
//             w: layout(w),
//         },
//     )
//     .unwrap()
//     .launch(
//         &(
//             y.physical_mut().as_mut_ptr(),
//             x.physical().as_ptr(),
//             w.physical().as_ptr(),
//             epsilon,
//         ),
//         queue,
//     );
// }

// pub fn rope<S, D, T, Pos>(
//     _: PhantomData<S>,
//     op: &S::Operator,
//     t: &mut Tensor<T>,
//     pos: &Tensor<Pos>,
//     theta: f32,
//     queue: &QueueOf<D>,
// ) where
//     D: Device,
//     S: Rope<D>,
//     S::Error: fmt::Debug,
//     T: DerefMut<Target = [D::Byte]>,
//     Pos: Deref<Target = [D::Byte]>,
// {
//     S::new(
//         op,
//         rope::LayoutAttrs {
//             t: layout(t),
//             pos: layout(pos),
//         },
//     )
//     .unwrap()
//     .launch(
//         &(
//             t.physical_mut().as_mut_ptr(),
//             pos.physical().as_ptr(),
//             theta,
//         ),
//         queue,
//     );
// }

// pub fn mat_mul<S, D, C, A, B>(
//     _: PhantomData<S>,
//     op: &S::Operator,
//     c: &mut Tensor<C>,
//     beta: f32,
//     a: &Tensor<A>,
//     b: &Tensor<B>,
//     alpha: f32,
//     queue: &QueueOf<D>,
// ) where
//     D: Device,
//     S: MatMul<D>,
//     S::Error: fmt::Debug,
//     C: DerefMut<Target = [D::Byte]>,
//     A: Deref<Target = [D::Byte]>,
//     B: Deref<Target = [D::Byte]>,
// {
//     S::new(
//         op,
//         mat_mul::LayoutAttrs {
//             c: layout(c),
//             a: layout(a),
//             b: layout(b),
//         },
//     )
//     .unwrap()
//     .launch(
//         &(
//             c.physical_mut().as_mut_ptr(),
//             beta,
//             a.physical().as_ptr(),
//             b.physical().as_ptr(),
//             alpha,
//         ),
//         queue,
//     );
// }

// pub fn reform<S, D, Dst, Src>(
//     _: PhantomData<S>,
//     op: &S::Operator,
//     dst: &mut Tensor<Dst>,
//     src: &Tensor<Src>,
//     queue: &QueueOf<D>,
// ) where
//     D: Device,
//     S: Reform<D>,
//     S::Error: fmt::Debug,
//     Dst: DerefMut<Target = [D::Byte]>,
//     Src: Deref<Target = [D::Byte]>,
// {
//     S::new(
//         op,
//         reform::LayoutAttrs {
//             dst: layout(dst),
//             src: layout(src),
//         },
//     )
//     .unwrap()
//     .launch(
//         &(dst.physical_mut().as_mut_ptr(), src.physical().as_ptr()),
//         queue,
//     );
// }

// pub fn softmax<S, D, Att>(
//     _: PhantomData<S>,
//     op: &S::Operator,
//     att: &mut Tensor<Att>,
//     queue: &QueueOf<D>,
// ) where
//     D: Device,
//     S: FuesdSoftmax<D>,
//     S::Error: fmt::Debug,
//     Att: DerefMut<Target = [D::Byte]>,
// {
//     S::new(op, fuesd_softmax::LayoutAttrs { att: layout(att) })
//         .unwrap()
//         .launch(&att.physical_mut().as_mut_ptr(), queue);
// }

// pub fn swiglu<S, D, Gate, Up>(
//     _: PhantomData<S>,
//     op: &S::Operator,
//     gate: &mut Tensor<Gate>,
//     up: &Tensor<Up>,
//     queue: &QueueOf<D>,
// ) where
//     D: Device,
//     S: Swiglu<D>,
//     S::Error: fmt::Debug,
//     Gate: DerefMut<Target = [D::Byte]>,
//     Up: Deref<Target = [D::Byte]>,
// {
//     S::new(
//         op,
//         swiglu::LayoutAttrs {
//             gate: layout(gate),
//             up: layout(up),
//         },
//     )
//     .unwrap()
//     .launch(
//         &(gate.physical_mut().as_mut_ptr(), up.physical().as_ptr()),
//         queue,
//     );
// }
