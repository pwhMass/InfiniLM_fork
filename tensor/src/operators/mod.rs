mod random_sample;

pub use random_sample::RandomSample;

use crate::Tensor;
use ndarray_layout::{ArrayLayout, Endian::BigEndian};
use operators::{random_sample::KVPair, TensorLayout};
use std::alloc::Layout;

impl<T> Tensor<T> {
    pub fn kv_pair_vec(n: usize, f: impl FnOnce(usize) -> T) -> Self {
        Self {
            element: KVPair::<()>::LAYOUT,
            layout: ArrayLayout::new_contiguous(&[n], BigEndian, size_of::<KVPair>()),
            physical: f(Layout::array::<KVPair>(n).unwrap().size()),
        }
    }

    #[inline]
    pub fn layout(&self) -> TensorLayout {
        TensorLayout::new(self.element, self.layout.shape(), self.layout.strides())
    }
}
