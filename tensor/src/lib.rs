mod fmt;
mod split;

use ggml_quants::{digit_layout::DigitLayout, DataBlock};
use operators::TensorLayout;
use std::{
    ops::{Deref, DerefMut, Range},
    slice::from_raw_parts,
};

pub use ndarray_layout::{ArrayLayout, Endian::BigEndian};
pub use split::{LocalSplitable, Splitable};

#[derive(Clone)]
pub struct Tensor<T> {
    element: DigitLayout,
    layout: ArrayLayout<5>,
    physical: T,
}

impl<T> Tensor<T> {
    #[inline]
    pub const unsafe fn from_raw_parts(
        element: DigitLayout,
        layout: ArrayLayout<5>,
        physical: T,
    ) -> Self {
        Self {
            element,
            layout,
            physical,
        }
    }

    pub fn new(element: DigitLayout, shape: &[usize], physical: T) -> Self {
        Self {
            element,
            layout: ArrayLayout::new_contiguous(shape, BigEndian, element.nbytes().unwrap()),
            physical,
        }
    }

    pub fn new_typed<D: DataBlock>(shape: &[usize], physical: T) -> Self {
        assert!(shape.len() <= 5);

        let mut buf = [0; 5];
        buf[..shape.len()].copy_from_slice(shape);
        buf[shape.len() - 1] /= D::COUNT;
        Self {
            element: D::ID,
            layout: ArrayLayout::new_contiguous(&buf[..shape.len()], BigEndian, size_of::<D>()),
            physical,
        }
    }
}

/// access
impl<T> Tensor<T> {
    #[inline]
    pub const fn dt(&self) -> DigitLayout {
        self.element
    }

    #[inline]
    pub fn shape(&self) -> &[usize] {
        self.layout.shape()
    }

    #[inline]
    pub fn strides(&self) -> &[isize] {
        self.layout.strides()
    }

    #[inline]
    pub fn offset(&self) -> usize {
        self.layout.offset()
    }

    #[inline]
    pub fn take(self) -> T {
        self.physical
    }

    #[inline]
    pub fn as_ref(&self) -> Tensor<&T> {
        Tensor {
            element: self.element,
            layout: self.layout.clone(),
            physical: &self.physical,
        }
    }

    #[inline]
    pub fn as_mut(&mut self) -> Tensor<&mut T> {
        Tensor {
            element: self.element,
            layout: self.layout.clone(),
            physical: &mut self.physical,
        }
    }

    #[inline]
    pub fn map<U>(self, f: impl FnOnce(T) -> U) -> Tensor<U> {
        Tensor {
            element: self.element,
            layout: self.layout,
            physical: f(self.physical),
        }
    }

    #[inline]
    pub fn layout(&self) -> TensorLayout {
        TensorLayout::new(self.element, self.layout.shape(), self.layout.strides())
    }
}

impl<T, B> Tensor<T>
where
    T: Deref<Target = [B]>,
{
    /// # Safety
    ///
    /// 这个函数将在移除生命周期约束的情况下引用原始数据，对这块存储空间进行读写的安全性由开发者保证。
    #[inline]
    pub unsafe fn map_slice_static(&self) -> Tensor<&'static [B]> {
        self.as_ref()
            .map(|x| unsafe { from_raw_parts(x.as_ptr(), x.len()) })
    }

    #[inline]
    pub fn map_slice(&self) -> Tensor<&[B]> {
        self.as_ref().map(|x| &x[..])
    }

    #[inline]
    pub fn base(&self) -> *const B {
        unsafe { self.physical.as_ptr().byte_add(self.layout.offset()) }
    }
}

impl<T, B> Tensor<T>
where
    T: DerefMut<Target = [B]>,
{
    #[inline]
    pub fn map_slice_mut(&mut self) -> Tensor<&mut [B]> {
        self.as_mut().map(|x| &mut x[..])
    }

    #[inline]
    pub fn base_mut(&mut self) -> *mut B {
        unsafe { self.physical.as_mut_ptr().byte_add(self.layout.offset()) }
    }
}

/// transform
impl<T> Tensor<T> {
    #[inline]
    pub fn transpose(self, perm: &[usize]) -> Self {
        Self {
            layout: self.layout.transpose(perm),
            ..self
        }
    }

    #[inline]
    pub fn index(self, axis: usize, index: usize) -> Self {
        Self {
            layout: self.layout.index(axis, index),
            ..self
        }
    }

    #[inline]
    pub fn slice(self, axis: usize, start: usize, step: isize, len: usize) -> Self {
        Self {
            layout: self.layout.slice(axis, start, step, len),
            ..self
        }
    }

    #[inline]
    pub fn tile(self, axis: usize, tiles: &[usize]) -> Self {
        Self {
            layout: self.layout.tile_be(axis, tiles),
            ..self
        }
    }

    #[inline]
    pub fn merge(self, range: Range<usize>) -> Option<Self> {
        self.layout
            .merge(range)
            .map(|layout| Self { layout, ..self })
    }
}

impl<T: Splitable> Tensor<T> {
    pub fn split<'a>(&'a self, axis: usize, parts: &'a [usize]) -> impl Iterator<Item = Self> + 'a {
        self.layout.split(axis, parts).map(|layout| Self {
            element: self.element,
            layout,
            physical: self.physical.split(),
        })
    }
}
