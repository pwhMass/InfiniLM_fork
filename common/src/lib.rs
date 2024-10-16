use std::ops::Deref;

pub enum Contiguous<'a, T> {
    Borrowed(&'a [u8]),
    Owned(T),
}

impl<T: Deref<Target = [u8]>> Deref for Contiguous<'_, T> {
    type Target = [u8];
    #[inline]
    fn deref(&self) -> &Self::Target {
        match self {
            Self::Borrowed(b) => b,
            Self::Owned(o) => o,
        }
    }
}

#[inline(always)]
pub fn borrow<T>(t: &[u8]) -> Contiguous<'_, T> {
    Contiguous::Borrowed(t)
}

#[inline(always)]
pub fn own<'a, T>(t: T) -> Contiguous<'a, T> {
    Contiguous::Owned(t)
}
