#![allow(unused)]

use crate::utok;
use std::{pin::Pin, ptr::NonNull};

pub struct NewBpe {
    vocabs: Pin<Box<[u8]>>,
    token_to_piece: Box<[Token]>,
    piece_to_token: Box<[utok]>,
}

struct Token {
    ptr: NonNull<u8>,
    len: u32,
    score: f32,
}

impl AsRef<str> for Token {
    #[inline]
    fn as_ref(&self) -> &str {
        use std::{slice::from_raw_parts, str::from_utf8_unchecked};
        unsafe { from_utf8_unchecked(from_raw_parts(self.ptr.as_ptr(), self.len as _)) }
    }
}

impl NewBpe {
    pub fn new<'a>(
        vocabs: impl IntoIterator<Item = &'a str>,
        scores: impl Iterator<Item = f32>,
        vocab_size_hint: usize,
    ) -> Self {
        let mut text_buf = Vec::with_capacity(vocab_size_hint * 4);
        let mut token_to_piece = Vec::<(usize, usize)>::with_capacity(vocab_size_hint);

        for vocab in vocabs.into_iter() {
            let vocab = vocab.as_bytes();
            let off = text_buf.len();
            let len = vocab.len();
            text_buf.extend_from_slice(vocab);
            token_to_piece.push((off, len));
        }
        let vocab_size = token_to_piece.len();

        let vocabs = unsafe { Pin::new_unchecked(text_buf.into_boxed_slice()) };
        let token_to_piece = token_to_piece
            .into_iter()
            .zip(scores)
            .map(|((off, len), score)| Token {
                ptr: unsafe { NonNull::new_unchecked(vocabs.as_ptr().add(off).cast_mut()) },
                len: len as _,
                score,
            })
            .collect::<Box<[_]>>();
        assert_eq!(token_to_piece.len(), vocab_size);

        let mut piece_to_token = (0..vocab_size as utok).collect::<Box<[_]>>();
        piece_to_token.sort_by_key(|&i| token_to_piece[i as usize].as_ref());

        Self {
            vocabs,
            token_to_piece,
            piece_to_token,
        }
    }
}
