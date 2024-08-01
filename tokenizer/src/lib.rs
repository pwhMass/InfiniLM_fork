mod bpe;
mod normalizer;
mod vocab_txt;

use common::utok;

pub trait Tokenizer {
    fn vocab_size(&self) -> usize;
    fn encode(&self, text: &str) -> Vec<utok>;
    fn decode(&self, token: utok) -> &str;
}

pub use bpe::BPE;
pub use normalizer::{BPECommonNormalizer, Normalizer};
pub use vocab_txt::VocabTxt;

const fn decode_with_ascii<'a>(piece: &'a str) -> &'a str {
    // 预填充 ASCII 码表的所有字符
    const BYTES: [u8; 256] = {
        let mut ans = [0; 256];
        let mut i = 0;
        while i < 256 {
            ans[i] = i as _;
            i += 1;
        }
        ans
    };
    // ascii 转数字
    const fn to_num(c: u8) -> u8 {
        match c {
            b'0'..=b'9' => c - b'0',
            b'a'..=b'f' => c - b'a' + 10,
            b'A'..=b'F' => c - b'A' + 10,
            _ => unreachable!(),
        }
    }
    // 按结构分解并转换
    match piece.as_bytes() {
        &[b'<', b'0', b'x', a, b, b'>'] if a.is_ascii_hexdigit() && b.is_ascii_hexdigit() => {
            use std::{slice::from_ref, str::from_utf8_unchecked};
            let byte = to_num(a) * 16 + to_num(b);
            let byte = from_ref(&BYTES[byte as usize]);
            unsafe { from_utf8_unchecked(byte) }
        }
        _ => piece,
    }
}

#[test]
fn test_decode_with_byte() {
    assert_eq!(decode_with_ascii("<0x0A>"), "\n");
    assert_eq!(decode_with_ascii("<0x20>"), " ");
    assert_eq!(decode_with_ascii("<0x2E>"), ".");
    assert_eq!(decode_with_ascii("<0x30>"), "0");
    assert_eq!(decode_with_ascii("<0x39>"), "9");
    assert_eq!(decode_with_ascii("<0x41>"), "A");
    assert_eq!(decode_with_ascii("<0x5A>"), "Z");
    assert_eq!(decode_with_ascii("<0x61>"), "a");
    assert_eq!(decode_with_ascii("<0x7A>"), "z");
}
