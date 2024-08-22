use common::GGufModel;
use ggus::{GGmlTokenType, GGufMetaDataValueType};
use std::str::{from_utf8, from_utf8_unchecked};
use tokeneer::{utok, Bpe, Method, Tokeneer};

/// A trait for tokenization.
pub trait Tokenize {
    /// Encode a text into a sequence of tokens.
    fn encode(&self, text: &str) -> Vec<utok>;
    /// Decode a token into str.
    fn decode(&self, token: utok) -> &str;
}

impl<M: tokeneer::Method> Tokenize for Tokeneer<M> {
    #[inline]
    fn encode(&self, text: &str) -> Vec<utok> {
        self.encode(text)
    }
    #[inline]
    fn decode(&self, token: utok) -> &str {
        unsafe { from_utf8_unchecked(self.internal().decode(token)) }
    }
}

/// Build a polymorphic tokenize from the GGuf model.
pub fn build_tokenize(gguf: &GGufModel) -> Box<dyn Tokenize> {
    let model = gguf.meta_kvs["tokenizer.ggml.model"]
        .value_reader()
        .read_str()
        .unwrap();
    match model {
        "llama" => Box::new(build_bpe(gguf)),
        _ => panic!("Unsupported tokenizer model: {model}"),
    }
}

fn build_bpe(gguf: &GGufModel) -> Tokeneer<Bpe> {
    let _pre = gguf.meta_kvs["tokenizer.ggml.pre"]
        .value_reader()
        .read_str()
        .unwrap();
    let mut tokens = gguf.meta_kvs["tokenizer.ggml.tokens"].value_reader();
    let mut scores = gguf.meta_kvs["tokenizer.ggml.scores"].value_reader();
    let mut token_type = gguf.meta_kvs["tokenizer.ggml.token_type"].value_reader();

    let unk = gguf.meta_kvs["tokenizer.ggml.unknown_token_id"]
        .value_reader()
        .read::<utok>()
        .unwrap();
    let bos = gguf.meta_kvs["tokenizer.ggml.bos_token_id"]
        .value_reader()
        .read::<utok>()
        .unwrap();
    let eos = gguf.meta_kvs["tokenizer.ggml.eos_token_id"]
        .value_reader()
        .read::<utok>()
        .unwrap();

    let (ty, len) = tokens.read_arr_header().unwrap();
    assert_eq!(ty, GGufMetaDataValueType::String);

    let (ty, len_) = scores.read_arr_header().unwrap();
    assert_eq!(ty, GGufMetaDataValueType::F32);
    assert_eq!(len_, len);

    let (ty, len_) = token_type.read_arr_header().unwrap();
    assert_eq!(ty, GGufMetaDataValueType::I32);
    assert_eq!(len_, len);

    let vocabs = (0..len).map(|_| tokens.read_str().unwrap());
    let scores = (0..len).map(|_| scores.read::<f32>().unwrap());
    let is_byte =
        (0..len).map(|_| token_type.read::<GGmlTokenType>().unwrap() == GGmlTokenType::Byte);

    let bpe = Bpe::new(vocabs, scores, is_byte, unk);
    let bos_piece = from_utf8(bpe.decode(bos)).unwrap().to_string();
    let eos_piece = from_utf8(bpe.decode(eos)).unwrap().to_string();

    let mut tokeneer = Tokeneer::new(bpe);
    tokeneer.extend_special([(bos_piece, vec![bos]), (eos_piece, vec![eos])]);
    tokeneer
}

// pub trait Normalizer {
//     fn encode<'a>(&self, text: &'a str) -> Cow<'a, str>;
//     fn decode<'a>(&self, text: &'a str) -> Cow<'a, str>;
// }

// impl Normalizer for () {
//     #[inline]
//     fn encode<'a>(&self, text: &'a str) -> Cow<'a, str> {
//         Cow::Borrowed(text)
//     }

//     #[inline]
//     fn decode<'a>(&self, text: &'a str) -> Cow<'a, str> {
//         Cow::Borrowed(text)
//     }
// }

// #[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
// pub struct BPECommonNormalizer;

// impl Normalizer for BPECommonNormalizer {
//     fn encode<'a>(&self, text: &'a str) -> Cow<'a, str> {
//         let mut ans = String::new();
//         if text
//             .chars()
//             .next()
//             .filter(char::is_ascii_alphabetic)
//             .is_some()
//         {
//             ans.push('▁');
//         }
//         for c in text.chars() {
//             ans.push(match c {
//                 ' ' => '▁',
//                 c => c,
//             });
//         }
//         Cow::Owned(ans)
//     }

//     #[inline]
//     fn decode<'a>(&self, text: &'a str) -> Cow<'a, str> {
//         if text.contains('▁') {
//             Cow::Owned(text.replace('▁', " "))
//         } else {
//             Cow::Borrowed(text)
//         }
//     }
// }
