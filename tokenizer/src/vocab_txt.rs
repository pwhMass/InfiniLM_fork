use crate::{decode_with_ascii, Tokenizer};
use common::utok;
use memmap2::Mmap;
use patricia_tree::PatriciaMap;
use std::{fs::File, io::Result, path::Path};

/// 一个基于朴素词表的分词器。
pub struct VocabTxt {
    /// 词表。
    words: Vec<String>,
    /// 词汇的前缀树。
    trie: PatriciaMap<utok>,
}

impl VocabTxt {
    pub fn from_txt_file(tokenizer: impl AsRef<Path>) -> Result<Self> {
        let file = File::open(tokenizer)?;
        let mmap = unsafe { Mmap::map(&file) }?;
        let text = unsafe { std::str::from_utf8_unchecked(&mmap) };

        let mut words = Vec::new();
        let mut trie = PatriciaMap::new();
        for (i, line) in text.lines().enumerate() {
            let piece = line.strip_prefix('"').unwrap().strip_suffix('"').unwrap();
            words.push(piece.to_string());
            trie.insert(piece, i as _);
        }
        Ok(Self { words, trie })
    }
}

impl Tokenizer for VocabTxt {
    fn vocab_size(&self) -> usize {
        self.words.len()
    }

    fn encode(&self, mut text: &str) -> Vec<utok> {
        let mut tokens = Vec::<utok>::new();

        while !text.is_empty() {
            if let Some((pre, tok)) = self.trie.get_longest_common_prefix(text) {
                tokens.push(*tok);
                text = &text[pre.len()..];
            } else {
                let mut chars = text.chars();
                let char = chars.next().unwrap();
                tokens.extend(char.to_string().bytes().map(|b| (b + 3) as utok));
                text = chars.as_str();
            }
        }

        tokens
    }

    #[inline]
    fn decode(&self, token: utok) -> &str {
        decode_with_ascii(self.words[token as usize].as_str())
    }
}
