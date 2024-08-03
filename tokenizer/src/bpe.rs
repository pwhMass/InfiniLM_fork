use crate::{decode_with_ascii, utok, Tokenizer};
use std::{io, iter::zip, ops::Deref, path::Path, pin::Pin, ptr::NonNull};

pub struct BPE {
    _vocab: Pin<Box<[u8]>>,
    tokens: Box<[Token]>,
    sorted_pieces: Box<[utok]>,
}

struct Token {
    /// 指向字符串内容的指针
    ptr: NonNull<u8>,
    /// 字符串长度
    len: u32,
    /// 字符串的合并排名，从 0 开始
    rank: u32,
}

unsafe impl Send for Token {}
unsafe impl Sync for Token {}

impl Deref for Token {
    type Target = str;
    #[inline]
    fn deref(&self) -> &Self::Target {
        use std::{slice::from_raw_parts, str::from_utf8_unchecked};
        unsafe { from_utf8_unchecked(from_raw_parts(self.ptr.as_ptr(), self.len as _)) }
    }
}

impl BPE {
    /// 打开 tokenizer.model 文件并构造一个 bpe 分词器。
    pub fn from_tokenizer_model(model_file: impl AsRef<Path>) -> io::Result<Self> {
        // 打开文件
        let file = std::fs::File::open(model_file)?;
        let mmap = unsafe { memmap2::Mmap::map(&file) }?;
        // 遍历文件，标记所有词汇的位置并记录最大长度
        let offsets = (0..)
            .scan(0usize, |offset, _| match &mmap[*offset..] {
                [10, total_len, 10, content @ ..] => {
                    let total_len = *total_len as usize;
                    *offset += total_len + 2;
                    Some(&content[..total_len - 2])
                }
                [..] => None,
            })
            .collect::<Vec<_>>();
        let vocabs = offsets.iter().map(|slice| {
            let len = slice[0] as usize;
            std::str::from_utf8(&slice[1..][..len]).unwrap()
        });
        let scores = offsets.iter().map(|slice| {
            let len = slice[0] as usize;
            let ptr = slice[len + 2..].as_ptr().cast::<f32>();
            unsafe { ptr.read_unaligned() }
        });

        Ok(Self::new(vocabs, scores, offsets.len()))
    }

    pub fn new<'a>(
        vocabs: impl IntoIterator<Item = &'a str>,
        scores: impl Iterator<Item = f32>,
        vocab_size_hint: usize,
    ) -> Self {
        let mut text_buf = Vec::with_capacity(vocab_size_hint * 4);
        // 重新编排词表
        // 将字符串的内容和元信息分离
        // 内容全部保存到 text_buf 以实现缓存友好性
        // 字符串起始位置在 text_buf 中的偏移量和字符串长度保存到 meta 中
        let meta = vocabs
            .into_iter()
            .map(str::as_bytes)
            .map(|vocab| {
                let off = text_buf.len();
                text_buf.extend_from_slice(vocab);
                (off, vocab.len())
            })
            .collect::<Vec<_>>();
        // 锁定字符串内容的位置，以实现安全的自引用
        let _vocab = unsafe { Pin::new_unchecked(text_buf.into_boxed_slice()) };
        // 对分词评分重新赋权，转换为整型
        let rank = rank(&scores.collect::<Vec<_>>());
        assert_eq!(
            meta.len(),
            rank.len(),
            "scores size mismatch with vocab size"
        );
        // tokens 中直接引用字符串位置，绑定评分
        let tokens = zip(meta, rank)
            .map(|((off, len), rank)| Token {
                ptr: unsafe { NonNull::new_unchecked(_vocab.as_ptr().add(off).cast_mut()) },
                len: len as _,
                rank,
            })
            .collect::<Box<[_]>>();
        // 对 token 按字符串的字典序排序，用于从字符串二分查找 token
        let mut sorted_pieces = (0..tokens.len() as utok).collect::<Box<[_]>>();
        sorted_pieces.sort_by_key(|&i| &*tokens[i as usize]);

        Self {
            _vocab,
            tokens,
            sorted_pieces,
        }
    }

    /// piece -> token
    #[inline]
    fn find_piece(&self, piece: &str) -> Option<utok> {
        self.sorted_pieces
            .binary_search_by_key(&piece, |&i| &*self.tokens[i as usize])
            .ok()
            .map(|i| self.sorted_pieces[i])
    }
}

impl Tokenizer for BPE {
    #[inline]
    fn vocab_size(&self) -> usize {
        self.tokens.len()
    }

    fn encode(&self, text: &str) -> Vec<utok> {
        let mut tokens = Vec::new();

        text.chars().map(|c| c.to_string()).for_each(|c| {
            if let Some(index) = self.find_piece(&c) {
                tokens.extend([index]);
            } else {
                tokens.extend(c.bytes().map(|c| c as utok + 3));
            }
        });

        fn map_pair(bpe: &BPE, tokens: &[utok], i: usize) -> Option<(utok, u32)> {
            bpe.find_piece(&format!(
                "{}{}",
                &*bpe.tokens[tokens[i] as usize],
                &*bpe.tokens[tokens[i + 1] as usize],
            ))
            .map(|tok| (tok, bpe.tokens[tok as usize].rank))
        }

        let mut merges = (0..tokens.len() - 1)
            .map(|i| map_pair(self, &tokens, i))
            .collect::<Vec<_>>();
        while let Some((i, (tok, _))) = merges
            .iter()
            .enumerate()
            .filter_map(|(i, tok)| tok.map(|tok| (i, tok)))
            .min()
        {
            tokens[i] = tok;
            tokens.remove(i + 1);
            merges.remove(i);
            if let Some(i) = i.checked_sub(1) {
                merges[i] = map_pair(self, &tokens, i);
            }
            if i + 1 < merges.len() {
                merges[i] = map_pair(self, &tokens, i);
            }
        }

        tokens
    }

    #[inline]
    fn decode(&self, token: utok) -> &str {
        decode_with_ascii(&*self.tokens[token as usize])
    }
}

/// 对一组评分排序、去重并重新赋权，转换为保持相同顺序的整型序列
fn rank(scores: &[f32]) -> Vec<u32> {
    use std::{
        cmp::Ordering,
        collections::{BTreeMap, BTreeSet},
    };

    #[derive(PartialEq, Debug)]
    struct FloatOrd(f32);
    impl Eq for FloatOrd {}
    impl PartialOrd for FloatOrd {
        #[inline]
        fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
            Some(self.cmp(other))
        }
    }
    impl Ord for FloatOrd {
        #[inline]
        fn cmp(&self, other: &Self) -> Ordering {
            self.0.total_cmp(&other.0)
        }
    }

    let map = scores
        // 排序 + 去重
        .iter()
        .copied()
        .map(FloatOrd)
        .collect::<BTreeSet<_>>()
        // 重新赋权
        .into_iter()
        .rev()
        .enumerate()
        .map(|(i, f)| (f, i as u32))
        .collect::<BTreeMap<_, _>>();

    scores.iter().map(|f| map[&FloatOrd(*f)]).collect()
}
