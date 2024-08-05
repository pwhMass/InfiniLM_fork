use crate::{as_byte_token, utok, Tokenizer};
use std::{collections::HashSet, io, iter::zip, ops::Deref, path::Path, pin::Pin, ptr::NonNull};

pub struct BPE {
    /// 保存所有词的字符串内容，以 u8 为单位所以不需要对齐，占用空间少
    _vocab: Pin<Box<[u8]>>,
    /// 按 token 顺序保存元信息
    tokens: Box<[Token]>,
    /// 按字符串的字典序排序的 token 索引，用于从字符串二分查找 token。
    /// 建立索引时直接剔除了不可能从 piece 构造的所有单字节
    sorted_pieces: Box<[utok]>,
    /// 用于索引单字节 token，因此不需要其他元信息
    bytes: Box<[utok; 256]>,
    /// token: <unk>
    unk: utok,
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
    type Target = [u8];
    #[inline]
    fn deref(&self) -> &Self::Target {
        unsafe { std::slice::from_raw_parts(self.ptr.as_ptr(), self.len as _) }
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
        // 产生词迭代器
        let vocabs = offsets.iter().map(|slice| {
            let len = slice[0] as usize;
            std::str::from_utf8(&slice[1..][..len]).unwrap()
        });
        // 产生评分迭代器
        let scores = offsets.iter().map(|slice| {
            let len = slice[0] as usize;
            let ptr = slice[len + 2..].as_ptr().cast::<f32>();
            unsafe { ptr.read_unaligned() }
        });
        // 产生字节标记迭代器
        let mut i = 0;
        let is_byte = std::iter::from_fn(|| {
            if i < 3 {
                i += 1;
                Some(false)
            } else if i < 3 + 256 {
                i += 1;
                Some(true)
            } else {
                Some(false)
            }
        });
        // 构造分词器
        Ok(Self::new(vocabs, scores, is_byte, 0, offsets.len()))
    }

    pub fn new<'a>(
        vocabs: impl IntoIterator<Item = &'a str>,
        scores: impl IntoIterator<Item = f32>,
        is_byte: impl IntoIterator<Item = bool>,
        unk: utok,
        vocab_size_hint: usize,
    ) -> Self {
        let mut text_buf = Vec::with_capacity(vocab_size_hint * 4);
        let mut bytes = Box::new([unk; 256]);
        // 重新编排词表
        // 将字符串的内容和元信息分离
        // 内容全部保存到 text_buf 以实现缓存友好性
        // 字符串起始位置在 text_buf 中的偏移量和字符串长度保存到 meta 中
        let meta = vocabs
            .into_iter()
            .map(str::as_bytes)
            .zip(is_byte)
            .enumerate()
            .map(|(t, (piece, is_byte))| {
                let off = text_buf.len();
                let len = if is_byte {
                    let b = as_byte_token(piece).unwrap();
                    text_buf.push(b);
                    bytes[b as usize] = t as utok;
                    1
                } else {
                    text_buf.extend_from_slice(piece);
                    piece.len()
                };
                (off, len)
            })
            .collect::<Vec<_>>();
        // 锁定字符串内容的位置，以实现安全的自引用
        let _vocab = unsafe { Pin::new_unchecked(text_buf.into_boxed_slice()) };
        // 对分词评分重新赋权，转换为整型
        let rank = rank(&scores.into_iter().collect::<Vec<_>>());
        assert_eq!(
            meta.len(),
            rank.len(),
            "scores size mismatch with vocab size"
        );
        // tokens 中直接引用字符串位置，绑定评分
        let ptr = NonNull::new(_vocab.as_ptr().cast_mut()).unwrap();
        let tokens = zip(meta, rank)
            .map(|((off, len), rank)| Token {
                ptr: unsafe { ptr.add(off) },
                len: len as _,
                rank,
            })
            .collect::<Box<[_]>>();
        // 对 token 按字符串的字典序排序，用于从字符串二分查找 token
        // <unk> 和 <0xyz> 不应该通过 piece 搜索到，使用 set 排除
        let bytes_set = bytes.iter().chain(&[unk]).cloned().collect::<HashSet<_>>();
        let mut sorted_pieces = (0..tokens.len() as utok)
            .filter(|i| !bytes_set.contains(i))
            .collect::<Box<[_]>>();
        sorted_pieces.sort_unstable_by_key(|&i| &*tokens[i as usize]);

        Self {
            _vocab,
            tokens,
            sorted_pieces,
            bytes,
            unk,
        }
    }

    /// piece -> token
    #[inline]
    fn find_piece(&self, piece: &[u8]) -> Option<utok> {
        match self
            .sorted_pieces
            .binary_search_by_key(&piece, |&i| self.token(i))
        {
            Ok(i) => Some(self.sorted_pieces[i]),
            Err(_) => match *piece {
                [b] => Some(self.bytes[b as usize]),
                [..] => None,
            },
        }
    }

    #[inline(always)]
    fn token(&self, token: utok) -> &Token {
        &self.tokens[token as usize]
    }
}

impl Tokenizer for BPE {
    #[inline]
    fn vocab_size(&self) -> usize {
        self.tokens.len()
    }

    #[inline]
    fn encode(&self, text: &str) -> Vec<utok> {
        let mut tokenizer = self.build_tokenizer(text);
        while tokenizer.merge() {}
        tokenizer.into_vec()
    }

    #[inline]
    fn decode(&self, token: utok) -> &str {
        unsafe { std::str::from_utf8_unchecked(self.token(token)) }
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

mod algorithm {
    use super::{utok, BPE};
    use std::{
        cmp::Ordering::{self, Equal},
        collections::BinaryHeap,
        fmt,
        iter::zip,
        ops::Range,
    };

    pub struct BpeTokenizer<'a> {
        text: &'a [u8],
        bpe: &'a BPE,
        marks: Vec<Mark>,
        merges: BinaryHeap<Merge>,
    }

    impl BPE {
        pub fn build_tokenizer<'a>(&'a self, text: &'a str) -> BpeTokenizer<'_> {
            let mut marks = vec![Mark::unk(self.unk); text.len()];
            let mut merges = BinaryHeap::new();

            let mut buf = [0u8; 4];
            let mut last = None;
            for (i, c) in text.char_indices() {
                let c = c.encode_utf8(&mut buf).as_bytes();
                last = if let Some(token) = self.find_piece(c) {
                    marks[i].token = token;
                    if let Some(pos) = last.take() {
                        marks[i].back_distance = (i - pos) as _;
                        if let Some(merge) = self.build_merge(
                            text.as_bytes(),
                            pos..i + c.len(),
                            (marks[pos].token, token),
                        ) {
                            merges.push(merge);
                        }
                    }
                    Some(i)
                } else {
                    for (&b, mark) in zip(c, &mut marks[i..]) {
                        mark.token = self.bytes[b as usize];
                    }
                    None
                };
            }

            BpeTokenizer {
                text: text.as_bytes(),
                bpe: self,
                marks,
                merges,
            }
        }

        fn build_merge(
            &self,
            text: &[u8],
            range: Range<usize>,
            pair: (utok, utok),
        ) -> Option<Merge> {
            self.find_piece(&text[range.clone()]).map(|merged| Merge {
                pos: range.start,
                pair,
                merged,
                rank: self.token(merged).rank,
            })
        }
    }

    #[derive(Clone, Copy, Debug)]
    struct Mark {
        token: utok,
        back_distance: u32,
    }

    impl Mark {
        #[inline(always)]
        const fn unk(unk: utok) -> Self {
            Self {
                token: unk,
                back_distance: 0,
            }
        }
    }

    #[derive(Clone, Copy, PartialEq, Eq, Debug)]
    struct Merge {
        pos: usize,
        pair: (utok, utok),
        merged: utok,
        rank: u32,
    }
    impl Ord for Merge {
        fn cmp(&self, other: &Self) -> Ordering {
            // 比较顺序：rank -> merged -> pos -> pair
            match self.rank.cmp(&other.rank) {
                Equal => match self.merged.cmp(&other.merged) {
                    Equal => match self.pos.cmp(&other.pos) {
                        Equal => self.pair.cmp(&other.pair),
                        other => other,
                    },
                    other => other,
                },
                other => other,
            }
            .reverse()
        }
    }
    impl PartialOrd for Merge {
        #[inline]
        fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
            Some(self.cmp(other))
        }
    }

    impl BpeTokenizer<'_> {
        pub fn merge(&mut self) -> bool {
            while let Some(Merge {
                pos, pair, merged, ..
            }) = self.merges.pop()
            {
                let mark0 = self.marks[pos];
                let len0 = self.bpe.token(mark0.token).len as usize;
                let Some(&mark1) = self.marks.get(pos + len0) else {
                    continue;
                };
                if pair != (mark0.token, mark1.token) {
                    continue;
                }

                self.marks[pos].token = merged;
                self.marks[pos + len0].token = self.bpe.unk;

                let len1 = self.bpe.token(mark1.token).len as usize;
                match self.marks.get_mut(pos + len0 + len1) {
                    None => {}
                    Some(next) => {
                        next.back_distance = (len0 + len1) as _;

                        let next = next.token;
                        let len2 = self.bpe.token(next).len as usize;
                        if let Some(merge) = self.bpe.build_merge(
                            self.text,
                            pos..pos + len0 + len1 + len2,
                            (merged, next),
                        ) {
                            self.merges.push(merge);
                        }
                    }
                }

                match self.marks[pos].back_distance as usize {
                    0 => {}
                    back => {
                        if let Some(merge) = self.bpe.build_merge(
                            self.text,
                            pos - back..pos + len0 + len1,
                            (self.marks[pos - back].token, merged),
                        ) {
                            self.merges.push(merge);
                        }
                    }
                }

                return true;
            }
            false
        }

        #[inline]
        pub fn into_vec(self) -> Vec<utok> {
            self.iter().collect()
        }

        #[inline]
        fn iter(&self) -> Iter {
            Iter {
                bpe: self.bpe,
                slice: &self.marks,
            }
        }
    }

    struct Iter<'a> {
        bpe: &'a BPE,
        slice: &'a [Mark],
    }

    impl Iterator for Iter<'_> {
        type Item = utok;

        fn next(&mut self) -> Option<Self::Item> {
            match self.slice {
                &[Mark { token, .. }, ref tail @ ..] => {
                    self.slice = &tail[self.bpe.token(token).len() - 1..];
                    Some(token)
                }
                [] => None,
            }
        }
    }

    impl fmt::Display for BpeTokenizer<'_> {
        fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
            use std::str::{from_utf8, from_utf8_unchecked};

            writeln!(f, "---------------------------")?;
            {
                writeln!(f, "text:")?;
                writeln!(f, "  {}", unsafe { from_utf8_unchecked(self.text) })?;
            }
            writeln!(f, "---------------------------")?;
            {
                writeln!(f, "tokens:")?;
                write!(f, "  ")?;
                for token in self.iter() {
                    let text = unsafe { from_utf8_unchecked(self.bpe.token(token)) };
                    write!(f, "{text}")?;
                }
                writeln!(f)?;
            }
            writeln!(f, "---------------------------")?;
            {
                writeln!(f, "tokens:")?;
                for token in self.iter() {
                    write!(f, "  {token:>6}: ")?;
                    match from_utf8(self.bpe.token(token)) {
                        Ok(s) => writeln!(f, "{s}")?,
                        Err(_) => writeln!(f, "{token:?}")?,
                    }
                }
            }
            writeln!(f, "---------------------------")?;
            {
                writeln!(f, "merges:")?;
                let mut merges = self.merges.clone();
                while let Some(Merge { rank, merged, .. }) = merges.pop() {
                    let text = unsafe { from_utf8_unchecked(self.bpe.token(merged)) };
                    writeln!(f, "  {rank:>6} | {text}")?;
                }
            }
            writeln!(f, "---------------------------")
        }
    }
}
