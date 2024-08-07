use crate::{as_byte_token, utok, Tokenizer};
use std::{
    collections::{HashMap, HashSet},
    io,
    iter::zip,
    ops::Deref,
    path::Path,
    pin::Pin,
    ptr::NonNull,
};

pub struct BPE {
    /// 保存所有词的字符串内容，以 u8 为单位所以不需要对齐，占用空间少
    _vocab: Pin<Box<[u8]>>,
    /// 按 token 顺序保存元信息
    tokens: Box<[TokenMeta]>,
    /// 按字符串的字典序排序的 token 索引，用于从字符串二分查找 token。
    /// 建立索引时直接剔除了不可能从 piece 构造的所有单字节
    sorted_pieces: Box<[utok]>,
    /// 用于索引单字节 token，因此不需要其他元信息
    bytes: Box<[utok; 256]>,
    /// token: <unk>
    unk: utok,
}

struct TokenMeta {
    /// 指向字符串内容的指针
    ptr: NonNull<u8>,
    /// 字符串长度
    len: u32,
    /// 字符串的合并排名，从 0 开始
    rank: u32,
}

unsafe impl Send for TokenMeta {}
unsafe impl Sync for TokenMeta {}

impl Deref for TokenMeta {
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
        let tokens = zip(meta, rank)
            .map(|((off, len), rank)| TokenMeta {
                ptr: unsafe { NonNull::new_unchecked(_vocab[off..].as_ptr().cast_mut()) },
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

    /// BPE 词表中，并非所有词都是合词规则可达的。此算法可识别“内部不可达”的 token。
    pub fn inaccessible(&self) -> HashMap<&str, utok> {
        self.sorted_pieces
            .iter()
            .filter_map(|&t| {
                let s = unsafe { std::str::from_utf8_unchecked(self.token(t)) };
                if self.encode(s).len() > 1 {
                    Some((s, t))
                } else {
                    None
                }
            })
            .collect()
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

    /// token id -> token meta
    #[inline(always)]
    fn token(&self, token: utok) -> &TokenMeta {
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
        tokenizer.iter().collect()
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

    pub struct Iter<'a> {
        bpe: &'a BPE,
        slice: &'a [Mark],
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
                merge: merged,
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
        merge: utok,
        rank: u32,
    }
    impl Ord for Merge {
        fn cmp(&self, other: &Self) -> Ordering {
            // 比较顺序：rank -> merged -> pos -> pair
            match self.rank.cmp(&other.rank) {
                Equal => match self.merge.cmp(&other.merge) {
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
        /// 尝试执行一次合并，返回是否成功执行了一次合并。
        pub fn merge(&mut self) -> bool {
            // 一次合并将涉及至多 4 个 token：
            //
            // t0 t1 t2 t3
            // -- -- -- --
            //      ↓
            // t0 merge t3
            // -- ----- --
            //
            // 成功的合并将至少消费合并队列中的 1 个项，
            // 同时至多向合并队列添加 2 个项：
            //
            // t0 merge t3
            //    --------
            // --------

            // 从合并队列消费
            while let Some(Merge {
                pos: p1,
                pair: (t1, t2),
                merge,
                ..
            }) = self.merges.pop()
            {
                // 确认合并项有效性
                if self.marks[p1].token != t1 {
                    continue;
                }
                let l1 = self.bpe.token(t1).len();
                let p2 = p1 + l1;
                if self.marks[p2].token != t2 {
                    continue;
                }
                // 合并
                self.marks[p1].token = merge;
                self.marks[p2].token = self.bpe.unk;

                let l2 = self.bpe.token(t2).len();
                let p3 = p2 + l2;
                // 创建 merge + t3 合并项
                match self.marks.get_mut(p3) {
                    None => {}
                    Some(Mark {
                        token,
                        back_distance,
                    }) => {
                        *back_distance = (l1 + l2) as _;

                        let t3 = *token;
                        let l3 = self.bpe.token(t3).len();
                        let p4 = p3 + l3;
                        if let Some(merge) = self.bpe.build_merge(self.text, p1..p4, (merge, t3)) {
                            self.merges.push(merge);
                        }
                    }
                }
                // 创建 t0 + merge 合并项
                match self.marks[p1].back_distance as usize {
                    0 => {}
                    l0 => {
                        let p0 = p1 - l0;
                        let t0 = self.marks[p0].token;
                        if let Some(merge) = self.bpe.build_merge(self.text, p0..p3, (t0, merge)) {
                            self.merges.push(merge);
                        }
                    }
                }
                // 成功合并
                return true;
            }
            false
        }

        #[inline]
        pub fn iter(&self) -> Iter {
            Iter {
                bpe: self.bpe,
                slice: &self.marks,
            }
        }
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
                while let Some(Merge {
                    rank,
                    merge: merged,
                    ..
                }) = merges.pop()
                {
                    let text = unsafe { from_utf8_unchecked(self.bpe.token(merged)) };
                    writeln!(f, "  {rank:>6} | {text}")?;
                }
            }
            writeln!(f, "---------------------------")
        }
    }
}
