mod default;

use crate::SampleArgs;
use tokeneer::utok;

pub(crate) use default::DefaultStrategy;

pub trait BatchStrategy<T: 'static> {
    fn is_empty(&self) -> bool;
    fn insert(&mut self, stub: SessionStub<T>);
    fn remove(&mut self, id: &SessionId) -> Option<SessionStub<T>>;
    fn prepare(&mut self) -> Round<T>;
    fn take_stubs(&mut self) -> Vec<SessionStub<T>>;
}

// 目前在有 prompt 的情况下，state.seq 的长度代表 prompt 还有多少未 prefill，也就是 `prompt[prompt.len() - state.seq..]` 代表未 prefill 的 prompt
pub(super) struct SessionStub<T> {
    pub session: Session<T>,
    pub state: State,
    pub prompt: Option<Box<[utok]>>,
}

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
#[repr(transparent)]
pub struct SessionId(pub usize);

pub struct Round<T> {
    pub overflow: Vec<Session<T>>,
    pub tokens: Vec<utok>,
    pub reqs: Vec<Req<T>>,
    pub sample: Vec<(SessionId, SampleInfo)>,
    pub output: Vec<(SessionId, usize)>,
    pub fast_map: Vec<(utok, utok)>,
    pub finished: Vec<Session<T>>,
}

impl<T> Default for Round<T> {
    fn default() -> Self {
        Self {
            overflow: Default::default(),
            tokens: Default::default(),
            reqs: Default::default(),
            sample: Default::default(),
            output: Default::default(),
            fast_map: Default::default(),
            finished: Default::default(),
        }
    }
}

#[derive(Clone, Copy)]
pub struct SampleInfo {
    pub args: SampleArgs,
    pub input_idx: usize,
    pub decode_len: usize,
}

pub struct Session<T> {
    pub id: SessionId,
    pub sample_args: SampleArgs,
    pub cache: Cache<T>,
}

pub struct Cache<T> {
    pub cache: T,
    pub capacity: usize,
    pub len: usize,
}

#[derive(Clone, Copy)]
pub(super) struct State {
    pub seq: usize,
    pub out: usize,
    pub decode_len: usize,
    pub max_steps: usize,
}

#[derive(Clone)]
pub(crate) struct Req<Cache> {
    pub cache: Cache,
    pub pos: usize,
    pub seq: usize,
}
