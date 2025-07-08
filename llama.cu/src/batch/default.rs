use super::{BatchStrategy, Req, Round, SessionId, SessionStub};
use crate::batch::SampleInfo;
use log::warn;
use std::{collections::BTreeMap, mem::take};

pub(crate) struct DefaultStrategy<T> {
    sess: BTreeMap<SessionId, SessionStub<T>>,
    pre_output: BTreeMap<SessionId, usize>,
    // 每次 prefill 的最大长度
    chunked_prefill_max_len: usize,
    max_toks: usize,
}

impl<T> DefaultStrategy<T> {
    pub fn new(chunked_prefill_len: Option<usize>, max_toks: usize) -> Self {
        Self {
            sess: Default::default(),
            pre_output: Default::default(),
            chunked_prefill_max_len: chunked_prefill_len.unwrap_or(usize::MAX),
            max_toks,
        }
    }
}

impl<T: 'static + Clone> BatchStrategy<T> for DefaultStrategy<T> {
    fn is_empty(&self) -> bool {
        self.sess.is_empty()
    }

    fn insert(&mut self, stub: SessionStub<T>) {
        assert!(self.sess.insert(stub.session.id, stub).is_none())
    }

    fn remove(&mut self, id: &SessionId) -> Option<SessionStub<T>> {
        self.sess.remove(id)
    }

    fn prepare(&mut self) -> Round<T> {
        let mut ans = Round::default();
        let mut out_idx = 0;

        let pre_output = take(&mut self.pre_output);

        let mut write_back_sessions = BTreeMap::new();

        while let Some((id, mut stub)) = self.sess.pop_first() {
            let max = stub.session.cache.capacity;
            let pos = stub.session.cache.len;
            let mut seq = stub.state.seq;
            let mut out = stub.state.out;
            let mut end = pos + seq;
            assert_eq!(out, 1, "TODO: 投机采样");
            // 验证缓存是否溢出
            if end > max {
                warn!("cache overflow {end} > {max}");
                // 缓存溢出，不再推理
                ans.overflow.push(stub.session);
                continue;
            }

            // 用于限制每次 tokens 总数
            let remain_tok_num = self.max_toks - ans.tokens.len();
            assert!(remain_tok_num > 0);

            let input_idx = ans.tokens.len();
            if let Some(prompt) = &stub.prompt {
                seq = self.chunked_prefill_max_len.min(seq).min(remain_tok_num);
                let (prompt, tail) = prompt[prompt.len() - stub.state.seq..].split_at(seq);

                if tail.is_empty() {
                    // 正常 prefill
                    if seq != prompt.len() {
                        log::debug!("{id:?} chunked prefil finished")
                    }
                    ans.tokens.extend(prompt);
                    // 更新 stub 信息
                    stub.state.seq = 1;
                    stub.prompt = None
                } else {
                    // chunked prefill
                    out = 0;
                    end = pos + seq;
                    ans.tokens.extend(prompt);
                    // 更新 stub 信息
                    stub.state.seq = tail.len()
                }
            } else {
                // decode
                assert_eq!(seq, 1);
                // fast embd
                ans.fast_map
                    .push((pre_output[&id] as _, ans.tokens.len() as _));
                ans.tokens.push(0)
            }

            // 尝试填充缓存
            stub.session.cache.len = end;
            // 填充推理信息
            ans.sample
                .extend((input_idx..input_idx + out).map(|input_idx| {
                    (
                        id,
                        SampleInfo {
                            args: stub.session.sample_args,
                            input_idx,
                            decode_len: stub.state.decode_len,
                        },
                    )
                }));
            ans.output.push((id, out));
            ans.reqs.push(Req {
                cache: stub.session.cache.cache.clone(),
                pos,
                seq,
            });
            if out > 0 {
                stub.state.decode_len += 1
            }

            // 输出处理
            if stub.state.decode_len == stub.state.max_steps {
                // 生成结束
                ans.finished.push(stub.session)
            } else {
                // 回填
                assert!(write_back_sessions.insert(id, stub).is_none());
                if out != 0 {
                    assert!(self.pre_output.insert(id, out_idx).is_none())
                }
            }
            out_idx += out;

            // 如果剩余 tokens 总数等于 0，则退出循环
            if self.max_toks == ans.tokens.len() {
                break;
            }
        }
        self.sess.append(&mut write_back_sessions);
        ans
    }

    fn take_stubs(&mut self) -> Vec<SessionStub<T>> {
        take(&mut self.sess).into_values().collect()
    }
}
