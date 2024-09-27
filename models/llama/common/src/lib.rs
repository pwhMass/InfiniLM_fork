mod args;
mod compute;
mod random_sample;
mod storage;

use ggus::ggml_quants::digit_layout::DigitLayout;
use tensor::Tensor;

pub use args::{Args as LlamaArgs, Request as LlamaRequest};
pub use compute::{BlkWeight, LlamaBlks, Operators, WeightLoader};
pub use ggus::ggml_quants::digit_layout::types as primitive;
pub use random_sample::RandomSample;
pub use storage::{BlkStorage as LlamaBlkStorage, Storage as LlamaStorage};

#[derive(Clone, Debug)]
pub struct LlamaMeta {
    pub dt_norm: DigitLayout,
    pub dt_mat: DigitLayout,
    pub nblk: usize,
    pub nh: usize,
    pub nkvh: usize,
    pub dh: usize,
    pub di: usize,
    pub dctx: usize,
    pub dvoc: usize,
    pub epsilon: f32,
    pub theta: f32,
    pub distribute: usize,
}

impl LlamaMeta {
    pub fn kv_cache<T>(&self, buf: usize, p: T) -> Tensor<T> {
        let &Self {
            dt_mat,
            nblk,
            nkvh,
            dh,
            distribute,
            ..
        } = self;
        Tensor::new(dt_mat, &[buf, nblk, 2, nkvh / distribute, dh], p)
    }

    pub fn embd<T>(&self, nt: usize, p: T) -> Tensor<T> {
        let &Self { dt_mat, nh, dh, .. } = self;
        Tensor::new(dt_mat, &[nt, nh * dh], p)
    }

    pub fn logits<T>(&self, nt: usize, p: T) -> Tensor<T> {
        let &Self { dt_mat, dvoc, .. } = self;
        Tensor::new(dt_mat, &[nt, dvoc], p)
    }

    pub fn token_embd<T>(&self, p: T) -> Tensor<T> {
        let &Self {
            dt_mat,
            nh,
            dh,
            dvoc,
            ..
        } = self;
        Tensor::new(dt_mat, &[dvoc, nh * dh], p)
    }

    pub fn attn_norm<T>(&self, p: T) -> Tensor<T> {
        self.norm(p)
    }

    pub fn attn_qkv<T>(&self, p: T, distributed: bool) -> Tensor<T> {
        let &Self {
            nh,
            nkvh,
            dh,
            distribute,
            ..
        } = self;
        let row = (nh + nkvh + nkvh) / distribute * dh;
        let col = nh * dh;
        self.mat(p, row, col, distributed)
    }

    pub fn attn_o<T>(&self, p: T, distributed: bool) -> Tensor<T> {
        let &Self {
            nh, dh, distribute, ..
        } = self;
        let row = nh * dh;
        let col = nh / distribute * dh;
        self.mat(p, row, col, distributed)
    }

    pub fn ffn_norm<T>(&self, p: T) -> Tensor<T> {
        self.norm(p)
    }

    pub fn ffn_gate_up<T>(&self, p: T, distributed: bool) -> Tensor<T> {
        let &Self {
            nh,
            dh,
            di,
            distribute,
            ..
        } = self;
        let row = (di + di) / distribute;
        let col = nh * dh;
        self.mat(p, row, col, distributed)
    }

    pub fn ffn_down<T>(&self, p: T, distributed: bool) -> Tensor<T> {
        let &Self {
            nh,
            dh,
            di,
            distribute,
            ..
        } = self;
        let row = nh * dh;
        let col = di / distribute;
        self.mat(p, row, col, distributed)
    }

    pub fn output_norm<T>(&self, p: T) -> Tensor<T> {
        self.norm(p)
    }

    pub fn output<T>(&self, p: T) -> Tensor<T> {
        self.token_embd(p).transpose(&[1, 0])
    }

    fn norm<T>(&self, p: T) -> Tensor<T> {
        let &Self {
            dt_norm, nh, dh, ..
        } = self;
        Tensor::new(dt_norm, &[nh * dh], p)
    }

    fn mat<T>(&self, p: T, row: usize, col: usize, distributed: bool) -> Tensor<T> {
        let &Self {
            dt_mat, distribute, ..
        } = self;
        if distributed {
            Tensor::new(dt_mat, &[row, col], p).transpose(&[1, 0])
        } else {
            Tensor::new(dt_mat, &[distribute, row, col], p).transpose(&[2, 1])
        }
    }
}
