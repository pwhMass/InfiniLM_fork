mod args;
mod compute;
mod random_sample;
mod storage;

pub use args::{Args as LlamaArgs, Request as LlamaRequest};
pub use compute::{BlkWeight, LlamaWorker, Operators, WeightLoader};
pub use random_sample::RandomSample;
pub use storage::{BlkStorage as LlamaBlkStorage, Storage as LlamaStorage};
pub use tensor::Tensor;
pub mod ext {
    pub use gguf::{
        ext::Mmap,
        ggml_quants::{
            digit_layout::{types as primitive, DigitLayout},
            f16, types as quant,
        },
    };
}

#[derive(Clone, Debug)]
pub struct LlamaMeta {
    pub dt_norm: ext::DigitLayout,
    pub dt_mat: ext::DigitLayout,
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
    pub fn kv_cache(&self, buf: usize) -> Tensor<usize> {
        let &Self {
            dt_mat,
            nblk,
            nkvh,
            dh,
            distribute,
            ..
        } = self;
        Tensor::new(dt_mat, &[buf, nblk, 2, nkvh / distribute, dh])
    }

    pub fn embd(&self, nt: usize) -> Tensor<usize> {
        let &Self { dt_mat, nh, dh, .. } = self;
        Tensor::new(dt_mat, &[nt, nh * dh])
    }

    pub fn logits(&self, nt: usize) -> Tensor<usize> {
        let &Self { dt_mat, dvoc, .. } = self;
        Tensor::new(dt_mat, &[nt, dvoc])
    }

    pub fn token_embd(&self) -> Tensor<usize> {
        let &Self {
            dt_mat,
            nh,
            dh,
            dvoc,
            ..
        } = self;
        Tensor::new(dt_mat, &[dvoc, nh * dh])
    }

    pub fn attn_norm(&self) -> Tensor<usize> {
        self.norm()
    }

    pub fn attn_qkv(&self, distributed: bool) -> Tensor<usize> {
        let &Self {
            nh,
            nkvh,
            dh,
            distribute,
            ..
        } = self;
        let row = (nh + nkvh + nkvh) / distribute * dh;
        let col = nh * dh;
        self.mat(row, col, distributed)
    }

    pub fn attn_o(&self, distributed: bool) -> Tensor<usize> {
        let &Self {
            nh, dh, distribute, ..
        } = self;
        let row = nh * dh;
        let col = nh / distribute * dh;
        self.mat(row, col, distributed)
    }

    pub fn ffn_norm(&self) -> Tensor<usize> {
        self.norm()
    }

    pub fn ffn_gate_up(&self, distributed: bool) -> Tensor<usize> {
        let &Self {
            nh,
            dh,
            di,
            distribute,
            ..
        } = self;
        let row = (di + di) / distribute;
        let col = nh * dh;
        self.mat(row, col, distributed)
    }

    pub fn ffn_down(&self, distributed: bool) -> Tensor<usize> {
        let &Self {
            nh,
            dh,
            di,
            distribute,
            ..
        } = self;
        let row = nh * dh;
        let col = di / distribute;
        self.mat(row, col, distributed)
    }

    pub fn output_norm(&self) -> Tensor<usize> {
        self.norm()
    }

    pub fn output(&self) -> Tensor<usize> {
        self.token_embd().transpose(&[1, 0])
    }

    fn norm(&self) -> Tensor<usize> {
        let &Self {
            dt_norm, nh, dh, ..
        } = self;
        Tensor::new(dt_norm, &[nh * dh])
    }

    fn mat(&self, row: usize, col: usize, distributed: bool) -> Tensor<usize> {
        let &Self {
            dt_mat, distribute, ..
        } = self;
        if distributed {
            Tensor::new(dt_mat, &[row, col]).transpose(&[1, 0])
        } else {
            Tensor::new(dt_mat, &[distribute, row, col]).transpose(&[2, 1])
        }
    }
}
