mod args;
mod compute;
mod storage;

pub use args::{Args as LlamaArgs, Request as LlamaRequest};
pub use compute::{BlkWeight, LlamaWorker, Operators, WeightLoader};
pub use storage::{BlkStorage as LlamaBlkStorage, Storage as LlamaStorage};
pub use tensor::{RandomSample, Tensor};
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
    pub dt_embd: ext::DigitLayout,
    pub dt_norm: ext::DigitLayout,
    pub dt_mat: ext::DigitLayout,

    pub nblk: usize,
    pub nctx: usize,
    pub nvoc: usize,
    pub nh: usize,
    pub nkvh: usize,
    pub d: usize,
    pub dh: usize,
    pub di: usize,

    pub epsilon: f32,
    pub theta: f32,
}

impl LlamaMeta {
    pub fn kv_cache(&self, buf: usize) -> Tensor<usize> {
        let &Self {
            dt_mat,
            nblk,
            nkvh,
            dh,
            ..
        } = self;
        Tensor::new(dt_mat, &[buf, nblk, 2, nkvh, dh])
    }

    pub fn embd(&self, nt: usize) -> Tensor<usize> {
        let &Self { dt_embd, d, .. } = self;
        Tensor::new(dt_embd, &[nt, d])
    }

    pub fn logits(&self, nt: usize) -> Tensor<usize> {
        let &Self { dt_embd, nvoc, .. } = self;
        Tensor::new(dt_embd, &[nt, nvoc])
    }

    pub fn token_embd(&self) -> Tensor<usize> {
        self.embd(self.nvoc)
    }

    pub fn attn_norm(&self) -> Tensor<usize> {
        self.norm()
    }

    pub fn attn_qkv(&self) -> Tensor<usize> {
        let &Self {
            nh, nkvh, d, dh, ..
        } = self;
        self.mat((nh + nkvh + nkvh) * dh, d)
    }

    pub fn attn_o(&self) -> Tensor<usize> {
        let &Self { nh, d, dh, .. } = self;
        self.mat(d, nh * dh)
    }

    pub fn ffn_norm(&self) -> Tensor<usize> {
        self.norm()
    }

    pub fn ffn_gate_up(&self) -> Tensor<usize> {
        let &Self { d, di, .. } = self;
        self.mat(di + di, d)
    }

    pub fn ffn_down(&self) -> Tensor<usize> {
        let &Self { d, di, .. } = self;
        self.mat(d, di)
    }

    pub fn output_norm(&self) -> Tensor<usize> {
        self.norm()
    }

    pub fn output(&self) -> Tensor<usize> {
        self.token_embd().transpose(&[1, 0])
    }

    fn norm(&self) -> Tensor<usize> {
        let &Self { dt_norm, d, .. } = self;
        Tensor::new(dt_norm, &[d])
    }

    fn mat(&self, row: usize, col: usize) -> Tensor<usize> {
        Tensor::new(self.dt_mat, &[row, col]).transpose(&[1, 0])
    }
}
