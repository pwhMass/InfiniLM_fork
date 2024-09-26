use operators::Hardware;
use tensor::Tensor;

pub struct Args<'a, H: Hardware> {
    /// shape: [nt, nh x dh]
    pub embd: Tensor<&'a mut [H::Byte]>,
    /// shape: [_, dh]
    pub sin: Tensor<&'a [H::Byte]>,
    /// shape: [_, dh]
    pub cos: Tensor<&'a [H::Byte]>,
    /// shape: [n_out, dvoc]
    pub logits: Tensor<&'a mut [H::Byte]>,

    pub requests: Vec<Request<'a, H>>,

    pub num_tokens: usize,
    pub max_seq_len: usize,
    pub max_att_len: usize,

    pub mlp_alpha: f32,
    pub residual: bool,
}

pub struct Request<'a, H: Hardware> {
    /// shape: [buf, nblk, 2, nkvh, dh]
    pub cache: Tensor<&'a mut [H::Byte]>,
    pub buf_len: usize,
    pub seq_len: usize,
    pub out_len: usize,
    pub pos: usize,
}
