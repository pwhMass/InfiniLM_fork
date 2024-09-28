use super::{args::Args, LlamaMeta};
use gguf::ggml_quants::digit_layout::types as ty;
use itertools::izip;
use operators::{
    attention_kv_cached::AttnKVCached,
    mat_mul::MatMul,
    mlp::Mlp,
    rearrange::Rearrange,
    rms_norm::RmsNorm,
    rope::{Rope, Seq},
    ByteOf, Hardware, LaunchError, Operator, QueueAlloc, QueueOf, Workspace,
};
use std::ops::{Deref, DerefMut};
use tensor::{split, Tensor};

pub trait Operators {
    type Hardware: Hardware;
    type RmsNorm: RmsNorm<Self::Hardware>;
    type MatMul: MatMul<Self::Hardware>;
    type Rope: Rope<Self::Hardware>;
    type AttnKVCached: AttnKVCached<Self::Hardware>;
    type Mlp: Mlp<Self::Hardware>;
    type Rearrange: Rearrange<Self::Hardware>;

    fn debug<T>(tensor: &Tensor<T>)
    where
        T: Deref<Target = [ByteOf<Self::Hardware>]>;
}

pub enum BlkWeight {
    AttnNorm,
    AttnQKV,
    AttnO,
    FfnNorm,
    FfnGateUp,
    FfnDown,
}

pub trait WeightLoader {
    type Hardware: Hardware;
    type Memory: Deref<Target = [ByteOf<Self::Hardware>]>;

    fn load_blk(
        &self,
        which: BlkWeight,
        iblk: usize,
        queue: &QueueOf<Self::Hardware>,
    ) -> Self::Memory;

    fn output_norm(&self, queue: &QueueOf<Self::Hardware>) -> Self::Memory;
    fn output(&self, queue: &QueueOf<Self::Hardware>) -> Self::Memory;
}

pub struct LlamaBlks<H, W, Ops>
where
    H: Hardware,
    Ops: Operators<Hardware = H>,
{
    meta: LlamaMeta,
    weights: WeightDecorator<W>,
    rms_norm: Ops::RmsNorm,
    mat_mul: Ops::MatMul,
    rope: Ops::Rope,
    attn_kv_cached: Ops::AttnKVCached,
    mlp: Ops::Mlp,
    rearrange: Ops::Rearrange,
}

impl<H, W, Ops> LlamaBlks<H, W, Ops>
where
    H: Hardware,
    Ops: Operators<Hardware = H>,
{
    pub fn new(processor: &H, meta: LlamaMeta, weights: W) -> Self {
        Self {
            weights: meta.decorator(weights),
            meta,
            rms_norm: Ops::RmsNorm::new(processor),
            mat_mul: Ops::MatMul::new(processor),
            rope: Ops::Rope::new(processor),
            attn_kv_cached: Ops::AttnKVCached::new(processor),
            mlp: Ops::Mlp::new(processor),
            rearrange: Ops::Rearrange::new(processor),
        }
    }

    #[inline]
    pub const fn meta(&self) -> &LlamaMeta {
        &self.meta
    }
}

impl<H, W, Ops> LlamaBlks<H, W, Ops>
where
    H: Hardware,
    H::Byte: 'static,
    W: WeightLoader<Hardware = H>,
    Ops: Operators<Hardware = H>,
{
    pub fn launch<QA>(
        &self,
        args: Args<H>,
        workspace: &mut [ByteOf<H>],
        queue_alloc: &QA,
    ) -> Result<(), LaunchError>
    where
        QA: QueueAlloc<Hardware = H>,
    {
        let Args {
            embd,
            sin,
            cos,
            mut logits,
            mut requests,
            num_tokens: nt,
            max_seq_len,
            max_att_len,
            mlp_alpha,
        } = args;
        let LlamaMeta {
            dt_mat,
            nblk,
            nh,
            nkvh,
            dh,
            di,
            distribute,
            ..
        } = self.meta;
        let d = nh * dh;
        let nh = nh / distribute;
        let nkvh = nkvh / distribute;
        let di = di / distribute;

        let ele = dt_mat.nbytes().unwrap();
        let embd_size = nt * d * ele;
        let qkv_size = nt * (nh + nkvh + nkvh) * dh * ele;
        let gate_up_size = nt * di * 2 * ele;
        let q_size = max_seq_len * nh * dh * ele;
        let att_size = nkvh * max_seq_len * max_att_len * ele;
        let workspace_size = embd_size + qkv_size.max(gate_up_size) + q_size + att_size;

        let mut workspace = Workspace::new(queue_alloc, workspace, workspace_size);

        let mut x = embd;
        let (x1, workspace) = workspace.split_at_mut(embd_size);
        let mut x1 = Tensor::new(dt_mat, x.shape(), x1);

        let pos = Tensor::new(
            ty::U32,
            &[nt],
            Ops::Rope::build_pos(
                nt,
                requests.iter().map(|req| Seq {
                    pos: req.pos,
                    len: req.seq_len,
                }),
                queue_alloc,
            ),
        );

        let req_split = requests.iter().map(|req| req.seq_len).collect::<Vec<_>>();

        let queue = queue_alloc.queue();
        for iblk in 0..nblk {
            {
                let w = self.weights.attn_norm(iblk, queue);
                self.rms_norm(&mut x1, &x, &w, workspace, queue_alloc)?;

                let (qkv, workspace) = workspace.split_at_mut(qkv_size);
                let mut qkv = Tensor::new(dt_mat, &[nt, (nh + nkvh + nkvh) * dh], qkv);

                let w = self.weights.attn_qkv(iblk, queue);
                self.mat_mul(&mut qkv, 0., &x1, &w, 1., workspace, queue_alloc)?;

                let qkv = qkv.tile(1, &[nh + nkvh + nkvh, dh]);

                split!(qkv => q, k, v; [nh, nkvh, nkvh] @ 1);
                let mut q = q;
                let mut k = k;
                let v = v;
                let o = x1
                    .as_mut()
                    .tile(1, &[nh * distribute, dh])
                    .map(|t| &mut t[..]);

                self.rope(&mut q, &pos, &sin, &cos, workspace, queue_alloc)?;
                self.rope(&mut k, &pos, &sin, &cos, workspace, queue_alloc)?;

                let q = q.transpose(&[1, 0]);
                let k = k.transpose(&[1, 0]);
                let v = v.transpose(&[1, 0]);
                let o = o.transpose(&[1, 0]);
                let q = q.split(1, &req_split);
                let k = k.split(1, &req_split);
                let v = v.split(1, &req_split);
                let o = o.split(1, &req_split);

                for (mut q, k, v, mut o, req) in izip!(q, k, v, o, &mut requests) {
                    let cache = req
                        .cache
                        .as_mut() // [buf, nblk, 2, nkvh, dh]
                        .index(1, iblk) // [buf, 2, nkvh, dh]
                        .transpose(&[2, 0]) // [nkvh, 2, buf, dh]
                        .map(|t| &mut t[..]);

                    split!(cache => kc, vc; [1, 1] @ 1);
                    self.attn_kv_cached(
                        &mut q,
                        &k,
                        &v,
                        &mut o,
                        &mut kc.index(1, 0),
                        &mut vc.index(1, 0),
                        req.pos,
                        workspace,
                        queue_alloc,
                    )?;
                }
            }

            let w = self.weights.attn_o(iblk, queue);
            self.mat_mul(&mut x, 1., &x1, &w, 1., workspace, queue_alloc)?;

            if distribute > 1 {
                todo!("all reduce")
            }

            let w = self.weights.ffn_norm(iblk, queue);
            self.rms_norm(&mut x1, &x, &w, workspace, queue_alloc)?;

            #[rustfmt::skip]
            self.mlp(&mut x, &x1, iblk, mlp_alpha, true, workspace, queue_alloc)?;

            if distribute > 1 {
                todo!("all reduce")
            }
        }

        // 集中要采样的 token
        // NOTICE: 输入之前将请求按 seq len 升序排列可降低移动开销
        let mut dst = 0;
        let mut src = 0;
        for req in &requests {
            src += req.seq_len;
            for src in src - req.out_len..src {
                if src != dst {
                    let src = unsafe { x.map_slice_static() }.index(0, src);
                    let mut dst = x.map_slice_mut().index(0, dst);
                    self.rearrange(&mut dst, &src, workspace, queue_alloc)?;
                }
                dst += 1;
            }
        }
        assert_eq!(dst, logits.shape()[0]);

        let w = self.weights.output_norm(queue);
        let mut x = x.map_slice_mut().slice(0, 0, 1, dst);
        let x_ = unsafe { x.map_slice_static() };
        self.rms_norm(&mut x, &x_, &w, workspace, queue_alloc)?;

        let output = self.weights.output(queue);
        self.mat_mul(&mut logits, 0., &x, &output, 1., workspace, queue_alloc)
    }
}

#[allow(clippy::too_many_arguments)]
impl<H, W, Ops> LlamaBlks<H, W, Ops>
where
    H: Hardware,
    W: WeightLoader<Hardware = H>,
    Ops: Operators<Hardware = H>,
{
    fn rms_norm<Y, X, W_, QA>(
        &self,
        y: &mut Tensor<Y>,
        x: &Tensor<X>,
        w: &Tensor<W_>,
        workspace: &mut [H::Byte],
        queue_alloc: &QA,
    ) -> Result<(), LaunchError>
    where
        Y: DerefMut<Target = [H::Byte]>,
        X: Deref<Target = [H::Byte]>,
        W_: Deref<Target = [H::Byte]>,
        QA: QueueAlloc<Hardware = H>,
    {
        self.rms_norm.launch(
            &operators::rms_norm::Args {
                y_layout: y.layout(),
                y_base: y.base_mut(),
                x_layout: x.layout(),
                x_base: x.base(),
                w_layout: w.layout(),
                w_base: w.base(),
                epsilon: self.meta.epsilon,
            },
            workspace,
            queue_alloc,
        )
    }

    fn mat_mul<C, A, B, QA>(
        &self,
        c: &mut Tensor<C>,
        beta: f32,
        a: &Tensor<A>,
        b: &Tensor<B>,
        alpha: f32,
        workspace: &mut [H::Byte],
        queue_alloc: &QA,
    ) -> Result<(), LaunchError>
    where
        C: DerefMut<Target = [H::Byte]>,
        A: Deref<Target = [H::Byte]>,
        B: Deref<Target = [H::Byte]>,
        QA: QueueAlloc<Hardware = H>,
    {
        self.mat_mul.launch(
            &operators::mat_mul::Args {
                c_layout: c.layout(),
                c_base: c.base_mut(),
                beta,
                a_layout: a.layout(),
                a_base: a.base(),
                b_layout: b.layout(),
                b_base: b.base(),
                alpha,
            },
            workspace,
            queue_alloc,
        )
    }

    fn rope<T, P, Sin, Cos, QA>(
        &self,
        t: &mut Tensor<T>,
        p: &Tensor<P>,
        sin: &Tensor<Sin>,
        cos: &Tensor<Cos>,
        workspace: &mut [H::Byte],
        queue_alloc: &QA,
    ) -> Result<(), LaunchError>
    where
        T: DerefMut<Target = [H::Byte]>,
        P: Deref<Target = [H::Byte]>,
        Sin: Deref<Target = [H::Byte]>,
        Cos: Deref<Target = [H::Byte]>,
        QA: QueueAlloc<Hardware = H>,
    {
        self.rope.launch(
            &operators::rope::Args {
                t_layout: t.layout(),
                t_base: t.base_mut(),
                p_layout: p.layout(),
                p_base: p.base(),
                sin_layout: sin.layout(),
                sin_base: sin.base(),
                cos_layout: cos.layout(),
                cos_base: cos.base(),
                theta: self.meta.theta,
            },
            workspace,
            queue_alloc,
        )
    }

    fn attn_kv_cached<Q, K, V, O, KC, VC, QA>(
        &self,
        q: &mut Tensor<Q>,
        k: &Tensor<K>,
        v: &Tensor<V>,
        o: &mut Tensor<O>,
        kc: &mut Tensor<KC>,
        vc: &mut Tensor<VC>,
        pos: usize,
        workspace: &mut [H::Byte],
        queue_alloc: &QA,
    ) -> Result<(), LaunchError>
    where
        Q: DerefMut<Target = [H::Byte]>,
        K: Deref<Target = [H::Byte]>,
        V: Deref<Target = [H::Byte]>,
        O: DerefMut<Target = [H::Byte]>,
        KC: DerefMut<Target = [H::Byte]>,
        VC: DerefMut<Target = [H::Byte]>,
        QA: QueueAlloc<Hardware = H>,
    {
        self.attn_kv_cached.launch(
            &operators::attention_kv_cached::Args {
                q_layout: q.layout(),
                q_base: q.base_mut(),
                k_layout: k.layout(),
                k_base: k.base(),
                v_layout: v.layout(),
                v_base: v.base(),
                o_layout: o.layout(),
                o_base: o.base_mut(),
                k_cache_layout: kc.layout(),
                k_cache_base: kc.base_mut(),
                v_cache_layout: vc.layout(),
                v_cache_base: vc.base_mut(),
                pos: pos.into(),
            },
            workspace,
            queue_alloc,
        )
    }

    fn mlp<Y, X, QA>(
        &self,
        y: &mut Tensor<Y>,
        x: &Tensor<X>,
        iblk: usize,
        down_alpha: f32,
        residual: bool,
        workspace: &mut [H::Byte],
        queue_alloc: &QA,
    ) -> Result<(), LaunchError>
    where
        Y: DerefMut<Target = [H::Byte]>,
        X: Deref<Target = [H::Byte]>,
        QA: QueueAlloc<Hardware = H>,
    {
        let queue = queue_alloc.queue();
        let w_gate_up = self.weights.ffn_gate_up(iblk, queue);
        let w_down = self.weights.ffn_down(iblk, queue);

        self.mlp.launch(
            &operators::mlp::Args {
                y_layout: y.layout(),
                y_base: y.base_mut(),
                x_layout: x.layout(),
                x_base: x.base(),
                w_gate_up_layout: w_gate_up.layout(),
                w_gate_up_base: w_gate_up.base(),
                w_down_layout: w_down.layout(),
                w_down_base: w_down.base(),
                down_alpha,
                residual,
            },
            workspace,
            queue_alloc,
        )
    }

    fn rearrange<Y, X, QA>(
        &self,
        dst: &mut Tensor<Y>,
        src: &Tensor<X>,
        workspace: &mut [H::Byte],
        queue_alloc: &QA,
    ) -> Result<(), LaunchError>
    where
        Y: DerefMut<Target = [H::Byte]>,
        X: Deref<Target = [H::Byte]>,
        QA: QueueAlloc<Hardware = H>,
    {
        self.rearrange.launch(
            &operators::rearrange::Args {
                dst_layout: dst.layout(),
                dst_base: dst.base_mut(),
                src_layout: src.layout(),
                src_base: src.base(),
            },
            workspace,
            queue_alloc,
        )
    }
}

struct WeightDecorator<W> {
    attn_norm: Tensor<()>,
    attn_qkv: Tensor<()>,
    attn_o: Tensor<()>,
    ffn_norm: Tensor<()>,
    ffn_gate_up: Tensor<()>,
    ffn_down: Tensor<()>,
    output_norm: Tensor<()>,
    output: Tensor<()>,
    weights: W,
}

impl LlamaMeta {
    fn decorator<W>(&self, weights: W) -> WeightDecorator<W> {
        WeightDecorator {
            attn_norm: self.attn_norm(()),
            attn_qkv: self.attn_qkv((), true),
            attn_o: self.attn_o((), true),
            ffn_norm: self.ffn_norm(()),
            ffn_gate_up: self.ffn_gate_up((), true),
            ffn_down: self.ffn_down((), true),
            output_norm: self.output_norm(()),
            output: self.output(()),
            weights,
        }
    }
}

impl<W: WeightLoader> WeightDecorator<W> {
    #[inline]
    pub fn attn_norm(&self, iblk: usize, queue: &QueueOf<W::Hardware>) -> Tensor<W::Memory> {
        combine(
            &self.attn_norm,
            self.weights.load_blk(BlkWeight::AttnNorm, iblk, queue),
        )
    }

    #[inline]
    pub fn attn_qkv(&self, iblk: usize, queue: &QueueOf<W::Hardware>) -> Tensor<W::Memory> {
        combine(
            &self.attn_qkv,
            self.weights.load_blk(BlkWeight::AttnQKV, iblk, queue),
        )
    }

    #[inline]
    pub fn attn_o(&self, iblk: usize, queue: &QueueOf<W::Hardware>) -> Tensor<W::Memory> {
        combine(
            &self.attn_o,
            self.weights.load_blk(BlkWeight::AttnO, iblk, queue),
        )
    }

    #[inline]
    pub fn ffn_norm(&self, iblk: usize, queue: &QueueOf<W::Hardware>) -> Tensor<W::Memory> {
        combine(
            &self.ffn_norm,
            self.weights.load_blk(BlkWeight::FfnNorm, iblk, queue),
        )
    }

    #[inline]
    pub fn ffn_gate_up(&self, iblk: usize, queue: &QueueOf<W::Hardware>) -> Tensor<W::Memory> {
        combine(
            &self.ffn_gate_up,
            self.weights.load_blk(BlkWeight::FfnGateUp, iblk, queue),
        )
    }

    #[inline]
    pub fn ffn_down(&self, iblk: usize, queue: &QueueOf<W::Hardware>) -> Tensor<W::Memory> {
        combine(
            &self.ffn_down,
            self.weights.load_blk(BlkWeight::FfnDown, iblk, queue),
        )
    }

    #[inline]
    pub fn output_norm(&self, queue: &QueueOf<W::Hardware>) -> Tensor<W::Memory> {
        combine(&self.output_norm, self.weights.output_norm(queue))
    }

    #[inline]
    pub fn output(&self, queue: &QueueOf<W::Hardware>) -> Tensor<W::Memory> {
        combine(&self.output, self.weights.output(queue))
    }
}

#[inline]
fn combine<T>(tensor: &Tensor<()>, p: T) -> Tensor<T> {
    tensor.clone().map(|()| p)
}
