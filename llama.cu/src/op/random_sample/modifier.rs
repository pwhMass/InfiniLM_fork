//! <https://zhuanlan.zhihu.com/p/667025336>

use crate::utils::offset_ptr;
use ggus::ggml_quants::f16;
use log::warn;
use nn::Tensor;
use operators::cuda::{CurrentCtx, DevByte, DevMem, Module, Ptx, Stream, VirByte, params};
use std::ffi::c_uint;
use tokeneer::utok;

pub(crate) struct LogitsModifier<'ctx> {
    module: Module<'ctx>,
    n: usize,
    eos: utok,
}

impl<'ctx> LogitsModifier<'ctx> {
    pub fn new(n: usize, eos: utok, ctx: &'ctx CurrentCtx) -> Self {
        Self {
            module: Self::compile(ctx),
            n,
            eos,
        }
    }
}

impl LogitsModifier<'_> {
    pub fn new_state<'ctx>(&self, stream: &Stream<'ctx>) -> DevMem<'ctx> {
        stream.malloc::<f16>(self.n)
    }

    pub unsafe fn next<const N: usize>(
        &self,
        logits: &Tensor<*const VirByte, N>,
        scale: *mut DevByte,
        tok: *const DevByte,
        mut temperature: f32,
        penalty: f32,
        stream: &Stream,
    ) {
        let n = self.n as c_uint;
        if temperature == 0. {
            temperature = 1.
        }
        stream.launch(
            &self.module.get_kernel(c"next"),
            (n.div_ceil(256), 256, 0),
            &params![
                offset_ptr(logits),
                scale,
                n,
                temperature,
                penalty.recip(),
                tok,
                self.eos
            ]
            .to_ptrs(),
        );
    }

    fn compile<'ctx>(ctx: &'ctx CurrentCtx) -> Module<'ctx> {
        const CODE: &str = include_str!("modify.cuh");
        let code = format!(
            r#"
{CODE}

extern "C" __global__ void next(
    half *logits,
    half *scale,
    unsigned int const n,
    float const temperature,
    float const penalty,
    unsigned int const *tok,
    unsigned int const eos
) {{
    next_kernel(logits, scale, n, temperature, penalty, tok, eos);
}}"#
        );
        let (ptx, log) = Ptx::compile(code, ctx.dev().compute_capability());
        match ptx {
            Ok(ptx) => {
                if !log.is_empty() {
                    warn!("{log}")
                }
                ctx.load(&ptx)
            }
            Err(e) => panic!("logits modify compilation failed with {e:?}, log:\n {log}"),
        }
    }
}
