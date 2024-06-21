use crate::{InferenceConfig, LayerStorage, Storage, Weight};
use common::{bf16, f16, Blob};
use digit_layout::{
    types::{BF16, F16, F32},
    AsDigit, DigitLayout,
};
use tensor::Tensor;

impl Storage {
    pub fn cast(self, dt: DigitLayout) -> Self {
        if self.config.dt == dt {
            return self;
        }
        Self {
            config: InferenceConfig { dt, ..self.config },
            embed_tokens: cast(self.embed_tokens, dt),
            layers: self
                .layers
                .into_iter()
                .map(|l| LayerStorage {
                    att_layernorm: cast(l.att_layernorm, dt),
                    att_qkv: cast(l.att_qkv, dt),
                    att_o: cast(l.att_o, dt),
                    mlp_layernorm: cast(l.mlp_layernorm, dt),
                    mlp_gate_up: cast(l.mlp_gate_up, dt),
                    mlp_down: cast(l.mlp_down, dt),
                })
                .collect(),
            lm_layernorm: cast(self.lm_layernorm, dt),
            lm_head: cast(self.lm_head, dt),
        }
    }
}

fn cast(src: Tensor<Weight>, dt: DigitLayout) -> Tensor<Weight> {
    match (src.data_layout(), dt) {
        (F16, BF16) => typed(src, |x: &f16| bf16::from_f32(x.to_f32())),
        (F16, F32) => typed(src, |x: &f16| x.to_f32()),
        (BF16, F16) => typed(src, |x: &bf16| f16::from_f32(x.to_f32())),
        (BF16, F32) => typed(src, |x: &bf16| x.to_f32()),
        (F32, F16) => typed(src, |x: &f32| f16::from_f32(*x)),
        (F32, BF16) => typed(src, |x: &f32| bf16::from_f32(*x)),
        _ => todo!(),
    }
}

fn typed<T: AsDigit + Sync, U: AsDigit + Send>(
    src: Tensor<Weight>,
    cast: impl Fn(&T) -> U + Sync,
) -> Tensor<Weight> {
    use rayon::iter::*;
    use tensor::{reslice, reslice_mut};

    assert_eq!(src.data_layout(), T::LAYOUT);
    let mut ans = Tensor::alloc(U::LAYOUT, src.shape(), Blob::new);

    reslice(src.physical())
        .par_iter()
        .zip(reslice_mut(ans.physical_mut()))
        .for_each(|(src, dst)| *dst = cast(src));

    ans.map_physical(|b| b.into())
}
