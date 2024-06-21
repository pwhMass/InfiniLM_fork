use common::utok;
use digit_layout::{
    types::{BF16, F16, F32},
    DigitLayout,
};

#[derive(serde::Serialize, serde::Deserialize, Debug)]
pub(crate) struct ConfigJson {
    pub bos_token_id: utok,
    pub eos_token_id: utok,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub max_position_embeddings: usize,
    pub num_attention_heads: usize,
    pub num_hidden_layers: usize,
    pub num_key_value_heads: usize,
    pub vocab_size: usize,
    #[serde(default = "default_rms_norm_eps")]
    pub rms_norm_eps: f32,
    #[serde(default = "default_rope_theta")]
    pub rope_theta: f32,
    pub torch_dtype: String,
}

impl ConfigJson {
    pub fn data_layout(&self) -> DigitLayout {
        match self.torch_dtype.as_str() {
            "float16" => F16,
            "float32" => F32,
            "bfloat16" => BF16,
            _ => todo!(),
        }
    }
}

pub(crate) fn data_layout_name(layout: DigitLayout) -> &'static str {
    match layout {
        F16 => "float16",
        F32 => "float32",
        BF16 => "bfloat16",
        _ => todo!(),
    }
}

#[inline(always)]
const fn default_rms_norm_eps() -> f32 {
    1e-5
}

#[inline(always)]
const fn default_rope_theta() -> f32 {
    1e4
}
