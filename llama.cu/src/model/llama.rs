use super::GGufModel;
use crate::utils::{Blob, Data, meta};
use ggus::{GGufMetaError, GGufMetaMapExt};
use log::info;
use nn::{
    Activation, Attention, Embedding, LLaMA, Linear, Mlp, NormType, Normalization, OutputHead,
    RoPE, Table, Tensor, TransformerBlk, digit_layout::types,
};

impl GGufModel<'_> {
    /// 构造 llama-like 模型
    pub fn llama(&self) -> nn::LLaMA<Tensor<&[u8], 2>> {
        let arch = meta![self => general_architecture];
        let dt_bias = match arch {
            "llama" => None,
            "qwen2" => Some(self.tensors["blk.0.attn_qkv.bias"].dt()),
            arch => panic!("unsupported arch {arch}"),
        };

        let nvoc = meta![self => tokenizer_ggml_tokens].len();
        let nctx = meta![self => llm_context_length];
        let nblk = meta![self => llm_block_count];
        let d = meta![self => llm_embedding_length];
        let nh = meta![self => llm_attention_head_count];
        let nkvh = meta![self => llm_attention_head_count_kv; nh];
        let dh = meta![self => llm_rope_dimension_count; d / nh];
        let di = meta![self => llm_feed_forward_length];
        let epsilon = meta![self => llm_attention_layer_norm_rms_epsilon; 1e-5];
        let dt_linear = self.tensors["blk.0.attn_qkv.weight"].dt();

        let get = |name: &str| self.tensors[name].as_deref();

        let token_embd = get("token_embd.weight");
        let out_norm = get("output_norm.weight");
        let out_linear = if self.tensors.contains_key("output.weight") {
            get("output.weight")
        } else {
            token_embd.clone()
        };

        LLaMA {
            embedding: Embedding {
                dt: token_embd.dt(),
                d,
                wte: Table {
                    row: nvoc,
                    weight: token_embd,
                },
                wpe: None,
            },
            blks: (0..nblk)
                .map(|iblk| {
                    TransformerBlk::new(
                        Normalization {
                            d,
                            epsilon: epsilon as _,
                            items: NormType::RmsNorm {
                                dt: out_norm.dt(),
                                scale: get(&format!("blk.{iblk}.attn_norm.weight")),
                            },
                        },
                        Attention {
                            nh,
                            nkvh,
                            qkv: Linear::new(
                                dt_linear,
                                [(nh + nkvh + nkvh) * dh, d],
                                get(&format!("blk.{iblk}.attn_qkv.weight")),
                                dt_bias.map(|dt| (dt, get(&format!("blk.{iblk}.attn_qkv.bias")))),
                            ),
                            rope: Some(RoPE {
                                multimodal: false,
                                nctx,
                                sin: get("sin_table"),
                                cos: get("cos_table"),
                            }),
                            output: Linear::new(
                                dt_linear,
                                [d, nh * dh],
                                get(&format!("blk.{iblk}.attn_output.weight")),
                                None,
                            ),
                        },
                        Normalization {
                            d,
                            epsilon: epsilon as _,
                            items: NormType::RmsNorm {
                                dt: out_norm.dt(),
                                scale: get(&format!("blk.{iblk}.ffn_norm.weight")),
                            },
                        },
                        Mlp {
                            up: Linear::new(
                                dt_linear,
                                [di * 2, d],
                                get(&format!("blk.{iblk}.ffn_gate_up.weight")),
                                None,
                            ),
                            act: Activation::SwiGLU,
                            down: Linear::new(
                                dt_linear,
                                [d, di],
                                get(&format!("blk.{iblk}.ffn_down.weight")),
                                None,
                            ),
                        },
                    )
                })
                .collect(),
            output_head: Some(OutputHead {
                out_norm: Normalization {
                    d,
                    epsilon: epsilon as _,
                    items: NormType::RmsNorm {
                        dt: out_norm.dt(),
                        scale: out_norm,
                    },
                },
                lm_head: Linear::new(out_linear.dt(), [nvoc, d], out_linear, None),
            }),
        }
    }

    /// 插入用于 RoPE 的 sin cos 表张量
    pub fn insert_rope_sin_cos(&mut self) {
        let nctx = meta![self => llm_context_length];
        let d = meta![self => llm_embedding_length];
        let nh = meta![self => llm_attention_head_count];
        let dh = meta![self => llm_rope_dimension_count; d / nh];
        let theta = meta![self => llm_rope_freq_base; 1e4];

        let arch = meta![self => general_architecture];
        let [sin, cos] = match self.get_str(&format!("{arch}.rope.scaling.type")) {
            Ok("longrope") => {
                let ctx_scale = 1.;

                let factors = &self.tensors["rope_factors_long.weight"];
                assert_eq!(factors.dt(), types::F32);
                assert_eq!(factors.shape(), [dh / 2]);
                let factors = unsafe {
                    std::slice::from_raw_parts(factors.get().as_ptr().cast::<f32>(), dh / 2)
                };

                info!("detected longrope, ctx scale = {ctx_scale}, scale factor = {factors:.2?}");
                build_sin_cos(nctx, dh, theta, |pos, i| {
                    pos as f32 * ctx_scale / factors[i]
                })
            }
            Err(GGufMetaError::NotExist) => build_sin_cos(nctx, dh, theta, |pos, _| pos as _),
            Ok(ty) => panic!("Unsupported rope scaling `{ty}`"),
            Err(e) => panic!("{e:?}"),
        };

        self.tensors.insert("sin_table", sin);
        self.tensors.insert("cos_table", cos);
    }

    /// 构造语言模型的 kv cache 张量
    pub fn lm_kv_cache<const N: usize>(&self) -> Tensor<usize, N> {
        let dt = self.tensors["token_embd.weight"].dt();
        let nblk = meta![self => llm_block_count];
        let nctx = meta![self => llm_context_length];
        let d = meta![self => llm_embedding_length];
        let nh = meta![self => llm_attention_head_count];
        let nkvh = meta![self => llm_attention_head_count_kv; nh];
        let dh = meta![self => llm_rope_dimension_count; d / nh];
        Tensor::from_dim_slice(dt, [nctx, nblk, 2, nkvh, dh])
    }
}

/// 构造 sin cos 表张量
fn build_sin_cos<'a, const N: usize>(
    nctx: usize,
    dh: usize,
    theta: f32,
    mut pos_scaling: impl FnMut(usize, usize) -> f32,
) -> [Tensor<Data<'a>, N>; 2] {
    let d = dh / 2;
    let ty = types::F32;
    let mut sin = Blob::new(nctx * d * ty.nbytes());
    let mut cos = Blob::new(nctx * d * ty.nbytes());
    let theta = theta.powf(-(d as f32).recip());

    {
        let ([], sin, []) = (unsafe { sin.align_to_mut() }) else {
            unreachable!()
        };
        let ([], cos, []) = (unsafe { cos.align_to_mut() }) else {
            unreachable!()
        };
        for pos in 0..nctx {
            for i in 0..d {
                let (sin_, cos_) = (pos_scaling(pos, i) * theta.powi(i as _)).sin_cos();
                sin[pos * d + i] = sin_;
                cos[pos * d + i] = cos_;
            }
        }
    }

    let tensor = |data: Blob| {
        Tensor::from_dim_slice(ty, [nctx, d]).map(|len| {
            assert_eq!(len, data.len());
            data.into()
        })
    };
    [tensor(sin), tensor(cos)]
}
