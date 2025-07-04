use super::GGufModel;
use crate::utils::meta;
use ggus::GGufMetaMapExt;
use nn::{
    Activation, Attention, Linear, Merger, Mlp, NormType, Normalization, PatchEmbd, Qwen2VLmmproj,
    RoPE, Tensor, TransformerBlk,
};

impl GGufModel<'_> {
    /// 构造 qw2vl_mmproj 模型
    pub fn _qw2vl_mmproj(&self) -> nn::Qwen2VLmmproj<Tensor<&[u8], 2>> {
        let nblk = meta![self => llm_block_count];
        let d = meta![self => llm_embedding_length];
        let nh = meta![self => llm_attention_head_count];
        let nkvh = meta![self => llm_attention_head_count_kv; nh];
        let dh = meta![self => llm_rope_dimension_count; d / nh];
        let _di = meta![self => llm_feed_forward_length];
        let epsilon = meta![self => llm_attention_layer_norm_epsilon; 1e-6];
        let d_patch = 14; // ggus todo
        let d_proj = 1536;
        let dt = self.tensors["v.blk.0.attn_qkv.weight"].dt();
        let dt_norm = self.tensors["v.blk.0.ln1.weight"].dt();

        let get = |name: &str| self.tensors[name].as_deref();

        Qwen2VLmmproj {
            patch_embd: PatchEmbd {
                dt,
                shape: [d, 3, d_patch, d_patch],
                patch_embd: get("v.patch_embd.weight"),
                patch_embd1: get("v.patch_embd.weight.1"),
            },
            vision_blks: (0..nblk)
                .map(|iblk| {
                    TransformerBlk::new(
                        Normalization {
                            d,
                            epsilon: epsilon as _,
                            items: NormType::LayerNorm {
                                dt_scale: dt_norm,
                                scale: get(&format!("v.blk.{iblk}.ln1.weight")),
                                dt_bias: dt_norm,
                                bias: get(&format!("v.blk.{iblk}.ln1.bias")),
                            },
                        },
                        Attention {
                            nh,
                            nkvh,
                            qkv: Linear::new(
                                dt,
                                [(nh + nkvh + nkvh) * dh, d],
                                get(&format!("v.blk.{iblk}.attn_qkv.weight")),
                                Some((dt_norm, get(&format!("v.blk.{iblk}.attn_qkv.bias")))),
                            ),
                            rope: Some(RoPE {
                                multimodal: true,
                                nctx: 34, // todo: from image
                                sin: get("sin_table"),
                                cos: get("cos_table"),
                            }),
                            output: Linear::new(
                                dt,
                                [d, nh * dh],
                                get(&format!("v.blk.{iblk}.attn_out.weight")),
                                Some((dt_norm, get(&format!("v.blk.{iblk}.attn_out.bias")))),
                            ),
                        },
                        Normalization {
                            d,
                            epsilon: epsilon as _,
                            items: NormType::LayerNorm {
                                dt_scale: dt_norm,
                                scale: get(&format!("v.blk.{iblk}.ln2.weight")),
                                dt_bias: dt_norm,
                                bias: get(&format!("v.blk.{iblk}.ln2.bias")),
                            },
                        },
                        Mlp {
                            up: Linear::new(
                                dt,
                                [d * 4, d],
                                get(&format!("v.blk.{iblk}.ffn_up.weight")),
                                Some((dt_norm, get(&format!("v.blk.{iblk}.ffn_up.bias")))),
                            ),
                            act: Activation::GeLU,
                            down: Linear::new(
                                dt,
                                [d, d * 4],
                                get(&format!("v.blk.{iblk}.ffn_down.weight")),
                                Some((dt_norm, get(&format!("v.blk.{iblk}.ffn_down.bias")))),
                            ),
                        },
                    )
                })
                .collect(),
            merger: Merger {
                post_norm: Normalization {
                    d,
                    epsilon: epsilon as _,
                    items: NormType::LayerNorm {
                        dt_scale: dt_norm,
                        scale: get("v.post_ln.weight"),
                        dt_bias: dt_norm,
                        bias: get("v.post_ln.bias"),
                    },
                },
                mlp: Mlp {
                    up: Linear::new(
                        dt,
                        [d * 4, d * 4],
                        get("mm.0.weight"),
                        Some((dt_norm, get("mm.0.bias"))),
                    ),
                    act: Activation::GeLU,
                    down: Linear::new(
                        dt,
                        [d_proj, d * 4],
                        get("mm.2.weight"),
                        Some((dt_norm, get("mm.2.bias"))),
                    ),
                },
            },
        }
    }
}
