#[allow(unused, non_upper_case_globals, non_camel_case_types, non_snake_case)]
pub mod bindings {
    include!(concat!(env!("OUT_DIR"), "/bindings.rs"));
}

#[test]
fn test_bindings() {
    use bindings::{create_model, DataType::*, DeviceType::DEVICE_CPU, LlamaMeta, LlamaWeights};
    use std::ptr::null;

    let llama_meta = LlamaMeta {
        dt_norm: DATA_TYPE_F32,
        dt_mat: DATA_TYPE_F16,
        nlayer: 22,
        nh: 32,
        nkvh: 4,
        dh: 64,
        di: 5632,
        dctx: 2048,
        dvoc: 32000,
        epsilon: 1e-5,
        theta: 1e4,
    };
    let llama_weights = LlamaWeights {
        nlayer: 22,
        input_embd: null(),
        ouput_norm: null(),
        output_embd: null(),
        attn_norm: null(),
        attn_qkv: null(),
        attn_o: null(),
        ffn_norm: null(),
        ffn_gate_up: null(),
        ffn_down: null(),
    };
    let model = unsafe { create_model(&llama_meta, &llama_weights, DEVICE_CPU, 1, null()) };
    assert!(model.is_null());
}
