use build_script_cfg::Cfg;
use search_corex_tools::find_corex;
use search_cuda_tools::{find_cuda_root, find_nccl_root};
use search_maca_tools::find_maca_root;
use std::{env, path::PathBuf};

fn main() {
    let nccl = Cfg::new("nccl");
    let src_dir =
        PathBuf::from(&env::var_os("CARGO_MANIFEST_DIR").unwrap()).join("src/op/random_sample");

    if let Some(maca_root) = find_maca_root() {
        nccl.define();

        cuda_cc::Builder::new("random_sample", maca_root, "htgpu_llvm/bin/htcc")
            .source(src_dir.join("sample.maca"))
            .header(src_dir.join("sample.h"))
            .cc_flags(["-x", "hpcc", "-fPIC"])
            .symbol("__MACA_ARCH__")
            .compile()
    } else if let Some(corex_root) = find_corex() {
        cuda_cc::Builder::new("random_sample", corex_root, "bin/clang++")
            .source(src_dir.join("sample.cu"))
            .header(src_dir.join("sample.h"))
            .cc_flags(["-x", "ivcore", "-fPIC"])
            .compile()
    } else if let Some(cuda_root) = find_cuda_root() {
        if find_nccl_root().is_some() {
            nccl.define()
        }

        cuda_cc::Builder::new("random_sample", cuda_root, "bin/nvcc")
            .source(src_dir.join("sample.cu"))
            .header(src_dir.join("sample.h"))
            .cc_flags(["-Xcompiler", "-fPIC"])
            .compile()
    } else {
        panic!("cuda not found, check $CUDA_ROOT env var")
    }
}
