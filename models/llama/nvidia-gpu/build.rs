fn main() {
    use build_script_cfg::Cfg;
    use search_cuda_tools::find_cuda_root;

    let cfg = Cfg::new("hw_detected");
    if find_cuda_root().is_some() {
        cfg.define();
    }
}
