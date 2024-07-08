pub fn list_turbo() {
    #[cfg(detected_cuda)]
    list_nv();
    #[cfg(detected_neuware)]
    list_cn();
}

#[cfg(detected_cuda)]
fn list_nv() {
    use llama_nv::cuda::{self, Device as Gpu};

    cuda::init();
    println!("NVidia CUDA environment detected.");
    for i in 0..Gpu::count() {
        let gpu = Gpu::new(i as _);
        println!(
            "GPU{i}: {} | cc={} | memory={}",
            gpu.name(),
            gpu.compute_capability(),
            gpu.total_memory(),
        );
    }
    println!();
}

#[cfg(detected_neuware)]
fn list_cn() {
    use llama_cn::cndrv::{self, Device as Mlu};

    cndrv::init();
    println!("Cambricon Neuware environment detected.");
    for i in 0..Mlu::count() {
        let mlu = Mlu::new(i as _);
        println!(
            "MLU{i}: {} | isa={} | memory={}",
            mlu.name(),
            mlu.isa(),
            mlu.total_memory(),
        );
    }
    println!();
}
