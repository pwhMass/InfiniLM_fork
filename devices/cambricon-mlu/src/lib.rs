#![cfg(detected_neuware)]

pub use operators::cndrv;
pub use tensor::Tensor;

pub fn synchronize() {
    cndrv::init();
    for i in 0..cndrv::Device::count() {
        cndrv::Device::new(i as _)
            .acquire_shared()
            .apply(|ctx| ctx.synchronize());
    }
}
