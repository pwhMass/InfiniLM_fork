use common_nv::cuda::{
    Context, ContextResource, ContextSpore, DevMemSpore, Device, Stream, StreamSpore,
};
use std::{mem::ManuallyDrop, sync::Arc};

pub(super) struct Resource {
    context: Context,
    compute: ManuallyDrop<StreamSpore>,
}

impl Resource {
    #[inline]
    pub fn new(device: &Device) -> Self {
        let context = device.retain_primary();
        let compute = context.apply(|ctx| ctx.stream().sporulate());
        Self {
            context,
            compute: ManuallyDrop::new(compute),
        }
    }

    #[inline]
    pub fn apply<T>(&self, f: impl FnOnce(&Stream) -> T) -> T {
        self.context.apply(|ctx| f(self.compute.sprout_ref(ctx)))
    }
}

impl Drop for Resource {
    #[inline]
    fn drop(&mut self) {
        let compute = unsafe { ManuallyDrop::take(&mut self.compute) };
        self.context.apply(|ctx| drop(compute.sprout(ctx)));
    }
}

pub struct Cache {
    res: Arc<Resource>,
    pub(super) mem: ManuallyDrop<DevMemSpore>,
}

impl Cache {
    #[inline]
    pub(super) fn new(res: &Arc<Resource>, len: usize) -> Self {
        Self {
            res: res.clone(),
            mem: ManuallyDrop::new(res.apply(|compute| compute.malloc::<u8>(len).sporulate())),
        }
    }
}

impl Drop for Cache {
    #[inline]
    fn drop(&mut self) {
        let mem = unsafe { ManuallyDrop::take(&mut self.mem) };
        self.res
            .apply(|stream| mem.sprout(stream.ctx()).drop_on(stream));
    }
}
