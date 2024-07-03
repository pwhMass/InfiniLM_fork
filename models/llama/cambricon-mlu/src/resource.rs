use common_cn::cndrv::{Context, ContextResource, ContextSpore, DevMemSpore};
use std::{mem::ManuallyDrop, sync::Arc};

pub struct Cache {
    res: Arc<Context>,
    pub(super) mem: ManuallyDrop<DevMemSpore>,
}

impl Cache {
    #[inline]
    pub(super) fn new(res: &Arc<Context>, len: usize) -> Self {
        Self {
            res: res.clone(),
            mem: ManuallyDrop::new(res.apply(|ctx| ctx.malloc::<u8>(len).sporulate())),
        }
    }
}

impl Drop for Cache {
    #[inline]
    fn drop(&mut self) {
        let mem = unsafe { ManuallyDrop::take(&mut self.mem) };
        self.res.apply(|ctx| drop(mem.sprout(ctx)));
    }
}
