use std::borrow::Cow;

use crate::{
    storage::{BackendDevice, BackendStorage},
    DType, Result,
};

pub struct CpuDevice;

#[derive(Clone)]
pub struct CpuStorage<T: DType>(pub(crate) Vec<T>);

impl<T: DType> BackendStorage<T> for CpuStorage<T> {
    fn to_cpu_storage(&self) -> Result<std::borrow::Cow<CpuStorage<T>>> {
        // Note: copying all data here.
        Ok(Cow::Owned(self.clone()))
    }
}

impl<T: DType> BackendDevice<T> for CpuDevice {
    type Storage = CpuStorage<T>;

    fn const_impl<S: crate::Shape>(&self, v: T) -> Result<Self::Storage> {
        Ok(CpuStorage(vec![v; S::element_count()]))
    }
}
