use std::borrow::Cow;

#[cfg(feature = "cuda")]
use crate::cuda_backend::CudaStorage;
use crate::{cpu_storage::CpuStorage, DType, Result, Shape};

pub enum Storage<T: DType> {
    #[cfg(feature = "cuda")]
    Cuda(CudaStorage<T>),
    Cpu(CpuStorage<T>),
}

impl<T: DType> Storage<T> {
    pub(crate) fn to_cpu_storage(&self) -> Result<Cow<CpuStorage<T>>> {
        match self {
            Self::Cpu(cpu) => cpu.to_cpu_storage(),
            #[cfg(feature = "cuda")]
            Self::Cuda(cuda) => cuda.to_cpu_storage(),
        }
    }
}

pub trait BackendStorage<T: DType> {
    fn to_cpu_storage(&self) -> Result<Cow<CpuStorage<T>>>;
}

pub trait BackendDevice<T: DType> {
    type Storage: BackendStorage<T>;

    fn const_impl<S: Shape>(&self, v: T) -> Result<Self::Storage>;
}
