use crate::{
    cpu_storage::CpuDevice,
    cuda_backend::CudaDevice,
    storage::{BackendDevice, Storage},
    DType, Result, Shape,
};

pub enum Device {
    Cuda(CudaDevice),
    Cpu,
}

impl Device {
    pub fn new_cuda(ordinal: usize) -> Result<Self> {
        Ok(Self::Cuda(CudaDevice::new(ordinal)?))
    }
    pub(crate) fn const_impl<T: DType, S: Shape>(&self, v: T) -> Result<Storage<T>> {
        match self {
            Self::Cuda(cuda) => Ok(Storage::Cuda(cuda.const_impl::<S>(v)?)),
            Self::Cpu => Ok(Storage::Cpu(CpuDevice.const_impl::<S>(v)?)),
        }
    }
}
