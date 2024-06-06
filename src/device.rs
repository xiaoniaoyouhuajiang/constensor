use crate::{
    cpu_storage::CpuDevice,
    cuda_backend::CudaDevice,
    storage::{BackendDevice, Storage},
    DType, Result, Shape,
};

/// Marker trait for devices
pub trait Dev: Clone {
    fn resolve() -> Result<Device>;
}

#[derive(Clone)]
pub struct Cpu;

impl Dev for Cpu {
    fn resolve() -> Result<Device> {
        Ok(Device::Cpu)
    }
}

#[derive(Clone)]
pub struct Cuda<const ORD: usize>;

macro_rules! cuda_device {
    ($ord:expr) => {
        impl Dev for Cuda<$ord> {
            fn resolve() -> Result<Device> {
                Ok(Device::Cuda(CudaDevice::new($ord)?))
            }
        }
    };
}

// NOTE: Support up to 10 ordinals
cuda_device!(0);
cuda_device!(1);
cuda_device!(2);
cuda_device!(3);
cuda_device!(4);
cuda_device!(5);
cuda_device!(6);
cuda_device!(7);
cuda_device!(8);
cuda_device!(9);

/// A concrete device.
#[derive(Clone)]
pub enum Device {
    Cuda(CudaDevice),
    Cpu,
}

impl Device {
    pub(crate) fn const_impl<T: DType, S: Shape>(&self, v: T) -> Result<Storage<T>> {
        match self {
            Self::Cuda(cuda) => Ok(Storage::Cuda(cuda.const_impl::<S>(v)?)),
            Self::Cpu => Ok(Storage::Cpu(CpuDevice.const_impl::<S>(v)?)),
        }
    }
}
