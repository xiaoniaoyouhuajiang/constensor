use crate::{device::Device, storage::Storage, DType, Result, Shape, R1, R2, R3};
use std::{
    borrow::Cow,
    marker::PhantomData,
    ops::Deref,
    sync::atomic::{AtomicUsize, Ordering},
};

static TENSOR_ID: AtomicUsize = AtomicUsize::new(0);

pub struct TensorId(usize);

impl TensorId {
    fn next() -> Self {
        Self(TENSOR_ID.fetch_add(1, Ordering::Relaxed))
    }
}

impl Deref for TensorId {
    type Target = usize;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

pub struct Tensor<S: Shape, T: DType> {
    id: TensorId,
    storage: Storage<T>,
    _ghost: PhantomData<(S, T)>,
}

impl<S: Shape, T: DType> Tensor<S, T> {
    pub fn zeros(device: &Device) -> Result<Self> {
        Ok(Self {
            id: TensorId::next(),
            storage: device.const_impl::<T, S>(T::ZERO)?,
            _ghost: PhantomData,
        })
    }
    pub fn full(v: T, device: &Device) -> Result<Self> {
        Ok(Self {
            id: TensorId::next(),
            storage: device.const_impl::<T, S>(v)?,
            _ghost: PhantomData,
        })
    }
}

impl<T: DType, const A: usize> Tensor<R1<A>, T> {
    /// Get data for a tensor.
    pub fn data(&self) -> Result<Cow<Vec<T>>> {
        let data = self.storage.to_cpu_storage()?;
        Ok(Cow::Owned(data.into_owned().0))
    }
}

impl<T: DType, const A: usize, const B: usize> Tensor<R2<A, B>, T> {
    /// Get data for a tensor.
    pub fn data(&self) -> Result<Cow<Vec<Vec<T>>>> {
        let data = self.storage.to_cpu_storage()?;
        let mut rows = Vec::new();
        for i in 0..A {
            let row = (0..B).map(|j| data.as_ref().0[i * A + j]).collect();
            rows.push(row)
        }
        Ok(Cow::Owned(rows))
    }
}

impl<T: DType, const A: usize, const B: usize, const C: usize> Tensor<R3<A, B, C>, T> {
    /// Get data for a tensor.
    pub fn data(&self) -> Result<Cow<Vec<Vec<Vec<T>>>>> {
        let data = self.storage.to_cpu_storage()?;
        let mut top_rows = Vec::new();
        for i in 0..A {
            let mut rows = Vec::new();
            for j in 0..B {
                let row = (0..C).map(|k| data.as_ref().0[i * A + j * B + k]).collect();
                rows.push(row)
            }
            top_rows.push(rows);
        }
        Ok(Cow::Owned(top_rows))
    }
}

/*macro_rules! binary_op {
    ($trait:ident, $fn:ident) => {
        impl<S: Shape, D: DType> $trait for Tensor<S, D> {
            type Output = Result<Tensor<S, D>>;
            fn $fn(self, rhs: Self) -> Self::Output {
                Ok(Self::from_tensor(self.inner.$fn(&rhs.inner)?))
            }
        }
    };
}

binary_op!(Add, add);
binary_op!(Mul, mul);
binary_op!(Sub, sub);
binary_op!(Div, div);
*/
