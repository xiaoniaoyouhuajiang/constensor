use crate::{
    device::{Cpu, Cuda, Dev},
    storage::Storage,
    DType, Result, Shape, R1, R2, R3,
};
use std::{borrow::Cow, marker::PhantomData, ops::Deref, sync::Arc};

#[derive(Clone)]
pub struct Tensor_<S: Shape, T: DType, D: Dev> {
    storage: Arc<Storage<T>>,
    _ghost: PhantomData<(S, T, D)>,
}

/// Tensors are n dimensional arrays. Only functions which allocate, copy
/// data, or do operations return `Result`s, so applying functions does not
/// return a result and instead builds up the graph.
#[derive(Clone)]
pub struct Tensor<S: Shape, T: DType, D: Dev>(Arc<Tensor_<S, T, D>>);

impl<S: Shape, T: DType, D: Dev> Deref for Tensor<S, T, D> {
    type Target = Tensor_<S, T, D>;

    fn deref(&self) -> &Self::Target {
        self.0.as_ref()
    }
}

fn from_storage<S: Shape, T: DType, D: Dev>(storage: Arc<Storage<T>>) -> Tensor<S, T, D> {
    Tensor(Arc::new(Tensor_ {
        storage,
        _ghost: PhantomData,
    }))
}

macro_rules! tensor_api {
    ($device:ty) => {
        impl<S: Shape, T: DType> Tensor<S, T, $device> {
            pub fn full(v: T) -> Result<Self> {
                let device = <$device>::resolve()?;
                Ok(from_storage(Arc::new(device.const_impl::<T, S>(v)?)))
            }
            pub fn zeros() -> Result<Self> {
                Self::full(T::ZERO)
            }
            pub fn ones() -> Result<Self> {
                Self::full(T::ONE)
            }
            pub fn zeros_like(&self) -> Result<Self> {
                Tensor::<S, T, $device>::zeros()
            }
            pub fn ones_like(&self) -> Result<Self> {
                Tensor::<S, T, $device>::ones()
            }
            pub fn full_like(&self, v: T) -> Result<Self> {
                Tensor::<S, T, $device>::full(v)
            }
        }

        impl<T: DType, const A: usize> Tensor<R1<A>, T, $device> {
            /// Get data for a vector.
            pub fn data(&self) -> Result<Cow<Vec<T>>> {
                let data = self.storage.to_cpu_storage()?;
                Ok(Cow::Owned(data.into_owned().0))
            }
        }

        impl<T: DType, const A: usize, const B: usize> Tensor<R2<A, B>, T, $device> {
            /// Get data for a matrix.
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

        impl<T: DType, const A: usize, const B: usize, const C: usize>
            Tensor<R3<A, B, C>, T, $device>
        {
            /// Get data for a 3 dimensional tensor.
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
    };
}

tensor_api!(Cpu);
tensor_api!(Cuda<0>);

/*macro_rules! binary_op {
    ($trait:ident, $fn:ident) => {
        impl<S: Shape, D: DType> $trait for Tensor<S, D> {
            type Output = Result<Tensor<S, D>>;
            fn $fn(self, rhs: Self) -> Self::Output {
                Ok(Self::fromTensor_(self.inner.$fn(&rhs.inner)?))
            }
        }
    };
}

binary_op!(Add, add);
binary_op!(Mul, mul);
binary_op!(Sub, sub);
binary_op!(Div, div);
*/
