use crate::{DType, Device, Result, Shape, R1, R2, R3};
use std::{marker::PhantomData, ops::Add};

pub struct Tensor<S: Shape, D: DType> {
    inner: candle_core::Tensor,
    _ghost: PhantomData<(S, D)>,
}

impl<S: Shape, D: DType> Tensor<S, D> {
    pub(crate) fn from_tensor(inner: candle_core::Tensor) -> Self {
        Self {
            inner,
            _ghost: PhantomData,
        }
    }

    pub fn zeros(device: &Device) -> Result<Self> {
        Ok(Self {
            inner: candle_core::Tensor::zeros(S::shape(), D::DTYPE, device)?,
            _ghost: PhantomData,
        })
    }

    pub fn ones(device: &Device) -> Result<Self> {
        Ok(Self {
            inner: candle_core::Tensor::ones(S::shape(), D::DTYPE, device)?,
            _ghost: PhantomData,
        })
    }

    pub fn full(data: D, device: &Device) -> Result<Self> {
        Ok(Self {
            inner: candle_core::Tensor::full(data, S::shape(), device)?,
            _ghost: PhantomData,
        })
    }

    pub fn shape(&self) -> Vec<usize> {
        S::shape()
    }
}

impl<S: Shape, D: DType> Add for Tensor<S, D> {
    type Output = Result<Tensor<S, D>>;
    fn add(self, rhs: Self) -> Self::Output {
        Ok(Self::from_tensor(self.inner.add(&rhs.inner)?))
    }
}

macro_rules! to_data {
    (($($C:ident),*), ($($N:tt),*), $rank:ident, $out:ty, $fn:ident) => {
        impl<D: DType, $($C $N: usize, )*> Tensor<$rank<$({ $N }, )*>, D> {
            /// Get data for a tensor.
            pub fn data(&self) -> $out {
                self.inner.$fn().unwrap()
            }
        }
    };
}

to_data!((const), (A), R1, Vec<D>, to_vec1);
to_data!((const, const), (A, B), R2, Vec<Vec<D>>, to_vec2);
to_data!((const, const, const), (A, B, C), R3, Vec<Vec<Vec<D>>>, to_vec3);
