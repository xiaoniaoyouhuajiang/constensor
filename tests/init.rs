use std::f32::consts::PI;

use candle_core::Device;
use const_tensor::{Tensor, R2};

#[test]
fn zeros() {
    let a: Tensor<R2<3, 4>, f32> = Tensor::zeros(&Device::Cpu).unwrap();
    let data = a.data();
    assert_eq!(data, vec![vec![0.0; 4]; 3])
}

#[test]
fn ones() {
    let a: Tensor<R2<3, 4>, f32> = Tensor::ones(&Device::Cpu).unwrap();
    let data = a.data();
    assert_eq!(data, vec![vec![1.0; 4]; 3])
}

#[test]
fn full() {
    let a: Tensor<R2<3, 4>, f32> = Tensor::full(PI, &Device::Cpu).unwrap();
    let data = a.data();
    assert_eq!(data, vec![vec![PI; 4]; 3])
}
