use std::f32::consts::PI;

use const_tensor::{Device, Tensor, R1, R2, R3};

#[test]
fn zeros() {
    let a: Tensor<R2<3, 4>, f32> = Tensor::zeros(&Device::new_cuda(0).unwrap()).unwrap();
    let data = a.data().unwrap();
    assert_eq!(*data.as_ref(), vec![vec![0.0; 4]; 3]);
}

#[test]
fn full() {
    let a: Tensor<R2<3, 4>, f32> = Tensor::full(PI, &Device::new_cuda(0).unwrap()).unwrap();
    let data = a.data().unwrap();
    assert_eq!(*data.as_ref(), vec![vec![PI; 4]; 3]);
}

#[test]
fn dim1() {
    let a: Tensor<R1<3>, f32> = Tensor::full(PI, &Device::new_cuda(0).unwrap()).unwrap();
    let data = a.data().unwrap();
    assert_eq!(*data.as_ref(), vec![PI; 3]);
}

#[test]
fn dim2() {
    let a: Tensor<R2<3, 4>, f32> = Tensor::full(PI, &Device::new_cuda(0).unwrap()).unwrap();
    let data = a.data().unwrap();
    assert_eq!(*data.as_ref(), vec![vec![PI; 4]; 3]);
}

#[test]
fn dim3() {
    let a: Tensor<R3<3, 4, 5>, f32> = Tensor::full(PI, &Device::new_cuda(0).unwrap()).unwrap();
    let data = a.data().unwrap();
    assert_eq!(*data.as_ref(), vec![vec![vec![PI; 5]; 4]; 3]);
}
