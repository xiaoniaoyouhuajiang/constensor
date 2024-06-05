use const_tensor::{Device, Tensor, R2};

#[test]
fn add() {
    let a: Tensor<R2<3, 4>, f32> = Tensor::full(1.0, &Device::Cpu).unwrap();
    let b: Tensor<R2<3, 4>, f32> = Tensor::full(2.0, &Device::Cpu).unwrap();
    let c = (a + b).unwrap();
    let data = c.data();
    assert_eq!(data, vec![vec![3.0; 4]; 3])
}

#[test]
fn mul() {
    let a: Tensor<R2<3, 4>, f32> = Tensor::full(1.0, &Device::Cpu).unwrap();
    let b: Tensor<R2<3, 4>, f32> = Tensor::full(2.0, &Device::Cpu).unwrap();
    let c = (a * b).unwrap();
    let data = c.data();
    assert_eq!(data, vec![vec![2.0; 4]; 3])
}

#[test]
fn sub() {
    let a: Tensor<R2<3, 4>, f32> = Tensor::full(1.0, &Device::Cpu).unwrap();
    let b: Tensor<R2<3, 4>, f32> = Tensor::full(2.0, &Device::Cpu).unwrap();
    let c = (a - b).unwrap();
    let data = c.data();
    assert_eq!(data, vec![vec![-1.0; 4]; 3])
}

#[test]
fn div() {
    let a: Tensor<R2<3, 4>, f32> = Tensor::full(1.0, &Device::Cpu).unwrap();
    let b: Tensor<R2<3, 4>, f32> = Tensor::full(2.0, &Device::Cpu).unwrap();
    let c = (a / b).unwrap();
    let data = c.data();
    assert_eq!(data, vec![vec![0.5; 4]; 3])
}
