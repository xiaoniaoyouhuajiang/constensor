use std::f32::consts::PI;

use const_tensor::{Device, Tensor, R1, R2, R3, R4, R5, R6};

macro_rules! generate_dim_test {
    (($($N:expr),*), $name:ident, $rank:ident) => {
        #[test]
        fn $name() {
            let tensor: Tensor<$rank<$($N, )*>, f32> = Tensor::zeros(&Device::Cpu).unwrap();
            assert_eq!(tensor.shape(), vec![$($N, )*]);
        }
    };
}

generate_dim_test!((1), dim1, R1);
generate_dim_test!((1, 2), dim2, R2);
generate_dim_test!((1, 2, 3), dim3, R3);
generate_dim_test!((1, 2, 3, 4), dim4, R4);
generate_dim_test!((1, 2, 3, 4, 5), dim5, R5);
generate_dim_test!((1, 2, 3, 4, 5, 6), dim6, R6);

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

#[test]
fn add() {
    let a: Tensor<R2<3, 4>, f32> = Tensor::full(1.0, &Device::Cpu).unwrap();
    let b: Tensor<R2<3, 4>, f32> = Tensor::full(2.0, &Device::Cpu).unwrap();
    let c = (a + b).unwrap();
    let data = c.data();
    assert_eq!(data, vec![vec![3.0; 4]; 3])
}
