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
