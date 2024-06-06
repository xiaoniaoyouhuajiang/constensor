use constensor_core::{Cpu, Tensor, R1, R2, R3};

#[cfg(feature = "cuda")]
use constensor_core::Cuda;

macro_rules! test_device_dtype {
    ($dtype:ty, $dev:ty, $zero:expr, $one:expr, $full:expr, $dtype_mod:ident) => {
        mod $dtype_mod {
            use super::*;

            #[test]
            fn zeros() {
                let a = Tensor::<R2<3, 4>, $dtype, $dev>::zeros().unwrap();
                let data = a.data().unwrap();
                assert_eq!(*data.as_ref(), vec![vec![$zero; 4]; 3]);
            }

            #[test]
            fn ones() {
                let a = Tensor::<R2<3, 4>, $dtype, $dev>::ones().unwrap();
                let data = a.data().unwrap();
                assert_eq!(*data.as_ref(), vec![vec![$one; 4]; 3]);
            }

            #[test]
            fn full() {
                let a = Tensor::<R2<3, 4>, $dtype, $dev>::full($full).unwrap();
                let data = a.data().unwrap();
                assert_eq!(*data.as_ref(), vec![vec![$full; 4]; 3]);
            }

            #[test]
            fn dim1() {
                let a = Tensor::<R1<3>, $dtype, $dev>::full($full).unwrap();
                let data = a.data().unwrap();
                assert_eq!(*data.as_ref(), vec![$full; 3]);
            }

            #[test]
            fn dim2() {
                let a = Tensor::<R2<3, 4>, $dtype, $dev>::full($full).unwrap();
                let data = a.data().unwrap();
                assert_eq!(*data.as_ref(), vec![vec![$full; 4]; 3]);
            }

            #[test]
            fn dim3() {
                let a = Tensor::<R3<3, 4, 5>, $dtype, $dev>::full($full).unwrap();
                let data = a.data().unwrap();
                assert_eq!(*data.as_ref(), vec![vec![vec![$full; 5]; 4]; 3]);
            }
        }
    };
}

test_device_dtype!(f32, Cpu, 0.0, 1.0, std::f32::consts::PI, f32_test);
test_device_dtype!(f64, Cpu, 0.0, 1.0, std::f64::consts::PI, f64_test);
test_device_dtype!(bool, Cpu, false, true, false, bool_test);
test_device_dtype!(u8, Cpu, 0, 1, u8::MAX, u8_test);
test_device_dtype!(u32, Cpu, 0, 1, u32::MAX, u32_test);
test_device_dtype!(i64, Cpu, 0, 1, i64::MAX, i64_test);
#[cfg(feature = "half")]
use half::f16;
#[cfg(feature = "half")]
test_device_dtype!(
    f16,
    Cpu,
    f16::from_f32_const(0.0),
    f16::from_f32_const(1.0),
    f16::from_f32_const(0.5),
    f16_test
);
#[cfg(feature = "bfloat")]
use half::bf16;
#[cfg(feature = "bfloat")]
test_device_dtype!(
    bf16,
    Cpu,
    bf16::from_f32_const(0.0),
    bf16::from_f32_const(1.0),
    bf16::from_f32_const(0.5),
    bf16_test
);

#[cfg(feature = "cuda")]
test_device_dtype!(f32, Cuda<0>, 0.0, 1.0, std::f32::consts::PI, f32_test_cuda);
#[cfg(feature = "cuda")]
test_device_dtype!(f64, Cuda<0>, 0.0, 1.0, std::f64::consts::PI, f64_test_cuda);
#[cfg(feature = "cuda")]
test_device_dtype!(bool, Cuda<0>, false, true, false, bool_test_cuda);
#[cfg(feature = "cuda")]
test_device_dtype!(u8, Cuda<0>, 0, 1, u8::MAX, u8_test_cuda);
#[cfg(feature = "cuda")]
test_device_dtype!(u32, Cuda<0>, 0, 1, u32::MAX, u32_test_cuda);
#[cfg(feature = "cuda")]
test_device_dtype!(i64, Cuda<0>, 0, 1, i64::MAX, i64_test_cuda);
#[cfg(all(feature = "half", feature = "cuda"))]
test_device_dtype!(
    f16,
    Cuda<0>,
    f16::from_f32_const(0.0),
    f16::from_f32_const(1.0),
    f16::from_f32_const(0.5),
    f16_test_cuda
);
#[cfg(all(feature = "bfloat", feature = "cuda"))]
test_device_dtype!(
    bf16,
    Cuda<0>,
    bf16::from_f32_const(0.0),
    bf16::from_f32_const(1.0),
    bf16::from_f32_const(0.5),
    bf16_test_cuda
);
