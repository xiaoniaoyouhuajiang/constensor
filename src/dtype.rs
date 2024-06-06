use std::fmt::Debug;

use cudarc::driver::DeviceRepr;

/// Marker trait for tensor datatypes.
pub trait DType: Debug + DeviceRepr + Clone + Copy {
    const ZERO: Self;
    const ONE: Self;
    const NAME: &'static str;
}

macro_rules! dtype {
    ($rt:ident, $zero:expr, $one:expr, $repr:expr) => {
        impl DType for $rt {
            const ZERO: $rt = $zero;
            const ONE: $rt = $one;
            const NAME: &'static str = $repr;
        }
    };
}

dtype!(bool, false, true, "bool");
dtype!(u8, 0u8, 1u8, "u8");
dtype!(u32, 0u32, 1u32, "u32");
dtype!(i64, 0i64, 1i64, "i64");
dtype!(f32, 0f32, 1f32, "f32");
dtype!(f64, 0f64, 1f64, "f64");
