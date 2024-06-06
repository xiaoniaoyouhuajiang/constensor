use std::fmt::Debug;

use cudarc::driver::DeviceRepr;

pub trait DType: Debug + DeviceRepr + Clone + Copy {
    const ZERO: Self;
    const NAME: &'static str;
}

macro_rules! dtype {
    ($rt:ident, $zero:expr, $repr:expr) => {
        impl DType for $rt {
            const ZERO: $rt = $zero;
            const NAME: &'static str = $repr;
        }
    };
}

dtype!(u8, 0u8, "u8");
dtype!(u32, 0u32, "u32");
dtype!(i64, 0i64, "i64");
dtype!(f32, 0f32, "f32");
dtype!(f64, 0f64, "f64");
