use std::fmt::Debug;

use candle_core::WithDType;

pub trait DType: WithDType + Debug {
    const ZERO: Self;
}

macro_rules! dtype {
    ($rt:ident, $zero:expr) => {
        impl DType for $rt {
            const ZERO: $rt = $zero;
        }
    };
}

dtype!(u8, 0u8);
dtype!(u32, 0u32);
dtype!(i64, 0i64);
dtype!(f32, 0f32);
dtype!(f64, 0f64);
