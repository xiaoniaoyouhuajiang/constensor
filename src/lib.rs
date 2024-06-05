mod dtype;
mod shape;
mod tensor;
pub use candle_core::{Device, Result};
pub use dtype::DType;
pub use shape::{Shape, R1, R2, R3, R4, R5, R6};
pub use tensor::Tensor;
