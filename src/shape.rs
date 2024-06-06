pub trait Shape {
    fn shape() -> Vec<usize>;
    fn element_count() -> usize {
        Self::shape().iter().product()
    }
}

macro_rules! shape {
    (($($C:ident),*), ($($N:tt),*), $name:ident) => {
        pub struct $name<$($C $N: usize, )*>;

        impl<$($C $N: usize, )*> Shape for $name<$({ $N }, )*> {
            fn shape() -> Vec<usize> {
                vec![$($N, )*]
            }
        }
    };
}

shape!((const), (A), R1);
shape!((const, const), (A, B), R2);
shape!((const, const, const), (A, B, C), R3);
shape!((const, const, const, const), (A, B, C, D), R4);
shape!((const, const, const, const, const), (A, B, C, D, E), R5);
shape!((const, const, const, const, const, const), (A, B, C, D, E, F), R6);
