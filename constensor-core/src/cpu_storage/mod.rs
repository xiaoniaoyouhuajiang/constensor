use std::borrow::Cow;
use std::cell::RefCell;
use std::rc::Rc;

use pool::{BufferPool, PooledBuffer, SharedPool};
use rayon::iter::{IntoParallelRefMutIterator, ParallelIterator};

use crate::{
    graph::GraphTensorId,
    storage::{BackendDevice, BackendStorage},
    DType, Op, Result, Shape,
};

mod pool;

pub struct CpuDevice;

#[derive(Clone)]
pub struct CpuStorage<T: DType>(pub(crate) Vec<T>);

impl<T: DType> BackendStorage<T> for CpuStorage<T> {
    fn to_cpu_storage(&self) -> Result<Cow<CpuStorage<T>>> {
        // Note: copying all data here.
        Ok(Cow::Owned(self.clone()))
    }
}

fn evaluate_node<T: DType, S: Shape>(
    op: &Op<T>,
    graph: &[Op<T>],
    pool: SharedPool<T>,
) -> Result<PooledBuffer<T>> {
    match op {
        Op::BinaryOp {
            l_id,
            r_id,
            operator,
        } => {
            let l_idx = <&GraphTensorId as Into<usize>>::into(l_id);
            let r_idx = <&GraphTensorId as Into<usize>>::into(r_id);
            let l = evaluate_node::<T, S>(&graph[l_idx], graph, pool.clone())?;
            let r = evaluate_node::<T, S>(&graph[r_idx], graph, pool.clone())?;
            let buf = pool.borrow_mut().get_buffer(S::element_count());
            let mut out = PooledBuffer::new(buf, pool.clone());
            out.extend_from_slice(&l);
            T::binary_simd_op(&mut out, r.into_inner(), *operator);
            Ok(out)
        }
        Op::Fill { v } => {
            let mut buf = pool.borrow_mut().get_buffer(S::element_count());
            buf.extend(std::iter::repeat(*v).take(S::element_count()));
            Ok(PooledBuffer::new(buf, pool.clone()))
        }
        Op::Arange { start, step, stop } => {
            let mut buf = pool.borrow_mut().get_buffer(S::element_count());
            let mut x = start.to_f64();
            while x < stop.to_f64() {
                buf.push(T::from_f64(x));
                x += step.to_f64();
            }
            Ok(PooledBuffer::new(buf, pool.clone()))
        }
        Op::UnaryOp { v_id, operator } => {
            let v_idx = <&GraphTensorId as Into<usize>>::into(v_id);
            let mut buf = evaluate_node::<T, S>(&graph[v_idx], graph, pool.clone())?;
            let op_fn = operator.to_closure();
            buf.par_iter_mut().for_each(|x| *x = op_fn(*x));
            Ok(buf)
        }
        Op::FusedMulAdd { a_id, b_id, c_id } => {
            let a_idx = <&GraphTensorId as Into<usize>>::into(a_id);
            let b_idx = <&GraphTensorId as Into<usize>>::into(b_id);
            let c_idx = <&GraphTensorId as Into<usize>>::into(c_id);
            let mut a = evaluate_node::<T, S>(&graph[a_idx], graph, pool.clone())?;
            let b = evaluate_node::<T, S>(&graph[b_idx], graph, pool.clone())?;
            let c = evaluate_node::<T, S>(&graph[c_idx], graph, pool.clone())?;
            T::fma_op(&mut *a, b.into_inner(), c.into_inner());
            Ok(a)
        }
        Op::NoOp => unreachable!("no-op ops should never be reached."),
    }
}

impl BackendDevice for CpuDevice {
    type Storage<X: DType> = CpuStorage<X>;

    fn compile_and_run_graph<S: Shape, T: DType>(
        &self,
        graph: &[Op<T>],
    ) -> Result<Self::Storage<T>> {
        {
            let pool = Rc::new(RefCell::new(BufferPool::<T>::new()));
            let result = evaluate_node::<T, S>(graph.last().unwrap(), graph, pool.clone())?;
            let vec = result.into_inner();
            Ok(CpuStorage(vec))
        }
    }
}
