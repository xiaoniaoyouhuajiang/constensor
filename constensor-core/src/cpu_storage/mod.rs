use petgraph::algo::toposort;
use petgraph::graphmap::DiGraphMap;
use std::borrow::Cow;
use std::cell::RefCell;
use std::rc::Rc;

use pool::{BufferPool, PooledBuffer};
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

impl BackendDevice for CpuDevice {
    type Storage<X: DType> = CpuStorage<X>;

    fn compile_and_run_graph<S: Shape, T: DType>(
        &self,
        graph: &[Op<T>],
    ) -> Result<Self::Storage<T>> {
        {
            // Create a shared buffer pool
            let pool = Rc::new(RefCell::new(BufferPool::<T>::new()));

            // Build a dependency graph of tensor indices
            let mut dep_graph = DiGraphMap::<usize, ()>::new();
            for idx in 0..graph.len() {
                dep_graph.add_node(idx);
            }

            for (idx, node) in graph.iter().enumerate() {
                match node {
                    Op::BinaryOp { l_id, r_id, .. } => {
                        let l_idx = <&GraphTensorId as Into<usize>>::into(l_id);
                        let r_idx = <&GraphTensorId as Into<usize>>::into(r_id);
                        dep_graph.add_edge(l_idx, idx, ());
                        dep_graph.add_edge(r_idx, idx, ());
                    }
                    Op::UnaryOp { v_id, .. } => {
                        let v_idx = <&GraphTensorId as Into<usize>>::into(v_id);
                        dep_graph.add_edge(v_idx, idx, ());
                    }
                    Op::FusedMulAdd { a_id, b_id, c_id } => {
                        let a_idx = <&GraphTensorId as Into<usize>>::into(a_id);
                        let b_idx = <&GraphTensorId as Into<usize>>::into(b_id);
                        let c_idx = <&GraphTensorId as Into<usize>>::into(c_id);
                        dep_graph.add_edge(a_idx, idx, ());
                        dep_graph.add_edge(b_idx, idx, ());
                        dep_graph.add_edge(c_idx, idx, ());
                    }
                    _ => {}
                }
            }

            // Compute topological order
            let order = toposort(&dep_graph, None).expect("Cycle detected in graph!");

            // Prepare storage for intermediate results
            let mut results: Vec<Option<PooledBuffer<T>>> = Vec::with_capacity(graph.len());
            results.resize_with(graph.len(), || None);

            // Evaluate nodes in topological order
            for idx in order {
                let op = &graph[idx];
                let computed = match op {
                    Op::BinaryOp {
                        l_id,
                        r_id,
                        operator,
                    } => {
                        let l_idx = <&GraphTensorId as Into<usize>>::into(l_id);
                        let r_idx = <&GraphTensorId as Into<usize>>::into(r_id);
                        let l_buf = results[l_idx].take().unwrap();
                        let r_buf = results[r_idx].take().unwrap();
                        let mut out_buf = PooledBuffer::new(
                            pool.borrow_mut().get_buffer(S::element_count()),
                            pool.clone(),
                        );
                        out_buf.extend_from_slice(&l_buf);
                        T::binary_simd_op(&mut out_buf, r_buf.into_inner(), *operator);
                        out_buf
                    }
                    Op::Fill { v } => {
                        let mut buf = pool.borrow_mut().get_buffer(S::element_count());
                        buf.extend(std::iter::repeat_n(*v, S::element_count()));
                        PooledBuffer::new(buf, pool.clone())
                    }
                    Op::Arange { start, step, stop } => {
                        let mut buf = pool.borrow_mut().get_buffer(S::element_count());
                        let mut x = start.to_f64();
                        while x < stop.to_f64() {
                            buf.push(T::from_f64(x));
                            x += step.to_f64();
                        }
                        PooledBuffer::new(buf, pool.clone())
                    }
                    Op::UnaryOp { v_id, operator } => {
                        let v_idx = <&GraphTensorId as Into<usize>>::into(v_id);
                        let mut buf = results[v_idx].take().unwrap();
                        let op_fn = operator.to_closure();
                        buf.par_iter_mut().for_each(|x| *x = op_fn(*x));
                        buf
                    }
                    Op::FusedMulAdd { a_id, b_id, c_id } => {
                        let a_idx = <&GraphTensorId as Into<usize>>::into(a_id);
                        let b_idx = <&GraphTensorId as Into<usize>>::into(b_id);
                        let c_idx = <&GraphTensorId as Into<usize>>::into(c_id);
                        let mut buf = results[a_idx].take().unwrap();
                        let b_buf = results[b_idx].take().unwrap();
                        let c_buf = results[c_idx].take().unwrap();
                        T::fma_op(&mut *buf, b_buf.into_inner(), c_buf.into_inner());
                        buf
                    }
                    Op::NoOp => unreachable!("NoOp should not be evaluated."),
                };
                results[idx] = Some(computed);
            }

            // Extract final result
            let final_idx = graph.len() - 1;
            let output = results[final_idx].take().unwrap().into_inner();
            Ok(CpuStorage(output))
        }
    }
}
