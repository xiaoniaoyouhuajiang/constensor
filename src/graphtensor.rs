use std::{
    marker::PhantomData,
    ops::Add,
    sync::{Arc, RwLock, RwLockReadGuard},
};

use crate::{
    device::Dev,
    graph::{Graph, GraphTensorId, Op},
    DType, Shape,
};

#[derive(Clone)]
pub struct GraphTensor<S: Shape, T: DType, D: Dev> {
    id: GraphTensorId,
    graph: Arc<RwLock<Graph<T>>>,
    _ghost: PhantomData<(S, T, D)>,
}

impl<S: Shape, T: DType, D: Dev> GraphTensor<S, T, D> {
    pub fn fill(mut graph: Graph<T>, v: T) -> Self {
        let id = graph.next_id();
        graph.add_op(Op::Fill { v, id });
        Self {
            id,
            graph: Arc::new(RwLock::new(graph)),
            _ghost: PhantomData,
        }
    }
    pub fn zeros(graph: Graph<T>) -> Self {
        Self::fill(graph, T::ZERO)
    }
    pub fn ones(graph: Graph<T>) -> Self {
        Self::fill(graph, T::ONE)
    }
    pub fn graph(&self) -> RwLockReadGuard<Graph<T>> {
        self.graph.read().unwrap()
    }
    pub fn id(&self) -> GraphTensorId {
        self.id
    }
}

impl<S: Shape, T: DType, D: Dev> Add for GraphTensor<S, T, D> {
    type Output = GraphTensor<S, T, D>;
    fn add(self, rhs: Self) -> Self::Output {
        self.graph.write().unwrap().add_op(Op::Add {
            l_id: self.id(),
            r_id: rhs.id(),
        });
        Self {
            id: self.graph.write().unwrap().next_id(),
            graph: self.graph.clone(),
            _ghost: PhantomData,
        }
    }
}
