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

/// A tensor representing an intermediary result of a graph. Performing operations
/// on this tensor will not cause any computations.
#[derive(Clone)]
pub struct GraphTensor<S: Shape, T: DType, D: Dev> {
    id: GraphTensorId,
    graph: Arc<RwLock<Graph<T>>>,
    _ghost: PhantomData<(S, T, D)>,
}

impl<S: Shape, T: DType, D: Dev> GraphTensor<S, T, D> {
    /// Create a tensor filled with some value.
    pub fn fill(mut graph: Graph<T>, v: T) -> Self {
        let id = graph.next_id();
        graph.add_op(Op::Fill { v, id });
        Self {
            id,
            graph: Arc::new(RwLock::new(graph)),
            _ghost: PhantomData,
        }
    }

    /// Create a tensor filled with zeros.
    pub fn zeros(graph: Graph<T>) -> Self {
        Self::fill(graph, T::ZERO)
    }

    /// Create a tensor filled with ones.
    pub fn ones(graph: Graph<T>) -> Self {
        Self::fill(graph, T::ONE)
    }

    /// Retrieve the graph for this `GraphTensor`.
    pub fn graph(&self) -> RwLockReadGuard<Graph<T>> {
        self.graph.read().unwrap()
    }

    /// Get the graph tensor ID.
    pub fn id(&self) -> GraphTensorId {
        self.id
    }
}

impl<S: Shape, T: DType, D: Dev> Add for GraphTensor<S, T, D> {
    type Output = GraphTensor<S, T, D>;
    /// Add an elementwise addition operation to the graph.
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
