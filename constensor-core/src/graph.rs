use std::{
    cell::Cell,
    env,
    fmt::Display,
    fs,
    path::Path,
    process::Command,
    rc::Rc,
    sync::{Arc, RwLock, RwLockReadGuard},
};

use crate::{DType, Result};

use petgraph::Graph as PetGraph;
use petgraph::{
    dot::{Config, Dot},
    graph::NodeIndex,
};

#[derive(Clone)]
pub struct Graph<T: DType> {
    data: Arc<RwLock<Vec<Op<T>>>>,
    id: Arc<RwLock<usize>>,
}

impl<T: DType> Graph<T> {
    /// Create an empty Graph
    pub fn empty() -> Self {
        Self {
            data: Arc::new(RwLock::new(Vec::new())),
            id: Arc::new(RwLock::new(0)),
        }
    }

    /// Read-only access to the list of operations
    pub fn get_ops(&self) -> RwLockReadGuard<Vec<Op<T>>> {
        self.data.read().unwrap()
    }

    /// Append an operation to the graph
    pub(crate) fn add_op(&self, op: Op<T>) {
        self.data.write().unwrap().push(op);
    }

    /// Generate the next unique tensor ID
    #[must_use]
    pub(crate) fn next_id(&mut self) -> GraphTensorId {
        let next = GraphTensorId::from(*self.id.read().unwrap());
        *self.id.write().unwrap() += 1;
        next
    }

    pub fn to_petgraph(&self) -> PetGraph<String, ()> {
        let ops = self.data.read().unwrap();
        let mut g = PetGraph::<String, ()>::new();
        // map from op‐index → Some(node) if we created a node, or None if it was a NoOp
        let mut idx_map: Vec<Option<NodeIndex>> = Vec::with_capacity(ops.len());

        // 1) Add only non‐NoOp nodes
        for op in ops.iter() {
            match op {
                Op::NoOp => {
                    idx_map.push(None);
                }
                _ => {
                    let label = match op {
                        Op::Fill { v } => format!("Fill({:?})", v),
                        Op::Arange { start, step, stop } => {
                            format!(
                                "Arange(start={:?}, step={:?}, stop={:?})",
                                start, step, stop
                            )
                        }
                        Op::BinaryOp { operator, .. } => format!("BinOp({})", operator.as_c_op()),
                        Op::UnaryOp { operator, .. } => format!("UnOp({:?})", operator),
                        Op::FusedMulAdd { .. } => "FMA".to_string(),
                        // we already matched NoOp above
                        _ => unreachable!(),
                    };
                    let node = g.add_node(label);
                    idx_map.push(Some(node));
                }
            }
        }

        // 2) Walk ops again and only connect edges for those dst nodes that exist
        for (i, op) in ops.iter().enumerate() {
            // if this op was NoOp, skip entirely
            let dst = match idx_map[i] {
                Some(dst) => dst,
                None => continue,
            };
            match op {
                Op::BinaryOp { l_id, r_id, .. } => {
                    if let Some(src) = idx_map[usize::from(l_id)] {
                        g.add_edge(src, dst, ());
                    }
                    if let Some(src) = idx_map[usize::from(r_id)] {
                        g.add_edge(src, dst, ());
                    }
                }
                Op::UnaryOp { v_id, .. } => {
                    if let Some(src) = idx_map[usize::from(v_id)] {
                        g.add_edge(src, dst, ());
                    }
                }
                Op::FusedMulAdd { a_id, b_id, c_id } => {
                    for src_id in [a_id, b_id, c_id] {
                        if let Some(src) = idx_map[usize::from(src_id)] {
                            g.add_edge(src, dst, ());
                        }
                    }
                }
                // NoOp and Fill/Arange don’t create incoming edges
                _ => {}
            }
        }

        g
    }

    /// Produce a DOT format string of this graph.
    pub fn to_dot(&self) -> String {
        let g = self.to_petgraph();
        format!("{:?}", Dot::with_config(&g, &[Config::EdgeNoLabel]))
    }

    /// Visualize the graph by saving it to this file.
    ///
    /// Install graphvis:
    /// - brew install graphviz
    /// - apt install graphviz
    pub fn visualize<P: AsRef<Path>>(&self, filename: P) -> Result<()> {
        let path = filename.as_ref();
        let tmp_dir = env::temp_dir();
        let dot_path = tmp_dir.join("graph.dot");
        let png_path = path.to_path_buf();

        fs::write(&dot_path, self.to_dot())?;
        let status = Command::new("dot")
            .args([
                "-Tpng",
                &dot_path.display().to_string(),
                "-o",
                &png_path.display().to_string(),
            ])
            .status()?;
        if !status.success() {
            panic!("Graphviz failed");
        }

        Ok(())
    }

    /// Optimize by looking for mul-add pairs, convert to FMA
    fn optimize_fma(&mut self) {
        let ops = self.data.write().unwrap().clone();
        let mut new_ops = ops.clone();

        // This contains the indices of the first of the pair.
        for (x_id, x) in ops.iter().enumerate() {
            if let Op::BinaryOp {
                l_id: a_id,
                r_id: b_id,
                operator: BinaryOpType::Mul,
            } = x
            {
                // Check if next op uses this
                if let Op::BinaryOp {
                    l_id: l_y,
                    r_id: r_y,
                    operator: BinaryOpType::Add,
                } = &ops[x_id + 1]
                {
                    let y_id = x_id + 1;
                    if <&GraphTensorId as Into<usize>>::into(l_y) == x_id
                        || <&GraphTensorId as Into<usize>>::into(r_y) == x_id
                    {
                        // Want to see what is being added to the result of the mul
                        let rhs_add = if <&GraphTensorId as Into<usize>>::into(l_y) == x_id {
                            r_y
                        } else {
                            l_y
                        };
                        new_ops[y_id] = Op::FusedMulAdd {
                            a_id: GraphTensorId::from(a_id.0.get()),
                            b_id: GraphTensorId::from(b_id.0.get()),
                            c_id: GraphTensorId::from(rhs_add.0.get()),
                        };
                        new_ops[x_id] = Op::NoOp;

                        // Look for ops which actually use this one
                        for user in ops.iter() {
                            let ids = match user {
                                Op::Arange {
                                    start: _,
                                    step: _,
                                    stop: _,
                                } => vec![],
                                Op::BinaryOp {
                                    l_id,
                                    r_id,
                                    operator: _,
                                } => vec![l_id, r_id],
                                Op::Fill { v: _ } => vec![],
                                Op::UnaryOp { v_id, operator: _ } => vec![v_id],
                                Op::FusedMulAdd { a_id, b_id, c_id } => {
                                    vec![a_id, b_id, c_id]
                                }
                                Op::NoOp => vec![],
                            };
                            let used_ids = ids
                                .into_iter()
                                .filter(|id| <&GraphTensorId as Into<usize>>::into(id) != y_id)
                                .collect::<Vec<_>>();
                            if !used_ids.is_empty() {
                                for id in used_ids {
                                    // Tell the ops which use the result of the fma to source from there
                                    id.0.set(y_id);
                                }
                            }
                        }
                    }
                }
            }
        }

        // Remove any NoOp entries before storing back to the graph
        let filtered_ops = new_ops
            .into_iter()
            .filter(|op| !matches!(op, Op::NoOp))
            .collect::<Vec<_>>();
        *self.data.write().unwrap() = filtered_ops;
    }

    /// Optimize this graph.
    ///
    /// Apply the following optimizations
    /// - Fuse mul,add
    pub fn optimize(&mut self) {
        self.optimize_fma();
    }
}

#[derive(PartialEq, Debug, Clone, Copy)]
pub enum BinaryOpType {
    Add,
    Div,
    Sub,
    Mul,
}

impl BinaryOpType {
    pub fn as_c_op(&self) -> &'static str {
        match self {
            Self::Add => "+",
            Self::Div => "/",
            Self::Sub => "-",
            Self::Mul => "*",
        }
    }

    pub fn as_closure<T: DType>(&self) -> impl Fn(T, T) -> T {
        match self {
            Self::Add => |x, y| x + y,
            Self::Div => |x, y| x / y,
            Self::Sub => |x, y| x - y,
            Self::Mul => |x, y| x * y,
        }
    }
}

#[derive(PartialEq, Debug, Clone)]
pub enum UnaryOpType {
    Neg,
    Sqrt,
}

impl UnaryOpType {
    pub fn fill_in_c_op(&self, val: impl Display) -> String {
        match self {
            Self::Neg => format!("-{val}"),
            Self::Sqrt => format!("static_cast<T>( sqrt( static_cast<double>({val}) ) )"),
        }
    }

    pub fn to_closure<T: DType>(&self) -> impl Fn(T) -> T {
        match self {
            Self::Neg => T::maybe_neg,
            Self::Sqrt => |x: T| x.sqrt(),
        }
    }
}

#[derive(PartialEq, Debug, Clone)]
pub enum Op<T: DType> {
    Fill {
        v: T,
    },
    Arange {
        start: T,
        step: T,
        stop: T,
    },
    BinaryOp {
        l_id: GraphTensorId,
        r_id: GraphTensorId,
        operator: BinaryOpType,
    },
    UnaryOp {
        v_id: GraphTensorId,
        operator: UnaryOpType,
    },
    /// a * b + c
    FusedMulAdd {
        a_id: GraphTensorId,
        b_id: GraphTensorId,
        c_id: GraphTensorId,
    },
    NoOp,
}

#[derive(Clone, PartialEq, Debug)]
/// Graph tensor IDs can be cloned.
pub struct GraphTensorId(Rc<Cell<usize>>);

impl From<GraphTensorId> for usize {
    fn from(value: GraphTensorId) -> Self {
        value.0.get()
    }
}

impl From<&GraphTensorId> for usize {
    fn from(value: &GraphTensorId) -> Self {
        value.0.get()
    }
}

impl From<usize> for GraphTensorId {
    fn from(value: usize) -> Self {
        Self(Rc::new(Cell::new(value)))
    }
}
