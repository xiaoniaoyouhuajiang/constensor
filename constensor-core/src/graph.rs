use std::{
    cell::Cell,
    env,
    fmt::Display,
    fs,
    ops::Neg,
    path::Path,
    process::Command,
    rc::Rc,
    sync::{Arc, RwLock, RwLockReadGuard},
};

use crate::{DType, Result};

use petgraph::dot::{Config, Dot};
use petgraph::Graph as PetGraph;

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

    /// Export this computational graph as a petgraph::Graph where nodes are operation labels.
    pub fn to_petgraph(&self) -> PetGraph<String, ()> {
        let ops = self.data.read().unwrap();
        let mut g = PetGraph::<String, ()>::new();
        let mut nodes = Vec::with_capacity(ops.len());
        // Add nodes with labels
        for op in ops.iter() {
            let label = match op {
                Op::Fill { v } => format!("Fill({:?})", v),
                Op::Arange { start, step } => format!("Arange(start={:?}, step={:?})", start, step),
                Op::BinaryOp { operator, .. } => format!("BinOp({})", operator.as_c_op()),
                Op::UnaryOp { operator, .. } => format!("UnOp({:?})", operator),
                Op::FusedMulAdd { .. } => "FMA".to_string(),
                Op::NoOp => "NoOp".to_string(),
            };
            nodes.push(g.add_node(label));
        }
        // Add edges to represent data dependencies
        for (i, op) in ops.iter().enumerate() {
            let dst = nodes[i];
            match op {
                Op::BinaryOp { l_id, r_id, .. } => {
                    let src_l = nodes[usize::from(l_id)];
                    let src_r = nodes[usize::from(r_id)];
                    g.add_edge(src_l, dst, ());
                    g.add_edge(src_r, dst, ());
                }
                Op::UnaryOp { v_id, .. } => {
                    let src = nodes[usize::from(v_id)];
                    g.add_edge(src, dst, ());
                }
                Op::FusedMulAdd { a_id, b_id, c_id } => {
                    let src_a = nodes[usize::from(a_id)];
                    let src_b = nodes[usize::from(b_id)];
                    let src_c = nodes[usize::from(c_id)];
                    g.add_edge(src_a, dst, ());
                    g.add_edge(src_b, dst, ());
                    g.add_edge(src_c, dst, ());
                }
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
            .args(&[
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
                                Op::Arange { start: _, step: _ } => vec![],
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

        *self.data.write().unwrap() = new_ops;
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

    pub fn to_closure<T: DType + Neg<Output = T>>(&self) -> impl Fn(T) -> T {
        match self {
            Self::Neg => |x: T| -x,
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
