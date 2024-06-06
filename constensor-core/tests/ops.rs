use constensor_core::{Cpu, Graph, GraphTensor, Op, R2};

#[test]
fn fill() {
    let graph = Graph::empty();
    let _ = GraphTensor::<R2<3, 4>, f32, Cpu>::fill(graph.clone(), 0.0);
    assert_eq!(
        *graph.get_ops(),
        vec![Op::Fill {
            v: 0.0,
            id: 0.into()
        }]
    );
}

#[test]
fn add() {
    let graph = Graph::empty();
    let a = GraphTensor::<R2<3, 4>, f32, Cpu>::fill(graph.clone(), 1.0);
    let b = GraphTensor::<R2<3, 4>, f32, Cpu>::fill(graph.clone(), 2.0);
    let _ = a + b;
    assert_eq!(
        *graph.get_ops(),
        vec![
            Op::Fill {
                v: 1.0,
                id: 0.into()
            },
            Op::Fill {
                v: 2.0,
                id: 1.into()
            },
            Op::Add {
                l_id: 0.into(),
                r_id: 1.into()
            }
        ]
    );
}
