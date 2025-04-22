use constensor_core::{Cpu, Graph, GraphTensor, Tensor, R1, R2};

fn main() {
    let mut graph: Graph<f32> = Graph::empty();
    let arange = GraphTensor::<R1<12>, f32, Cpu>::arange(&mut graph, 0., 100.);
    dbg!(&arange.to_tensor().unwrap().data());
    let x = GraphTensor::<R2<3, 4>, f32, Cpu>::fill(&mut graph, 1.0);
    let y = GraphTensor::<R2<3, 4>, f32, Cpu>::fill(&mut graph, 2.0);
    let z = y.clone() + y * x;

    graph.optimize();

    graph.visualize("graph.png").unwrap();

    let tensor: Tensor<R2<3, 4>, f32, Cpu> = z.to_tensor().unwrap();

    assert_eq!(tensor.data().unwrap().to_vec(), vec![vec![4.0; 4]; 3],);
}
