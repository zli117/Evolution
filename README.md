# Evolution [![Build Status](https://travis-ci.com/zli117/Evolution.svg?token=j4y2W3bQxnm7LkxGR6Um&branch=master)](https://travis-ci.com/zli117/Evolution)
Evolve a neural net from scratch


## Features

  - [X] Supports Keras
  - [X] Has a flexible way to define the constraints
  - [X] Visualization
  - [X] Interfaces has full type hint
  - [X] Train multiple sessions in parallel
  
## How it works
In order to evolve a neural network, we need to solve two problems:

1. Representing the network architecture in a way that can
   be easily changed
2. Finding an algorithm to decide on when should we mutate a
   population of networks.
  
The second problem is relatively easier to resolve. There are numerous
algorithms proposed to evolve neural networks. For example,
(Real et al., 2018) proposed an evolution algorithm using age as a
regularization, and (Lui et al., 2017) used an asynchronous algorithm to
evolve networks. In this project, I decided to go with
(Real et al., 2018) as the approach of using age as a regularization 
looks interesting. The implementation of the evolution algorithm is the
`aging_evolution` in
[evolution/evolve/evolve_strategy.py](evolution/evolve/evolve_strategy.py)

The first problem, on the other hand, is much harder. The challenges are
mainly due to the diverse nature of different neural network layers, the
building blocks of modern deep neural net. They could give
out tensor with different shapes on the same input, depending on the
parameters. For example, the output of a convolution layer with kernel
shape `3 x 3`, 10 output channels, stride of 1 and no padding on an
input tensor with shape `10 x 10 x 3` will be `8 x 8 x 10` while one
with kernel shape `5 x 5` and 20 output channels will give an output
with shape `6 x 6 x 20`. Since for most of the tasks, there will be a
fixed input shape (`32 x 32 x 3` for CIFAR10 for example) and fixed
output shape (`10` for CIFAR10), it's challenging to make sure each
layer has a correct set of parameters.

Further, there are special connections such as skip
connections for residual learning. Normally, neural nets layers 
follow a nice sequential order, meaning that each layer's output will
only go to the next layer, and each layer will only take input from the
previous layer. However, as (He et al., 2016) introduced residual 
learning to deep convolution network, the connection are not
as simple as sequential anymore. This requires the encoding scheme to be
flexible enough to accommodate skip connections. 

There are several papers describing ways to encode network 
architectures, but I found the hierarchical representation introduced in
(Lui et al., 2017) most flexible and interesting. In this project, I
based the idea on that paper, with more detailed specifications. For
instance, to ensure the encoded architecture is valid, I enforced
several graph invariants on the higher level edges
(in [evolution/encoding/complex_edge.py](evolution/encoding/complex_edge.py)):

1. The graph has no circle 
2. Output is always reachable from input (implied from 3)
3. All the vertices should be reachable from input
4. All the vertices could reach output

And invariants for the class properties:

* `input_vertex` is not None
* `output_vertex` is not None
* `vertices_topo_order` always contains vertices sorted in topological 
  order
* Each edge's `end_vertex` should point to the the end vertex of this
  edge, when the edge is in the graph

These invariants are thoroughly tested in the corresponding unit tests
[test/complex_edge_test.py](test/complex_edge_test.py) and 
[test/mutatble_edge_test.py](test/mutable_edge_test.py). For more 
details about the hierarchical graph, take a look at Figure 1 in 
(Lui et al., 2017) for an illustration.

## Run the example

1. Install the dependencies in `run-requirements.txt` by
   `pip3 install -r run-requirements.txt`. Note that if you have Nvidia 
   GPU, you should manually install Tensorflow with GPU instead.
2. Run the CIFAR10 example: `python3 -m examples.cifar10 -o logs`
   under project root directory.
3. Start Tensorboard: `tensorboard --logdir logs` to observe progress, 
   and visualize models.

## References
* Real, E., Aggarwal, A., Huang, Y., & Le, Q. V. (2018). Regularized
  evolution for image classifier architecture search. arXiv preprint
  arXiv:1802.01548.
* He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep residual learning 
  for image recognition. In Proceedings of the IEEE conference on 
  computer vision and pattern recognition (pp. 770-778).
* Liu, H., Simonyan, K., Vinyals, O., Fernando, C., & Kavukcuoglu, K.
  (2017). Hierarchical representations for efficient architecture 
  search. arXiv preprint arXiv:1711.00436.
