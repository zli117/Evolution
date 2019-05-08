import argparse
from typing import Optional

import tensorflow as tf
from tensorflow import keras

from evolution.encoding.base import BatchNorm
from evolution.encoding.base import Dense
from evolution.encoding.base import DepthwiseConv2D
from evolution.encoding.base import Dropout
from evolution.encoding.base import Flatten
from evolution.encoding.base import IdentityOperation
from evolution.encoding.base import MaxPool2D
from evolution.encoding.base import PointConv2D
from evolution.encoding.base import ReLU
from evolution.encoding.base import SeparableConv2D
from evolution.encoding.base import Vertex
from evolution.encoding.fixed_edge import FixedEdge
from evolution.encoding.mutable_edge import MutableEdge
from evolution.evolve.evolve_strategy import aging_evolution
from evolution.evolve.mutation_strategy import MutateOneLayer

batch_size = 32
num_classes = 10
epochs = 20

(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)


class TopLayer(FixedEdge):

    def __init__(self) -> None:
        super().__init__(name='TopLayer')

    def construct_new_instance(self) -> 'FixedEdge':
        return TopLayer()

    def build_graph(self) -> None:
        conv_edge1 = MutableEdge((BatchNorm(),
                                  PointConv2D((20, 40)), DepthwiseConv2D(),
                                  IdentityOperation(),
                                  SeparableConv2D((20, 40)), Dropout(0.25),
                                  ReLU()), max_vertices=10,
                                 initialize_with_identity=False)
        conv_edge2 = conv_edge1.deep_copy()
        vertex1 = Vertex(name='V1')
        vertex2 = Vertex(name='V2')
        self.input_vertex.add_edge(conv_edge1, vertex1)
        vertex7 = Vertex(name='V7')
        vertex1.add_edge(MaxPool2D(), vertex7)
        vertex7.add_edge(conv_edge2, vertex2)

        vertex3 = Vertex(name='V3')
        vertex2.add_edge(Flatten(), vertex3)
        vertex4 = Vertex(name='V4')
        vertex3.add_edge(Dense(512), vertex4)
        vertex5 = Vertex(name='V5')
        vertex4.add_edge(ReLU(), vertex5)
        vertex6 = Vertex(name='V6')

        vertex5.add_edge(Dropout(0.5), vertex6)
        vertex6.add_edge(Dense(num_classes), self.output_vertex)

    def build(self, x: tf.Tensor) -> tf.Tensor:
        logit = super().build(x)
        return keras.layers.Activation('softmax')(logit)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Example for evolving a neural net on cifar10 dataset')
    parser.add_argument('-p', type=int, default=20, help='Population size')
    parser.add_argument('-i', type=int, default=10,
                        help='Number of evolution iterations')
    parser.add_argument('-s', type=int, default=5,
                        help='Sample how many individuals in each iteration')
    parser.add_argument('-o', type=str, required=True,
                        help='Log directory path')

    args: Optional[argparse.Namespace] = None

    try:
        args = parser.parse_args()
    except argparse.ArgumentError:
        parser.print_help()
        exit(1)

    train_eval_args = {
        'k_folds': 3,
        'X': x_train,
        'y': y_train,
        'X_test': x_test,
        'y_test': y_test,
        'fit_args': {'batch_size': batch_size, 'epochs': epochs,
                     'shuffle': True},
        'optimizer_factory': lambda: keras.optimizers.RMSprop(lr=1e-4,
                                                              decay=1e-6),
        'loss': 'categorical_crossentropy',
        'metrics': 'accuracy',
        'log_dir': args.l,
    }
    model, performance = aging_evolution(args.p, args.i, args.s, TopLayer(),
                                         MutateOneLayer(), train_eval_args)
    print('Best accuracy:', performance)
