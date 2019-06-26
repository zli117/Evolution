import argparse
import os
from typing import Any, Optional

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf  # noqa
from tensorflow import keras  # noqa

from evolution.encoding.base import BatchNorm  # noqa
from evolution.encoding.base import Dense  # noqa
from evolution.encoding.base import DepthwiseConv2D  # noqa
from evolution.encoding.base import Dropout  # noqa
from evolution.encoding.base import Flatten  # noqa
from evolution.encoding.base import IdentityOperation  # noqa
from evolution.encoding.base import MaxPool2D  # noqa
from evolution.encoding.base import PointConv2D  # noqa
from evolution.encoding.base import ReLU  # noqa
from evolution.encoding.base import SeparableConv2D  # noqa
from evolution.encoding.base import Vertex  # noqa
from evolution.encoding.fixed_edge import FixedEdge  # noqa
from evolution.encoding.mutable_edge import MutableEdge  # noqa
from evolution.evolve.evolve_strategy import aging_evolution  # noqa
from evolution.evolve.mutation_strategy import MutateOneLayer  # noqa
from evolution.train.trainer import ParallelTrainer  # noqa

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


class Cifar10ParallelTrainer(ParallelTrainer):

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

    def optimizer_factory(self) -> keras.optimizers.Optimizer:
        return keras.optimizers.RMSprop(lr=1e-4, decay=1e-6)


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
        'num_process': 3,
        'x_train': x_train,
        'y_train': y_train,
        'x_valid': x_test,
        'y_valid': y_test,
        'fit_args': {'batch_size': batch_size,
                     'epochs': epochs,
                     'shuffle': True,
                     'verbose': 0},
        'loss': 'categorical_crossentropy',
        'metrics': 'accuracy', }
    model, performance = aging_evolution(20, 10, 5, TopLayer(),
                                         MutateOneLayer(),
                                         Cifar10ParallelTrainer(
                                             **train_eval_args))
