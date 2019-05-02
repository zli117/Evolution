import tensorflow as tf

from evolution.base import Edge
from evolution.base import Vertex

class FixedEdge(Edge):

    def mutate(self) -> bool:
        pass

    def build(self, x: tf.Tensor) -> tf.Tensor:
        pass

    @property
    def layers_below(self) -> int:
        pass

    def deep_copy(self) -> 'Edge':
        pass