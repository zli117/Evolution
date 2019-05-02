"""
Basic units
"""
from abc import ABC
from abc import abstractmethod
from random import randint
from typing import List, Optional, Tuple

import tensorflow as tf
from tensorflow import keras


class Vertex(object):

    def __init__(self):
        self.out_bound_edges: List['Edge'] = []
        self.collected: List[tf.Tensor] = []
        self.order: int = 0

    def reset(self) -> None:
        self.collected = []

    def collect(self, x: tf.Tensor) -> None:
        self.collected.append(x)

    def aggregate(self) -> tf.Tensor:
        if not self.collected:
            # Should throw error
            pass
        elif len(self.collected) == 1:
            return self.collected[0]
        else:
            return keras.layers.concatenate(self.collected)

    def submit(self) -> None:
        aggregated = self.aggregate()
        for out_edge in self.out_bound_edges:
            out_edge.build(aggregated)

    def remove_edge(self, edge: 'Edge') -> bool:
        if edge in self.out_bound_edges:
            self.out_bound_edges.remove(edge)
            return True
        return False


class Edge(ABC):

    def __init__(self) -> None:
        self._source_vertex: Optional[Vertex] = None
        self._end_vertex: Optional[Vertex] = None

    @property
    def source_vertex(self) -> Optional[Vertex]:
        return self._source_vertex

    @source_vertex.setter
    def source_vertex(self, vertex: Vertex) -> None:
        self._source_vertex = vertex

    @property
    def end_vertex(self) -> Optional[Vertex]:
        return self._end_vertex

    @end_vertex.setter
    def end_vertex(self, vertex: Vertex) -> None:
        self._end_vertex = vertex

    def submit(self, x: tf.Tensor) -> None:
        processed = self.build(x)
        if self._end_vertex:
            self._end_vertex.collect(processed)

    @abstractmethod
    def mutate(self) -> bool:
        pass

    @abstractmethod
    def build(self, x: tf.Tensor) -> tf.Tensor:
        pass

    @property
    @abstractmethod
    def layers_below(self) -> int:
        pass

    @abstractmethod
    def deep_copy(self) -> 'Edge':
        pass


class IdentityOperation(Edge):

    def mutate(self) -> bool:
        return False

    def build(self, x: tf.Tensor) -> tf.Tensor:
        return x

    @property
    def layers_below(self) -> int:
        return 1

    def deep_copy(self) -> 'Edge':
        return IdentityOperation()


class _LayerWrapperMutableChannels(Edge):

    def __init__(self, out_channel_range: Tuple[int, int]) -> None:
        super().__init__()
        self.out_channel_range = out_channel_range
        self._layer: keras.layers.Layer = None
        self.mutate()

    def mutate(self) -> bool:
        out_channels = randint(*self.out_channel_range)
        self._layer = self.build_layer(out_channels)
        return True

    def build(self, x: tf.Tensor) -> tf.Tensor:
        return self._layer(x)

    @property
    def layers_below(self) -> int:
        return 1

    @abstractmethod
    def build_layer(self, out_channels: int) -> keras.layers.Layer:
        pass


class _LayerWrapperImmutableChannels(Edge):

    def __init__(self) -> None:
        super().__init__()
        self._layer: keras.layers.Layer = self.build_layer()

    def mutate(self) -> bool:
        return False

    def build(self, x: tf.Tensor) -> tf.Tensor:
        return self._layer(x)

    @property
    def layers_below(self) -> int:
        return 1

    @abstractmethod
    def build_layer(self) -> keras.layers.Layer:
        pass


class PointConv2D(_LayerWrapperMutableChannels):

    def build_layer(self, out_channels: int) -> keras.layers.Layer:
        return keras.layers.Conv2D(kernel_size=(1, 1),
                                   filters=out_channels, padding='same')

    def deep_copy(self) -> 'Edge':
        return PointConv2D(self.out_channel_range)


class SeparableConv2D(_LayerWrapperMutableChannels):

    def build_layer(self, out_channels: int) -> keras.layers.Layer:
        return keras.layers.SeparableConv2D(kernel_size=(3, 3),
                                            filters=out_channels,
                                            padding='same')

    def deep_copy(self) -> 'Edge':
        return SeparableConv2D(self.out_channel_range)


class DepthwiseConv2D(_LayerWrapperImmutableChannels):

    def build_layer(self) -> keras.layers.Layer:
        return keras.layers.DepthwiseConv2D(kernel_size=(3, 3), padding='same')

    def deep_copy(self) -> 'Edge':
        return DepthwiseConv2D()


class MaxPool2D(_LayerWrapperImmutableChannels):

    def build_layer(self) -> keras.layers.Layer:
        return keras.layers.MaxPool2D(pool_size=3, padding='same')

    def deep_copy(self) -> 'Edge':
        return MaxPool2D()


class AvePool2D(_LayerWrapperImmutableChannels):

    def build_layer(self) -> keras.layers.Layer:
        return keras.layers.AveragePooling2D(pool_size=3, padding='same')

    def deep_copy(self) -> 'Edge':
        return AvePool2D()


class BatchNorm(_LayerWrapperImmutableChannels):

    def build_layer(self) -> keras.layers.Layer:
        return keras.layers.BatchNormalization()

    def deep_copy(self) -> 'Edge':
        return BatchNorm()


class ReLU(_LayerWrapperImmutableChannels):

    def build_layer(self) -> keras.layers.Layer:
        return keras.layers.ReLU()

    def deep_copy(self) -> 'Edge':
        return ReLU()


class ELU(_LayerWrapperImmutableChannels):

    def build_layer(self) -> keras.layers.Layer:
        return keras.layers.ELU()

    def deep_copy(self) -> 'Edge':
        return ELU()
