"""
Basic units
"""
from abc import ABC, abstractmethod
from random import randint
from typing import Tuple

import tensorflow as tf
from tensorflow import keras


class Operation(ABC):

    def __init__(self) -> None:
        pass

    @abstractmethod
    def output_shape(self, input_shape: tf.TensorShape) -> tf.TensorShape:
        pass

    @abstractmethod
    def mutate(self) -> bool:
        pass

    @abstractmethod
    def build(self, x: tf.Tensor) -> tf.Tensor:
        pass

    @abstractmethod
    @property
    def layers_below(self) -> int:
        pass


class IdentityOperation(Operation):

    def output_shape(self, input_shape: tf.TensorShape) -> tf.TensorShape:
        return input_shape

    def mutate(self) -> bool:
        return False

    def build(self, x: tf.Tensor) -> tf.Tensor:
        return x

    @property
    def layers_below(self) -> int:
        return 1


class _LayerWrapperMutableChannels(Operation):

    def __init__(self, out_channel_range: Tuple[int, int]) -> None:
        super().__init__()
        self.out_channel_range = out_channel_range
        self._layer: keras.layers.Layer = None
        self.mutate()

    def output_shape(self, input_shape: tf.TensorShape) -> tf.TensorShape:
        return self._layer.compute_output_shape(input_shape)

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


class _LayerWrapperImmutableChannels(Operation):

    def __init__(self) -> None:
        super().__init__()
        self._layer: keras.layers.Layer = self.build_layer()

    def output_shape(self, input_shape: tf.TensorShape) -> tf.TensorShape:
        return self._layer.compute_output_shape(input_shape)

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
                                   filters=out_channels)


class SeparableConv2D(_LayerWrapperMutableChannels):

    def build_layer(self, out_channels: int) -> keras.layers.Layer:
        return keras.layers.SeparableConv2D(kernel_size=(3, 3),
                                            filters=out_channels)


class DepthwiseConv2D(_LayerWrapperImmutableChannels):

    def build_layer(self) -> keras.layers.Layer:
        return keras.layers.DepthwiseConv2D(kernel_size=(3, 3))


class MaxPool2D(_LayerWrapperImmutableChannels):

    def build_layer(self) -> keras.layers.Layer:
        return keras.layers.MaxPool2D(pool_size=3)


class AvePool2D(_LayerWrapperImmutableChannels):

    def build_layer(self) -> keras.layers.Layer:
        return keras.layers.AveragePooling2D(pool_size=3)


class BatchNorm(_LayerWrapperImmutableChannels):

    def build_layer(self) -> keras.layers.Layer:
        return keras.layers.BatchNormalization()


class ReLU(_LayerWrapperImmutableChannels):

    def build_layer(self) -> keras.layers.Layer:
        return keras.layers.ReLU()


class ELU(_LayerWrapperImmutableChannels):

    def build_layer(self) -> keras.layers.Layer:
        return keras.layers.ELU()
