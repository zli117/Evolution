import os
from abc import ABC
from abc import abstractmethod
from dataclasses import dataclass
from multiprocessing import Pool
from typing import Dict, Any, Tuple, Generator

import numpy as np
import tensorflow as tf
from sklearn.model_selection import KFold
from tensorflow import keras

from evolution.encoding.base import Edge


class BaseTrainer(ABC):

    @abstractmethod
    def train_and_eval(self, higher_level_model: Edge, name: str) -> float:
        pass

    @abstractmethod
    def optimizer_factory(self) -> keras.optimizers.Optimizer:
        pass


@dataclass
class Trainer(BaseTrainer):
    k_folds: int
    num_process: int
    x_train: np.array
    y_train: np.array
    x_valid: np.array
    y_valid: np.array
    fit_args: Dict[str, Any]
    loss: Any
    metrics: Any

    def _data_generator(self) -> Generator[Tuple[np.array, np.array,
                                                 np.array, np.array, int],
                                           None, None]:
        kf = KFold(n_splits=self.k_folds)
        for i, index in enumerate(kf.split(self.x_train)):
            train_idx, valid_idx = index
            x_train: np.array = self.x_train[train_idx]
            x_valid: np.array = self.x_train[valid_idx]
            y_train: np.array = self.y_train[train_idx]
            y_valid: np.array = self.y_train[valid_idx]
            yield x_train, x_valid, y_train, y_valid, i

    def _worker(self, param: Tuple[np.array, np.array, np.array, np.array,
                                   int]) -> float:
        x_train, x_valid, y_train, y_valid, cv_iter = param
        gpu_options = tf.GPUOptions(
            per_process_gpu_memory_fraction=1 / self.num_process)
        with tf.Session(
                config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
            keras.backend.set_session(sess)
            input_tensor = keras.Input(shape=x_train.shape[1:])
            out = edge.build(input_tensor)
            model = keras.Model(inputs=input_tensor, outputs=out)
            tensor_board = keras.callbacks.TensorBoard(
                batch_size=10, write_graph=True,
                log_dir=os.path.join(os.path.join('logs', name),
                                     'cv_%d' % cv_iter))

            model.compile(loss=self.loss,
                          optimizer=self.optimizer_factory(),
                          metrics=[self.metrics])
            model.fit(x_train, y_train, validation_data=(x_valid, y_valid),
                      callbacks=[tensor_board], **self.fit_args)

            _, test_metrics = model.evaluate(self.x_valid, self.y_valid,
                                             verbose=1)
            return test_metrics

    def train_and_eval(self, edge: Edge, name: str) -> float:
        with Pool(self.num_process) as pool:
            history = list(pool.map(self._worker, self._data_generator()))
            # history = list(pool.map(self.test, self._data_generator()))
            return float(np.mean(history))

    def test(self, param):
        return 1

    def optimizer_factory(self) -> keras.optimizers.Optimizer:
        return keras.optimizers.Adam()
