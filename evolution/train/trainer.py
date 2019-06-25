import os
from abc import ABC
from abc import abstractmethod
from dataclasses import dataclass
from multiprocessing import Pool
from typing import Dict, Any, Tuple, Generator, List

import numpy as np
import tensorflow as tf
from sklearn.model_selection import KFold
from tensorflow import keras
from tensorflow.python.client import device_lib

from evolution.encoding.base import Edge


class BaseTrainer(ABC):

    @abstractmethod
    def train_and_eval(self, higher_level_model: Edge, name: str) -> float:
        pass

    @abstractmethod
    def optimizer_factory(self) -> keras.optimizers.Optimizer:
        pass


@dataclass
class ParallelTrainer(BaseTrainer):
    k_folds: int
    num_process: int
    x_train: np.array
    y_train: np.array
    x_valid: np.array
    y_valid: np.array
    fit_args: Dict[str, Any]
    loss: Any
    metrics: Any

    def _param_generator(self, edge: Edge, name: str,
                         gpus: List[Any],
                         cpus: List[Any]) -> Generator[Tuple[np.array,
                                                             np.array,
                                                             np.array,
                                                             np.array,
                                                             Edge, str, str,
                                                             float],
                                                       None, None]:
        kf = KFold(n_splits=self.k_folds)
        total_gpu_memory = sum([device.memory_limit for device in gpus])
        if total_gpu_memory == 0:
            device_allocation: List[Tuple[str, float]] = [(str(device.name), 0)
                                                          for device in cpus]
        else:
            gpu_process_count = [
                int(self.num_process * device.memory_limit / total_gpu_memory)
                for device in gpus]
            device_allocation = [
                (str(device.name), 1 / count)
                for device, count
                in zip(gpus, gpu_process_count)]

        for i, index in enumerate(kf.split(self.x_train)):
            train_idx, valid_idx = index
            x_train: np.array = self.x_train[train_idx]
            x_valid: np.array = self.x_train[valid_idx]
            y_train: np.array = self.y_train[train_idx]
            y_valid: np.array = self.y_train[valid_idx]
            dev_name, allocation = device_allocation[i % len(device_allocation)]
            yield (x_train, x_valid, y_train, y_valid, edge,
                   os.path.join(os.path.join('logs', name), 'cv_%d' % i),
                   dev_name, allocation)

    def _worker(self, param: Tuple[np.array, np.array, np.array, np.array,
                                   Edge, str, str, float]) -> float:
        x_train, x_valid, y_train, y_valid, edge, name, device, memory = param
        gpu_options = tf.compat.v1.GPUOptions(
            per_process_gpu_memory_fraction=0.7 * memory)
        # graph = tf.Graph()
        # with graph.device(device):
        with tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(
                gpu_options=gpu_options)) as sess:
            devices = sess.list_devices()
            for d in devices:
                print(d.name)

            tf.compat.v1.keras.backend.set_session(sess)

            input_tensor = keras.Input(shape=x_train.shape[1:])
            out = edge.build(input_tensor)
            model = keras.Model(inputs=input_tensor, outputs=out)
            model.compile(loss=self.loss,
                          optimizer=self.optimizer_factory(),
                          metrics=[self.metrics])

            tensor_board = keras.callbacks.TensorBoard(
                batch_size=10, write_graph=True,
                log_dir=name)
            printing_callback = keras.callbacks.LambdaCallback(
                on_epoch_end=lambda epoch, logs: print(
                    '%s end epoch %s' % (name, epoch)))

            model.fit(x_train, y_train, validation_data=(x_valid, y_valid),
                      callbacks=[tensor_board, printing_callback],
                      **self.fit_args)

            _, test_metrics = model.evaluate(self.x_valid, self.y_valid,
                                             verbose=1)
            return test_metrics

    def train_and_eval(self, edge: Edge, name: str) -> float:
        local_devices = device_lib.list_local_devices()
        available_gpus = [device for device
                          in local_devices
                          if device.device_type == 'GPU']
        available_cpus = [device for device
                          in local_devices
                          if device.device_type == 'CPU']

        with Pool(self.num_process) as pool:
            history = list(
                pool.map(self._worker,
                         self._param_generator(edge, name, available_gpus,
                                               available_cpus)))
            return float(np.mean(history))

    def optimizer_factory(self) -> keras.optimizers.Optimizer:
        return keras.optimizers.Adam()
