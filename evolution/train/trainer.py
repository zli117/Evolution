import os
from abc import ABC
from abc import abstractmethod
from dataclasses import dataclass
from multiprocessing import Manager
from multiprocessing import Pool
from multiprocessing import Process
from queue import Queue
from typing import Dict, Any, Tuple, Generator, List

import numpy as np
import tensorflow as tf
from sklearn.model_selection import KFold
from tensorflow import keras
from tensorflow.python.client import device_lib

from evolution.encoding.base import Edge
from evolution.train.progress_observer import ProgressObserver


class BaseTrainer(ABC):

    @abstractmethod
    def train_and_eval(self, higher_level_model: Edge, name: str,
                       observers: Tuple[ProgressObserver, ...] = ()) -> float:
        pass

    @abstractmethod
    def optimizer_factory(self) -> keras.optimizers.Optimizer:
        pass


# Since Tensorflow will allocate more memory than specified in
# per_process_gpu_memory_fraction, we need to shrink the allocation fraction.
MEMORY_SHRINK_FACTOR = 0.65


@dataclass
class Params(object):
    x_train: np.array
    x_valid: np.array
    y_train: np.array
    y_valid: np.array
    edge: Edge
    cv_idx: int
    log_dir: str
    device: str
    memory_fraction: float
    name: str
    progress_queue: 'Queue[Tuple[str, int, int, int]]'


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
    log_dir: str

    def __post_init__(self) -> None:
        super().__init__()

    @staticmethod
    def _get_device_info_worker(devices: List[Any]) -> None:
        devices.extend(device_lib.list_local_devices())

    def _get_device_info(self) -> List[Any]:
        manager = Manager()
        devices: List[Any] = manager.list()

        # Since this is a CUDA call, if done in parent process will hang the
        # sub-processes. Fix is to run CUDA call in a separate process. See:
        # https://github.com/tensorflow/tensorflow/issues/8220
        process = Process(target=self._get_device_info_worker,
                          args=(devices,))
        process.start()
        process.join()

        return devices

    def _param_generator(self, edge: Edge, name: str, gpus: List[Any],
                         cpus: List[Any],
                         queue: Queue) -> Generator[Params, None, None]:
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
                (str(device.name), MEMORY_SHRINK_FACTOR / count)
                for device, count
                in zip(gpus, gpu_process_count)]

        for i, index in enumerate(kf.split(self.x_train)):
            train_idx, valid_idx = index
            x_train: np.array = self.x_train[train_idx]
            x_valid: np.array = self.x_train[valid_idx]
            y_train: np.array = self.y_train[train_idx]
            y_valid: np.array = self.y_train[valid_idx]
            dev_name, allocation = device_allocation[i % len(
                device_allocation)]
            yield Params(x_train=x_train, x_valid=x_valid, y_train=y_train,
                         y_valid=y_valid, edge=edge, cv_idx=i,
                         log_dir=os.path.join(os.path.join(self.log_dir, name),
                                              'cv_%d' % i), device=dev_name,
                         memory_fraction=allocation, name=name,
                         progress_queue=queue)

    def _worker(self, params: Params) -> float:
        gpu_options = tf.compat.v1.GPUOptions(
            per_process_gpu_memory_fraction=params.memory_fraction)
        with tf.device(params.device):
            with tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(
                    gpu_options=gpu_options)) as sess:
                tf.compat.v1.keras.backend.set_session(sess)

                input_tensor = keras.Input(shape=params.x_train.shape[1:])
                out = params.edge.build(input_tensor)
                model = keras.Model(inputs=input_tensor, outputs=out)
                model.compile(loss=self.loss,
                              optimizer=self.optimizer_factory(),
                              metrics=[self.metrics])

                tensor_board = keras.callbacks.TensorBoard(
                    batch_size=10, write_graph=True,
                    log_dir=params.log_dir)

                total_epochs = self.fit_args.get('epochs', 1)

                progress_callback = keras.callbacks.LambdaCallback(
                    on_epoch_end=lambda epoch, logs: params.progress_queue.put(
                        (params.name, epoch, params.cv_idx, total_epochs)))

                model.fit(params.x_train, params.y_train,
                          validation_data=(params.x_valid, params.y_valid),
                          callbacks=[tensor_board, progress_callback],
                          **self.fit_args)

                _, test_metrics = model.evaluate(self.x_valid, self.y_valid,
                                                 verbose=0)
                return test_metrics

    def _run_train_pool(self, edge: Edge, name: str,
                        history: List[float], progress_queue: Queue) -> None:
        devices = self._get_device_info()

        available_gpus = [device for device
                          in devices
                          if device.device_type == 'GPU']
        available_cpus = [device for device
                          in devices
                          if device.device_type == 'CPU']

        with Pool(self.num_process) as pool:
            history.extend(pool.map(self._worker,
                                    self._param_generator(edge, name,
                                                          available_gpus,
                                                          available_cpus,
                                                          progress_queue)))
        progress_queue.put(None)

    def train_and_eval(self, edge: Edge, name: str,
                       observers: Tuple[ProgressObserver, ...] = ()) -> float:
        manager = Manager()
        history: List[float] = manager.list()
        queue: Queue = manager.Queue()
        process = Process(target=self._run_train_pool,
                          args=(edge, name, history, queue))
        process.start()
        for name, epoch, cv_idx, total_epoch in iter(queue.get, None):
            for observer in observers:
                observer.on_progress(name, cv_idx, epoch, self.k_folds,
                                     total_epoch)
        process.join()
        return sum(history) / len(history)

    def optimizer_factory(self) -> keras.optimizers.Optimizer:
        return keras.optimizers.Adam()
