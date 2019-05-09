import os
from abc import ABC
from abc import abstractmethod
from dataclasses import dataclass
from typing import Dict, Any, Callable

import numpy as np
from sklearn.model_selection import KFold
from tensorflow import keras

from evolution.encoding.base import Edge


class BaseTrainer(ABC):

    @abstractmethod
    def train_and_eval(self, higher_level_model: Edge) -> float:
        pass


@dataclass
class OneGPUTrainer(BaseTrainer):
    k_folds: int
    x_train: np.array
    y_train: np.array
    x_valid: np.array
    y_valid: np.array
    fit_args: Dict[str, Any]
    optimizer_factory: Callable[[], keras.optimizers.Optimizer]
    loss: Any
    metrics: Any
    name: str

    def train_and_eval(self, model: Edge) -> float:
        kf = KFold(n_splits=self.k_folds)
        history = []
        for i, index in enumerate(kf.split(self.x_train)):
            keras.backend.clear_session()
            train_idx, valid_idx = index
            x_train: np.array = self.x_train[train_idx]
            x_valid: np.array = self.x_train[valid_idx]
            y_train: np.array = self.y_train[train_idx]
            y_valid: np.array = self.y_train[valid_idx]

            input_tensor = keras.Input(shape=x_train.shape[1:])

            out = model.build(input_tensor)

            model = keras.Model(inputs=input_tensor, outputs=out)

            tensor_board = keras.callbacks.TensorBoard(
                batch_size=10, write_graph=True,
                log_dir=os.path.join(os.path.join('logs', self.name),
                                     'cv_%d' % i))

            model.compile(loss=self.loss,
                          optimizer=self.optimizer_factory(),
                          metrics=[self.metrics])
            model.fit(x_train, y_train, validation_data=(x_valid, y_valid),
                      callbacks=[tensor_board], **self.fit_args)

            _, test_metrics = model.evaluate(self.x_valid, self.y_valid,
                                             verbose=1)
            history.append(test_metrics)

        return float(np.mean(history))
