import os
from typing import Dict, Any, Callable

import numpy as np
from sklearn.model_selection import KFold
from tensorflow import keras

from evolution.encoding.base import Edge


def train_and_eval(edge: Edge, k_folds: int, X: np.array, y: np.array,
                   X_test: np.array, y_test: np.array, fit_args: Dict[str, Any],
                   optimizer_factory: Callable[[], keras.optimizers.Optimizer],
                   loss: str, metrics: str, name: str) -> float:
    kf = KFold(n_splits=k_folds)
    history = []
    for i, index in enumerate(kf.split(X)):
        train_idx, valid_idx = index
        x_train: np.array = X[train_idx]
        x_valid: np.array = X[valid_idx]
        y_train: np.array = y[train_idx]
        y_valid: np.array = y[valid_idx]

        input_tensor = keras.Input(shape=x_train.shape[1:])

        out = edge.build(input_tensor)

        model = keras.Model(inputs=input_tensor, outputs=out)

        tensor_board = keras.callbacks.TensorBoard(batch_size=10,
                                                   write_graph=True,
                                                   log_dir=os.path.join(
                                                       os.path.join('logs',
                                                                    name),
                                                       'cv_%d' % i))

        model.compile(loss=loss,
                      optimizer=optimizer_factory(),
                      metrics=[metrics])
        model.fit(x_train, y_train, validation_data=(x_valid, y_valid),
                  callbacks=[tensor_board], **fit_args)

        _, test_metrics = model.evaluate(X_test, y_test, verbose=1)
        history.append(test_metrics)

    return float(np.mean(history))
