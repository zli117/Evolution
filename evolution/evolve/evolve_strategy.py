from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Tuple, cast

import numpy as np
from tqdm import tqdm

from evolution.encoding.complex_edge import ComplexEdge
from evolution.evolve.mutation_strategy import MutationStrategy
from evolution.train.progress_observer import ProgressObserver
from evolution.train.trainer import BaseTrainer


class EvolveStrategy(ABC):

    @abstractmethod
    def run(self) -> Tuple[ComplexEdge, float]:
        pass


@dataclass
class AgingEvolution(EvolveStrategy, ProgressObserver):
    population_size: int
    iterations: int
    sample_size: int
    initial_model: ComplexEdge
    mutation_strategy: MutationStrategy
    trainer: BaseTrainer

    def __post_init__(self) -> None:
        fmt = '{l_bar}{bar}[{elapsed}<{remaining}, {rate_fmt}{postfix} ]'
        self.progress_bar = tqdm(total=100, bar_format=fmt)
        self.trainer.add_observer(self)

    def run(self) -> Tuple[ComplexEdge, float]:
        population: List[Tuple[ComplexEdge, float]] = []
        history: List[Tuple[ComplexEdge, float]] = []
        self.progress_bar.reset()

        while len(population) < self.population_size:
            copy: ComplexEdge = cast(ComplexEdge,
                                     self.initial_model.deep_copy())
            for _ in range(np.random.randint(1, 5)):
                self.mutation_strategy(copy)
            metrics = self.trainer.train_and_eval(
                copy, name='gen_%d' % len(population))
            population.append((copy, metrics))
            history.append((copy, metrics))

        while len(history) < self.iterations:
            sample = np.random.choice(population, size=self.sample_size)
            max_metrics = 0.0
            parent: ComplexEdge = sample[0][0]
            for edge, metrics in sample:
                if metrics > max_metrics:
                    parent = edge
            child: ComplexEdge = cast(ComplexEdge, parent.deep_copy())
            self.mutation_strategy(child)
            child_metrics = self.trainer.train_and_eval(
                child, name='gen_%d' % len(history))
            population.append((child, child_metrics))
            history.append((child, child_metrics))
            population.pop(0)

        self.progress_bar.close()

        return max(history, key=lambda x: x[1])

    def on_progress(self, name: str, cv_idx: int, epoch_idx: int,
                    total_cv: int) -> None:
        total_progress = (self.population_size + self.iterations) * total_cv
        self.progress_bar.update(1 / total_progress * 100)
        tqdm.write('%s on cv %d ends epoch %s' % (name, cv_idx, epoch_idx))
