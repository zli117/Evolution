from typing import List, Tuple, cast, Dict, Any

import numpy as np

from evolution.encoding.complex_edge import ComplexEdge
from evolution.evolve.mutation_strategy import MutationStrategy
from evolution.train.model import train_and_eval


def aging_evolution(population_size: int, iterations: int, sample_size: int,
                    initial_model: ComplexEdge,
                    mutation: MutationStrategy,
                    train_eval_args: Dict[str, Any]
                    ) -> Tuple[ComplexEdge, float]:
    population: List[Tuple[ComplexEdge, float]] = []
    history: List[Tuple[ComplexEdge, float]] = []
    while len(population) < population_size:
        copy: ComplexEdge = cast(ComplexEdge, initial_model.deep_copy())
        for _ in range(np.random.randint(1, 5)):
            mutation(copy)
        metrics = train_and_eval(copy, name='gen_%d' % len(population),
                                 **train_eval_args)
        population.append((copy, metrics))
        history.append((copy, metrics))

    while len(history) < iterations:
        sample = np.random.choice(population, size=sample_size)
        max_metrics = 0.0
        parent: ComplexEdge = sample[0][0]
        for edge, metrics in sample:
            if metrics > max_metrics:
                parent = edge
        child: ComplexEdge = cast(ComplexEdge, parent.deep_copy())
        mutation(child)
        child_metrics = train_and_eval(child, name='gen_%d' % len(history),
                                       **train_eval_args)
        population.append((child, child_metrics))
        history.append((child, child_metrics))
        population.pop(0)

    return max(history, key=lambda x: x[1])
