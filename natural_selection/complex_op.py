"""

"""
from enum import Enum
from queue import Queue
from random import choice
from typing import Tuple, List

import tensorflow as tf

from natural_selection.base import IdentityOperation
from natural_selection.base import Operation
from natural_selection.base import Vertex


class MutationTypes(Enum):
    REMOVE_EDGE = 0,
    ADD_EDGE = 1,
    REMOVE_NODE = 2,
    ADD_NODE = 3,
    MUTATE_EDGE = 4,


class Initialization(Enum):
    IDENTITY = 0,
    RANDOM_FROM_OPS = 1,


class HigherLevelOperation(Operation):

    def __init__(self, operations: Tuple[Operation],
                 initialize: Initialization = Initialization.IDENTITY) -> None:
        super().__init__()
        self.operations = operations
        self.input_vertex = Vertex()
        self.output_vertex = Vertex()
        if initialize == Initialization.IDENTITY:
            edge: Operation = IdentityOperation()
        else:
            edge = choice(self.operations)
        self.input_vertex.out_bound_edges.append(edge)
        self.vertices: List[Vertex] = [self.input_vertex, self.output_vertex]
        self._compute_tiers()

    def _compute_tiers(self) -> None:
        # Basically BFS
        bfs_queue: Queue = Queue()
        self.input_vertex.tier = 0
        bfs_queue.put(self.input_vertex)
        while not bfs_queue.empty():
            vertex: Vertex = bfs_queue.get()
            for out_edge in vertex.out_bound_edges:
                neighbor = out_edge.end_vertex
                if neighbor:
                    neighbor.tier = vertex.tier + 1
                    bfs_queue.put(neighbor)

    def output_shape(self, input_shape: tf.TensorShape) -> tf.TensorShape:
        pass

    def mutate(self) -> bool:
        pass

    def build(self, x: tf.Tensor) -> tf.Tensor:
        pass

    @property
    def layers_below(self) -> int:
        pass
