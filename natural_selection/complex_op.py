"""

"""
from enum import Enum
from typing import Tuple, List, Set, Dict

import numpy as np
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


class ComplexOperation(Operation):

    def __init__(self, available_operations: Tuple[Operation],
                 initialize_with_identity: bool = True) -> None:
        super().__init__()
        self.available_operations = available_operations
        self.input_vertex = Vertex()
        self.output_vertex = Vertex()
        self.inuse_operations: Set[Operation] = set()

        if initialize_with_identity:
            edge: Operation = IdentityOperation()
        else:
            edge, = np.random.choice(self.available_operations, size=1)
        self.inuse_operations.add(edge)
        self.input_vertex.out_bound_edges.append(edge)
        self.vertices_topo_order: List[Vertex] = [self.input_vertex,
                                                  self.output_vertex]
        self._compute_topo_order()

    def _topo_sort_recursion(self, current: Vertex,
                             vertex_list: List[Vertex],
                             accessing_set: Set[Vertex],
                             finished_status: Dict[Vertex, bool]) -> bool:
        """

        Args:
            current:
            vertex_list:
            accessing_set:
            finished_status:

        Returns:

        """
        if current in accessing_set:
            # Error. Circle
            return False
        if current in finished_status:
            return finished_status[current]
        can_reach_output = current == self.output_vertex
        for out_edge in current.out_bound_edges:
            if out_edge.end_vertex:
                can_reach_output |= self._topo_sort_recursion(
                    out_edge.end_vertex, vertex_list, accessing_set,
                    finished_status)
        finished_status[current] = can_reach_output
        accessing_set.remove(current)
        if can_reach_output:
            vertex_list.append(current)
        return can_reach_output

    def _compute_topo_order(self) -> None:
        """
        Sort the vertices in topological order. Maintains the invariant that
        vertices_topo_order contains vertices sorted in topological order.

        Returns:
            None
        """
        vertex_list: List[Vertex] = []
        accessing_set: Set[Vertex] = set()
        finished_status: Dict[Vertex, bool] = dict()
        self._topo_sort_recursion(self.input_vertex, vertex_list,
                                  accessing_set, finished_status)
        self.vertices_topo_order = vertex_list
        for order, vertex in enumerate(vertex_list):
            vertex.order = order

    def _mutation_add_edge(self) -> None:
        vertex1, vertex2 = np.random.choice(self.vertices_topo_order, size=2,
                                            replace=False)
        # Never have backward edge, to prevent cycle
        if vertex1.order < vertex2.order:
            from_vertex: Vertex = vertex2
            to_vertex: Vertex = vertex1
        else:
            from_vertex = vertex1
            to_vertex = vertex2

        edge: Operation = np.random.choice(self.available_operations, size=1)[0]
        from_vertex.out_bound_edges.append(edge)
        edge.end_vertex = to_vertex
        self.inuse_operations.add(edge)
        self._compute_topo_order()

    def _mutation_mutate_edge(self) -> None:
        pass

    def _mutation_remove_edge(self) -> None:
        pass

    def _mutation_add_node(self) -> None:
        pass

    def _mutation_remove_node(self) -> None:
        pass

    def mutate(self) -> bool:
        mutation_type, = np.random.choice(list(MutationTypes), size=1)
        if mutation_type == MutationTypes.ADD_EDGE:
            self._mutation_add_edge()
        elif mutation_type == MutationTypes.MUTATE_EDGE:
            self._mutation_mutate_edge()
        elif mutation_type == MutationTypes.REMOVE_EDGE:
            self._mutation_remove_edge()
        elif mutation_type == MutationTypes.ADD_NODE:
            self._mutation_add_node()
        elif mutation_type == MutationTypes.REMOVE_NODE:
            self._mutation_remove_node()

        return True

    def build(self, x: tf.Tensor) -> tf.Tensor:
        for vertex in self.vertices_topo_order:
            vertex.reset()
        self.input_vertex.collect(x)
        for vertex in reversed(self.vertices_topo_order):
            vertex.submit()
        return self.output_vertex.aggregate()

    @property
    def layers_below(self) -> int:
        max_layers = 1
        for operation in self.inuse_operations:
            max_layers = max(max_layers, operation.layers_below)
        return max_layers + 1
