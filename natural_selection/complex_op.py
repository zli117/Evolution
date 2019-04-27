"""

"""
from enum import Enum
from queue import Queue
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
    """
    Complex operation class. This operation encapsulates a small graph of
    nodes and operations. The graph follows such invariants:
    1. The graph has no circle
    2. Output is always reachable from input (implied from 3)
    3. All the vertices should be reachable from input
    4. All the vertices could reach output

    Class level invariants:
    1. input_vertex is not None
    2. output_vertex is not None
    3. inuse_operations contains all the edges that are currently in the graph
    4. vertices_topo_order always contains vertices sorted in topological order
    5. Each edge's end_vertex should point to the the end vertex of this
    edge, when the edge is in the graph
    """

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
        self.vertices_topo_order: List[Vertex] = [self.output_vertex,
                                                  self.input_vertex]
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
        from_vertex: Vertex = max(vertex1, vertex2, key=lambda v: v.order)
        to_vertex: Vertex = min(vertex1, vertex2, key=lambda v: v.order)

        edge: Operation = np.random.choice(self.available_operations, size=1)[0]
        from_vertex.out_bound_edges.append(edge)
        edge.end_vertex = to_vertex
        self.inuse_operations.add(edge)
        self._compute_topo_order()

    def _mutation_mutate_edge(self) -> None:
        vertex, = np.random.choice(self.vertices_topo_order[1:], size=1)
        edge, = np.random.choice(vertex.out_bound_edges, size=1)
        vertex.remove_edge(edge)
        new_edge: Operation = np.random.choice(
            self.available_operations, size=1)[0]
        new_edge.end_vertex = edge.end_vertex
        edge.end_vertex = None
        vertex.out_bound_edges.append(new_edge)

    def _check_output_reachable(self) -> bool:
        """
        Checks for the invariant "All the vertices should be reachable from
        input". Assumes there's no circle in the graph.

        Returns:
            True if output is reachable, False otherwise.
        """
        # Standard BFS
        visited_set: Set[Vertex] = set()
        queue: Queue = Queue()
        visited_set.add(self.input_vertex)
        queue.put(self.input_vertex)
        while not queue.empty():
            current = queue.get()
            for out_edge in current.out_bound_edges:
                if out_edge.end_vertex in visited_set:
                    continue
                new_vertex = out_edge.end_vertex
                queue.put(new_vertex)
                visited_set.add(new_vertex)
                if new_vertex == self.output_vertex:
                    return True
        return False

    def _mutation_remove_edge(self) -> None:
        """
        Randomly remove an edge. Note that in current implementation,
        the probability for each edge being drawn is not the same. The edge
        from a vertex with more outgoing edges is more like to be selected.

        Returns:
            None
        """
        while True:
            vertex, = np.random.choice(self.vertices_topo_order[1:], size=1)
            edge, = np.random.choice(vertex.out_bound_edges, size=1)
            vertex.remove_edge(edge)
            if not self._check_output_reachable():
                # Put the edge back and try a different one
                vertex.out_bound_edges.append(edge)
            else:
                edge.end_vertex = None
                break
        # Since we have changed the graph structure
        self._compute_topo_order()

    def _mutation_add_node(self) -> None:
        vertex1, vertex2 = np.random.choice(self.vertices_topo_order, size=2,
                                            replace=False)
        # Never have backward edge, to prevent cycle
        from_vertex: Vertex = max(vertex1, vertex2, key=lambda v: v.order)
        to_vertex: Vertex = min(vertex1, vertex2, key=lambda v: v.order)

        first, second = np.random.choice(self.available_operations, size=2,
                                         replace=True)
        vertex = Vertex()
        first.end_vertex = vertex
        from_vertex.out_bound_edges.append(first)
        second.end_vertex = to_vertex
        vertex.out_bound_edges.append(second)
        # We changed graph structure
        self._compute_topo_order()

    def _mutation_remove_node(self) -> None:
        if len(self.vertices_topo_order) > 2:
            while True:
                vertex: Vertex = np.random.choice(
                    self.vertices_topo_order[1:-1], size=1)[0]
                out_edges = list(vertex.out_bound_edges)
                vertex.out_bound_edges.clear()
                if not self._check_output_reachable():
                    vertex.out_bound_edges.extend(out_edges)
                else:
                    break
            # We changed graph structure. This will also delete the vertex
            # and all income edges
            self._compute_topo_order()

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
