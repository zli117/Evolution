"""
A module for complex hierarchical graph.
"""
from enum import Enum
from queue import Queue
from typing import Tuple, List, Set, Dict

import numpy as np
import tensorflow as tf

from evolution.base import Edge
from evolution.base import IdentityOperation
from evolution.base import Vertex


class MutationTypes(Enum):
    REMOVE_EDGE = 0,
    ADD_EDGE = 1,
    REMOVE_NODE = 2,
    ADD_NODE = 3,
    MUTATE_EDGE = 4,


class ComplexEdge(Edge):
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
    3. vertices_topo_order always contains vertices sorted in topological order
    4. Each edge's end_vertex should point to the the end vertex of this
    edge, when the edge is in the graph
    """

    def __init__(self, available_operations: Tuple[Edge, ...],
                 initialize_with_identity: bool = True,
                 max_vertices: int = -1) -> None:
        super().__init__()

        if 0 < max_vertices < 2:
            raise RuntimeError(
                'Max vertices: %d is too small. Must be at least 2 if '
                'enabled.' % max_vertices)

        self.available_operations = available_operations
        self.max_vertices = max_vertices
        self.input_vertex = Vertex()
        self.output_vertex = Vertex()

        if initialize_with_identity:
            edge: Edge = IdentityOperation()
        else:
            edge, = np.random.choice(self.available_operations, size=1)
        self.input_vertex.out_bound_edges.append(edge)
        edge.end_vertex = self.output_vertex
        self.vertices_topo_order: List[Vertex] = [self.output_vertex,
                                                  self.input_vertex]
        self.sort_vertices()

    def deep_copy(self) -> 'Edge':
        copy_avail_operations = tuple([op.deep_copy()
                                       for op in self.available_operations])
        copy = ComplexEdge(copy_avail_operations,
                           max_vertices=self.max_vertices)
        # Copy vertices
        for _ in range(len(self.vertices_topo_order) - 2):
            copy.vertices_topo_order.append(Vertex())

        # Clear existing edges
        copy.input_vertex.out_bound_edges.clear()
        # Copy edges
        for i, vertex in enumerate(self.vertices_topo_order):
            copy_vertex = copy.vertices_topo_order[i]
            for edge in vertex.out_bound_edges:
                copy_edge = edge.deep_copy()
                copy_vertex.out_bound_edges.append(copy_edge)
                copy_edge.end_vertex = copy.vertices_topo_order[
                    edge.end_vertex.order]

        copy.sort_vertices()
        return copy

    def _topo_sort_recursion(self, current: Vertex,
                             vertex_list: List[Vertex],
                             accessing_set: Set[int],
                             finished_status: Dict[int, bool]) -> bool:
        """

        Args:
            current:
            vertex_list:
            accessing_set:
            finished_status:

        Returns:

        """
        current_ref = id(current)
        if current_ref in accessing_set:
            return False
        if current_ref in finished_status:
            return finished_status[current_ref]
        accessing_set.add(current_ref)
        to_remove: List[Edge] = []
        for out_edge in current.out_bound_edges:
            if out_edge.end_vertex:
                # If can't reach output, the vertex will be removed, as well
                # as the edge to it.
                if not self._topo_sort_recursion(out_edge.end_vertex,
                                                 vertex_list, accessing_set,
                                                 finished_status):
                    to_remove.append(out_edge)
        can_reach_output = (current is self.output_vertex
                            or len(to_remove) != len(current.out_bound_edges))
        finished_status[current_ref] = can_reach_output
        accessing_set.remove(current_ref)

        for edge in to_remove:
            current.out_bound_edges.remove(edge)
            edge.end_vertex = None

        if can_reach_output:
            vertex_list.append(current)
        return can_reach_output

    def sort_vertices(self) -> None:
        """
        Sort the vertices in topological order. Maintains the invariant that
        vertices_topo_order contains vertices sorted in topological order.

        Returns:
            None
        """
        vertex_list: List[Vertex] = []
        accessing_set: Set[int] = set()
        finished_status: Dict[int, bool] = dict()
        self._topo_sort_recursion(self.input_vertex, vertex_list,
                                  accessing_set, finished_status)
        self.vertices_topo_order = vertex_list
        for order, vertex in enumerate(vertex_list):
            vertex.order = order

    def mutation_add_edge(self) -> None:
        vertex1, vertex2 = np.random.choice(self.vertices_topo_order, size=2,
                                            replace=False)
        # Never have backward edge, to prevent cycle
        from_vertex: Vertex = max(vertex1, vertex2, key=lambda v: v.order)
        to_vertex: Vertex = min(vertex1, vertex2, key=lambda v: v.order)

        edge: Edge = np.random.choice(self.available_operations, size=1)[0]
        edge = edge.deep_copy()
        from_vertex.out_bound_edges.append(edge)
        edge.end_vertex = to_vertex
        self.sort_vertices()

    def mutation_mutate_edge(self) -> None:
        vertex, = np.random.choice(self.vertices_topo_order[1:], size=1)
        edge, = np.random.choice(vertex.out_bound_edges, size=1)
        vertex.remove_edge(edge)
        new_edge: Edge = np.random.choice(
            self.available_operations, size=1)[0]
        new_edge = new_edge.deep_copy()
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

    def mutation_remove_edge(self) -> bool:
        """
        Randomly remove an edge. Note that in current implementation,
        the probability for each edge being drawn is not the same.

        Returns:
            True if mutated. False otherwise.
        """
        # Go through all the edge in random order, until one that will not
        # break the graph is found
        for vertex in np.random.permutation(self.vertices_topo_order):
            for edge in np.random.permutation(vertex.out_bound_edges):
                vertex.remove_edge(edge)
                if not self._check_output_reachable():
                    # Put the edge back and try a different one
                    vertex.out_bound_edges.append(edge)
                else:
                    edge.end_vertex = None
                    self.sort_vertices()
                    return True
        return False

    def mutation_add_vertex(self) -> bool:
        if 2 <= self.max_vertices <= len(self.vertices_topo_order):
            return False
        vertex1, vertex2 = np.random.choice(self.vertices_topo_order, size=2,
                                            replace=False)
        # Never have backward edge, to prevent cycle
        from_vertex: Vertex = max(vertex1, vertex2, key=lambda v: v.order)
        to_vertex: Vertex = min(vertex1, vertex2, key=lambda v: v.order)

        edges: Tuple[Edge, Edge] = np.random.choice(self.available_operations,
                                                    size=2, replace=True)
        first, second = edges
        vertex = Vertex()
        first = first.deep_copy()
        second = second.deep_copy()
        first.end_vertex = vertex
        from_vertex.out_bound_edges.append(first)
        second.end_vertex = to_vertex
        vertex.out_bound_edges.append(second)

        # We changed graph structure
        self.sort_vertices()
        return True

    def mutation_remove_vertex(self) -> bool:
        # We must have input and output
        if len(self.vertices_topo_order) > 2:
            # No input and output vertices
            for vertex in np.random.permutation(self.vertices_topo_order[1:-1]):
                out_edges = list(vertex.out_bound_edges)
                vertex.out_bound_edges.clear()
                if not self._check_output_reachable():
                    vertex.out_bound_edges.extend(out_edges)
                else:
                    self.sort_vertices()
                    return True
        return False

    def mutate(self) -> bool:
        mutation_type, = np.random.choice(list(MutationTypes), size=1)
        if mutation_type == MutationTypes.ADD_EDGE:
            self.mutation_add_edge()
        elif mutation_type == MutationTypes.MUTATE_EDGE:
            self.mutation_mutate_edge()
        elif mutation_type == MutationTypes.REMOVE_EDGE:
            return self.mutation_remove_edge()
        elif mutation_type == MutationTypes.ADD_NODE:
            return self.mutation_add_vertex()
        elif mutation_type == MutationTypes.REMOVE_NODE:
            return self.mutation_remove_vertex()

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
        for vertex in self.vertices_topo_order:
            for operation in vertex.out_bound_edges:
                max_layers = max(max_layers, operation.layers_below)
        return max_layers + 1
