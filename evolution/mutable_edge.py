"""
A module for complex hierarchical graph.
"""
from enum import Enum
from typing import Tuple

import numpy as np

from evolution.base import Edge
from evolution.base import IdentityOperation
from evolution.base import Vertex
from evolution.complex_edge import ComplexEdge


class MutationTypes(Enum):
    REMOVE_EDGE = 0,
    ADD_EDGE = 1,
    REMOVE_NODE = 2,
    ADD_NODE = 3,
    MUTATE_EDGE = 4,


class MutableEdge(ComplexEdge):

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

        if initialize_with_identity:
            edge: Edge = IdentityOperation()
        else:
            edge, = np.random.choice(self.available_operations, size=1)
        self.input_vertex.out_bound_edges.append(edge)
        edge.end_vertex = self.output_vertex

        self.sort_vertices()

    def deep_copy(self) -> 'Edge':
        copy_avail_operations = tuple([op.deep_copy()
                                       for op in self.available_operations])
        copy = MutableEdge(copy_avail_operations,
                           max_vertices=self.max_vertices)
        super().deep_copy_to(copy)
        return copy

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
                if not self.check_output_reachable():
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
            for vertex in np.random.permutation(self.vertices_topo_order[
                                                1:-1]):
                out_edges = list(vertex.out_bound_edges)
                vertex.out_bound_edges.clear()
                if not self.check_output_reachable():
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
