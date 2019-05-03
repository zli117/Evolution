"""
A module for complex hierarchical graph.
"""
from abc import abstractmethod
from queue import Queue
from typing import List, Set, Dict

import tensorflow as tf

from evolution.encoding.base import Edge
from evolution.encoding.base import Vertex


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

    def __init__(self, name: str) -> None:
        super().__init__()
        self.input_vertex = Vertex(name='input')
        self.output_vertex = Vertex(name='output')

        self.vertices_topo_order: List[Vertex] = [self.output_vertex,
                                                  self.input_vertex]
        self.name = name
        self._layers_below = -1

    def deep_copy_graph(self, copy: 'ComplexEdge') -> None:
        """
        Deep copy the graph to another complex edge
        Assuming the invariants holds

        Args:
            copy: To which the graph should be copied to
        Returns:
            None
        """
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
                # Check here to make mypy happy
                if edge.end_vertex:
                    copy_edge.end_vertex = copy.vertices_topo_order[
                        edge.end_vertex.order]

        copy.sort_vertices()

    def deep_copy_info(self, copy: 'ComplexEdge') -> None:
        copy.name = self.name
        copy._layers_below = self._layers_below

    @abstractmethod
    def deep_copy(self) -> Edge:
        pass

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
            raise RuntimeError('Found cycle in graph')
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

    def check_output_reachable(self) -> bool:
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

    @abstractmethod
    def mutate(self) -> bool:
        pass

    def invalidate_layer_count(self) -> None:
        self._layers_below = -1

    def build(self, x: tf.Tensor) -> tf.Tensor:
        for vertex in self.vertices_topo_order:
            vertex.reset()
        with tf.name_scope('%s.layer_%d' % (self.name, self.level)):
            self.input_vertex.collect(x)
            for vertex in reversed(self.vertices_topo_order[1:]):
                vertex.submit()
            return self.output_vertex.aggregate()

    @property
    def level(self) -> int:
        if self._layers_below < 1:
            max_layers = 1
            for vertex in self.vertices_topo_order:
                for operation in vertex.out_bound_edges:
                    max_layers = max(max_layers, operation.level)
            self._layers_below = max_layers + 1
        return self._layers_below
