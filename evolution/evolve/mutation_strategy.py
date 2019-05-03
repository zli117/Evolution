from abc import ABC
from abc import abstractmethod
from queue import Queue
from typing import List, Set, cast

import numpy as np

from evolution.encoding.base import Edge
from evolution.encoding.complex_edge import ComplexEdge


class MutationStrategy(ABC):

    @abstractmethod
    def __call__(self, edge_to_mutate: ComplexEdge) -> bool:
        pass


class MutateOneLayer(MutationStrategy):

    def _mutate_one_level(self, root_edge: ComplexEdge,
                          level_to_mutate: int) -> bool:
        """
        Mutate one level in the tree of edges

        Args:
            root_edge: The root edge
            level_to_mutate: Which level to mutate

        Returns:
            True if anything has been mutated. False otherwise
        """
        # Traverse the tree
        queue: Queue = Queue()
        edges_of_level: List[Edge] = []

        # Standard BFS on tree to collect all edges
        visited_set: Set[Edge] = set()
        visited_set.add(root_edge)
        queue.put(root_edge)
        while not queue.empty():
            current: Edge = queue.get()
            if current.level == level_to_mutate:
                edges_of_level.append(current)
            if current.level == 1:
                continue
            for vertex in cast(ComplexEdge, current).vertices_topo_order:
                for edge in vertex.out_bound_edges:
                    if edge in visited_set:
                        continue
                    visited_set.add(edge)
                    queue.put(edge)

        # Try edges in random order until there's one we can mutate
        for edge in np.random.permutation(edges_of_level):
            if edge.mutate():
                edge.invalidate_layer_count()
                return True

        return False

    def __call__(self, root_edge: ComplexEdge) -> bool:
        for level in np.random.permutation(range(1, root_edge.level + 1)):
            if self._mutate_one_level(root_edge, level):
                return True
        return False
