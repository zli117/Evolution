from abc import abstractmethod

from evolution.base import Edge
from evolution.complex_edge import ComplexEdge


class FixedEdge(ComplexEdge):

    def __init__(self) -> None:
        super().__init__()
        self.build_graph()
        if not self.check_output_reachable():
            raise RuntimeError('Output not reachable')
        self.sort_vertices()

    def mutate(self) -> bool:
        return False

    @abstractmethod
    def deep_copy(self) -> Edge:
        """
        Invoke the corresponding constructor of decedent class.
        Returns:
            The edge that's identical to this fixed edge
        """
        pass

    @abstractmethod
    def build_graph(self) -> None:
        pass
