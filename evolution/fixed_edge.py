from abc import abstractmethod

from evolution.complex_edge import ComplexEdge


class FixedEdge(ComplexEdge):

    def mutate(self) -> bool:
        return False

    @abstractmethod
    def build_graph(self) -> None:
        pass
