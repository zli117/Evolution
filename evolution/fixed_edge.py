from abc import abstractmethod

from evolution.base import Edge
from evolution.complex_edge import ComplexEdge


class FixedEdge(ComplexEdge):

    def __init__(self, name: str = '') -> None:
        super().__init__(name)
        self.build_graph()
        if not self.check_output_reachable():
            raise RuntimeError('Output not reachable')
        self.sort_vertices()

    def mutate(self) -> bool:
        return False

    def deep_copy(self) -> Edge:
        new_instance = self.construct_new_instance()
        super().deep_copy_info(new_instance)
        return new_instance

    @abstractmethod
    def construct_new_instance(self) -> 'FixedEdge':
        """
        Invoke the corresponding constructor of decedent class.
        Returns:
            The edge that's identical to this fixed edge
        """
        pass

    @abstractmethod
    def build_graph(self) -> None:
        pass
