from evolution.base import Edge
from evolution.base import IdentityOperation
from evolution.base import Vertex
from evolution.fixed_edge import FixedEdge


class NoCycleConnected(FixedEdge):

    def deep_copy(self) -> Edge:
        return NoCycleConnected()

    def build_graph(self) -> None:
        vertex1 = Vertex()
        vertex2 = Vertex()

        edge1 = IdentityOperation()
        edge2 = IdentityOperation()
        edge3 = IdentityOperation()
        edge4 = IdentityOperation()
        edge5 = IdentityOperation()

        self.input_vertex.out_bound_edges.extend([edge1, edge2])
        edge1.end_vertex = vertex1
        edge2.end_vertex = vertex2
        vertex2.out_bound_edges.append(edge3)
        edge3.end_vertex = vertex1

        vertex1.out_bound_edges.append(edge4)
        vertex2.out_bound_edges.append(edge5)
        edge4.end_vertex = self.output_vertex
        edge5.end_vertex = self.output_vertex


class NoCycleNotConnected(FixedEdge):

    def deep_copy(self) -> Edge:
        return NoCycleConnected()

    def build_graph(self) -> None:
        vertex1 = Vertex()
        vertex2 = Vertex()

        edge1 = IdentityOperation()
        edge2 = IdentityOperation()
        edge3 = IdentityOperation()

        self.input_vertex.out_bound_edges.extend([edge1, edge2])
        edge1.end_vertex = vertex1
        edge2.end_vertex = vertex2
        vertex2.out_bound_edges.append(edge3)
        edge3.end_vertex = vertex1


class CycleConnected(FixedEdge):

    def deep_copy(self) -> Edge:
        return NoCycleConnected()

    def build_graph(self) -> None:
        vertex1 = Vertex()
        vertex2 = Vertex()

        edge1 = IdentityOperation()
        edge2 = IdentityOperation()
        edge3 = IdentityOperation()
        edge4 = IdentityOperation()
        edge5 = IdentityOperation()

        self.input_vertex.out_bound_edges.extend([edge1, edge2])
        edge1.end_vertex = vertex1
        edge2.end_vertex = vertex2
        vertex2.out_bound_edges.append(edge3)
        edge3.end_vertex = vertex1

        vertex1.out_bound_edges.extend([edge4, edge5])
        edge4.end_vertex = self.output_vertex
        edge5.end_vertex = self.input_vertex


def test_invariant_checks():
    try:
        NoCycleConnected()
    except RuntimeError:
        assert False

    try:
        NoCycleNotConnected()
        assert False
    except RuntimeError:
        pass

    try:
        CycleConnected()
        assert False
    except RuntimeError:
        pass
