from typing import Tuple

import pytest

from evolution.base import IdentityOperation
from evolution.base import MaxPool2D
from evolution.base import PointConv2D
from evolution.base import Vertex
from evolution.mutable_edge import MutableEdge


@pytest.fixture()
def basic_graph_no_v12() -> Tuple[MutableEdge, Vertex, Vertex, Vertex,
                                  Vertex]:
    complex_operation = MutableEdge((PointConv2D((1, 4)), MaxPool2D()))
    vertex1 = Vertex()
    vertex2 = Vertex()
    vertex3 = Vertex()
    vertex4 = Vertex()
    edge1 = IdentityOperation()
    edge2 = IdentityOperation()
    edge3 = IdentityOperation()
    edge4 = IdentityOperation()
    edge5 = IdentityOperation()
    edge6 = IdentityOperation()
    complex_operation.input_vertex.out_bound_edges.clear()
    complex_operation.input_vertex.out_bound_edges.extend([edge1, edge2, edge3])
    edge1.end_vertex = vertex1
    edge2.end_vertex = vertex2
    edge3.end_vertex = vertex4
    vertex1.out_bound_edges.append(edge6)
    edge6.end_vertex = complex_operation.output_vertex
    vertex2.out_bound_edges.append(edge4)
    edge4.end_vertex = complex_operation.output_vertex
    vertex3.out_bound_edges.append(edge5)
    edge5.end_vertex = complex_operation.output_vertex

    return complex_operation, vertex1, vertex2, vertex3, vertex4


@pytest.fixture()
def basic_graph(basic_graph_no_v12) -> Tuple[MutableEdge, Vertex, Vertex,
                                             Vertex, Vertex]:
    complex_operation, vertex1, vertex2, vertex3, vertex4 = basic_graph_no_v12
    edge = IdentityOperation()
    vertex1.out_bound_edges.append(edge)
    edge.end_vertex = vertex2
    return complex_operation, vertex1, vertex2, vertex3, vertex4
