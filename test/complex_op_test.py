from typing import Tuple

import pytest

from natural_selection.base import IdentityOperation
from natural_selection.base import MaxPool2D
from natural_selection.base import PointConv2D
from natural_selection.base import Vertex
from natural_selection.complex_op import ComplexOperation


def test_complex_op_creation():
    complex_operation = ComplexOperation((PointConv2D((1, 4)),))
    assert len(complex_operation.available_operations) == 1
    assert len(complex_operation.vertices_topo_order) == 2
    assert (complex_operation.vertices_topo_order[0]
            is complex_operation.output_vertex)
    assert (complex_operation.vertices_topo_order[1]
            is complex_operation.input_vertex)


@pytest.fixture
def basic_graph() -> Tuple[ComplexOperation, Vertex, Vertex, Vertex, Vertex]:
    complex_operation = ComplexOperation((PointConv2D((1, 4)), MaxPool2D()))
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
    edge7 = IdentityOperation()
    complex_operation.input_vertex.out_bound_edges.clear()
    complex_operation.input_vertex.out_bound_edges.extend([edge1, edge2, edge3])
    edge1.end_vertex = vertex1
    edge2.end_vertex = vertex2
    edge3.end_vertex = vertex4
    vertex1.out_bound_edges.extend([edge6, edge7])
    edge6.end_vertex = complex_operation.output_vertex
    edge7.end_vertex = vertex2
    vertex2.out_bound_edges.append(edge4)
    edge4.end_vertex = complex_operation.output_vertex
    vertex3.out_bound_edges.append(edge5)
    edge5.end_vertex = complex_operation.output_vertex

    return complex_operation, vertex1, vertex2, vertex3, vertex4


def test_sort_vertices(basic_graph):
    complex_operation, vertex1, vertex2, vertex3, vertex4 = basic_graph

    complex_operation.sort_vertices()

    assert len(complex_operation.vertices_topo_order) == 4
    assert (complex_operation.vertices_topo_order[0]
            is complex_operation.output_vertex)
    assert (complex_operation.vertices_topo_order[1] is vertex2)
    assert (complex_operation.vertices_topo_order[2] is vertex1)
    assert (complex_operation.vertices_topo_order[3]
            is complex_operation.input_vertex)
    assert len(complex_operation.input_vertex.out_bound_edges) == 2

    edge8 = IdentityOperation()
    vertex2.out_bound_edges.append(edge8)
    edge8.end_vertex = vertex3

    complex_operation.sort_vertices()

    assert vertex3 in complex_operation.vertices_topo_order
    assert complex_operation.output_vertex.order == 0
    assert vertex3.order == 1
    assert vertex2.order == 2
    assert vertex1.order == 3
    assert complex_operation.input_vertex.order == 4

    assert len(complex_operation.output_vertex.out_bound_edges) == 0
    assert len(complex_operation.input_vertex.out_bound_edges) == 2
    assert len(vertex1.out_bound_edges) == 2
    assert len(vertex2.out_bound_edges) == 2
    assert len(vertex3.out_bound_edges) == 1


def test_add_edge1(basic_graph, mocker):
    # Make sure there's no cycle
    complex_operation, vertex1, vertex2, vertex3, vertex4 = basic_graph

    def mock(*args, **kwargs):
        if kwargs['size'] == 2:
            return [complex_operation.output_vertex,
                    complex_operation.input_vertex]
        if kwargs['size'] == 1:
            assert isinstance(args[0][0], PointConv2D)
            return [MaxPool2D()]

    mocker.patch('numpy.random.choice', side_effect=mock)

    complex_operation.mutation_add_edge()
    assert len(complex_operation.vertices_topo_order) == 4
    for vertex in complex_operation.vertices_topo_order:
        order = vertex.order
        for edge in vertex.out_bound_edges:
            if order < edge.end_vertex.order:
                # Edge in back direction => Circle
                assert False
    for edge in complex_operation.input_vertex.out_bound_edges:
        if edge.end_vertex is complex_operation.output_vertex:
            print(type(edge), type(edge.end_vertex))
            assert isinstance(edge, MaxPool2D)
            break
    else:
        # The new edge is not there
        assert False


def test_add_edge2(basic_graph, mocker):
    # Make sure it won't break if there are multiple edges between two vertices
    complex_operation, vertex1, vertex2, vertex3, vertex4 = basic_graph
    complex_operation.sort_vertices()

    def mock(*args, **kwargs):
        if kwargs['size'] == 2:
            return [vertex1, vertex2]
        if kwargs['size'] == 1:
            assert isinstance(args[0][0], PointConv2D)
            return [MaxPool2D()]

    mocker.patch('numpy.random.choice', side_effect=mock)

    complex_operation.mutation_add_edge()
    assert len(vertex1.out_bound_edges) == 3
    to_vertex2_count = 0
    for edge in vertex1.out_bound_edges:
        if edge.end_vertex is vertex2:
            to_vertex2_count += 1
    assert to_vertex2_count == 2


def test_mutate_edge(basic_graph, mocker):
    # Make sure it won't break if there are multiple edges between two vertices
    complex_operation, vertex1, vertex2, vertex3, vertex4 = basic_graph

    edge_to_replace = MaxPool2D()
    complex_operation.input_vertex.out_bound_edges.append(edge_to_replace)
    edge_to_replace.end_vertex = complex_operation.output_vertex

    new_edge = PointConv2D((2, 3))

    complex_operation.sort_vertices()

    def mock(*args, **kwargs):
        if isinstance(args[0][0], Vertex):
            return [complex_operation.input_vertex]
        if edge_to_replace in args[0]:
            return [edge_to_replace]
        else:
            return [new_edge]

    mocker.patch('numpy.random.choice', side_effect=mock)

    before_out_edges = list(complex_operation.input_vertex.out_bound_edges)
    before_out_edges.remove(edge_to_replace)

    complex_operation.mutation_mutate_edge()
    assert edge_to_replace.end_vertex is None
    assert new_edge in complex_operation.input_vertex.out_bound_edges
    assert (len(complex_operation.input_vertex.out_bound_edges)
            == len(before_out_edges) + 1)

    # Everything before not mutated is still there
    for edge in before_out_edges:
        assert edge in complex_operation.input_vertex.out_bound_edges
