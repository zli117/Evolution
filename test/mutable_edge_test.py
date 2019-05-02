from typing import cast

import tensorflow as tf

from evolution.base import Edge
from evolution.base import IdentityOperation
from evolution.base import MaxPool2D
from evolution.base import PointConv2D
from evolution.base import Vertex
from evolution.mutable_edge import MutableEdge


def test_complex_op_creation():
    complex_operation = MutableEdge((PointConv2D((1, 4)),))
    assert len(complex_operation.available_operations) == 1
    assert len(complex_operation.vertices_topo_order) == 2
    assert (complex_operation.vertices_topo_order[0]
            is complex_operation.output_vertex)
    assert (complex_operation.vertices_topo_order[1]
            is complex_operation.input_vertex)


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


class MockEdge(Edge):

    def __init__(self, mutated: bool = True):
        super().__init__()
        self.mutated = mutated
        self.deep_copy_count = 0

    def mutate(self) -> bool:
        return self.mutated

    def build(self, x: tf.Tensor) -> tf.Tensor:
        return x

    def invalidate_layer_count(self) -> None:
        pass

    @property
    def layers_below(self) -> int:
        return 1

    def deep_copy(self) -> Edge:
        self.deep_copy_count += 1
        return self


def test_mutate_edge(basic_graph, mocker):
    complex_operation, vertex1, vertex2, vertex3, vertex4 = basic_graph

    edge_to_replace = MaxPool2D()
    complex_operation.input_vertex.out_bound_edges.append(edge_to_replace)
    edge_to_replace.end_vertex = complex_operation.output_vertex

    new_edge = MockEdge()

    complex_operation.sort_vertices()

    def mock(*args, **_):
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
    assert new_edge.deep_copy_count == 1
    assert (len(complex_operation.input_vertex.out_bound_edges)
            == len(before_out_edges) + 1)

    # Everything before not mutated is still there
    for edge in before_out_edges:
        assert edge in complex_operation.input_vertex.out_bound_edges


def test_remove_edge_fail1():
    complex_operation = MutableEdge((PointConv2D((1, 4)), MaxPool2D()))
    assert not complex_operation.mutation_remove_edge()


def test_remove_edge_fail2():
    complex_operation = MutableEdge((PointConv2D((1, 4)), MaxPool2D()))
    edge1 = IdentityOperation()
    edge2 = IdentityOperation()
    complex_operation.input_vertex.out_bound_edges.clear()
    complex_operation.input_vertex.out_bound_edges.append(edge1)
    middle_vertex = Vertex()
    complex_operation.vertices_topo_order.append(middle_vertex)
    edge1.end_vertex = middle_vertex
    middle_vertex.out_bound_edges.append(edge2)
    edge2.end_vertex = complex_operation.output_vertex

    assert not complex_operation.mutation_remove_edge()


def test_remove_edge_success():
    complex_operation = MutableEdge((PointConv2D((1, 4)), MaxPool2D()))
    edge1 = IdentityOperation()
    edge2 = IdentityOperation()
    complex_operation.input_vertex.out_bound_edges.clear()
    complex_operation.input_vertex.out_bound_edges.append(edge1)
    middle_vertex = Vertex()
    complex_operation.vertices_topo_order.append(middle_vertex)
    edge1.end_vertex = middle_vertex
    middle_vertex.out_bound_edges.append(edge2)
    edge2.end_vertex = complex_operation.output_vertex

    # Edge from input to output. So now we can remove one edge
    edge3 = IdentityOperation()
    complex_operation.input_vertex.out_bound_edges.append(edge3)
    edge3.end_vertex = complex_operation.output_vertex

    assert complex_operation.mutation_remove_edge()
    assert len(complex_operation.input_vertex.out_bound_edges) == 1


def test_mutation_add_node(basic_graph_no_v12, mocker):
    complex_operation, vertex1, vertex2, vertex3, vertex4 = basic_graph_no_v12

    complex_operation.sort_vertices()
    v1_order = vertex1.order
    v2_order = vertex2.order

    # Artificially make v2 lower order than v1 since they are parallel. The
    # order could be arbitrary

    vertex2.order = min(v1_order, v2_order)
    vertex1.order = max(v1_order, v2_order)

    edge1 = MaxPool2D()
    edge2 = MockEdge()

    def mock(*args, **_):
        if isinstance(args[0][0], Vertex):
            return [vertex1, vertex2]
        if isinstance(args[0][0], Edge):
            return [edge1, edge2]

    mocker.patch('numpy.random.choice', side_effect=mock)

    complex_operation.mutation_add_vertex()
    assert edge2.end_vertex is vertex2
    assert edge2.deep_copy_count == 1
    assert vertex2.order < vertex1.order


def test_mutation_add_node_max_vertex():
    complex_operation = MutableEdge((PointConv2D((1, 4)),), max_vertices=2)
    assert not complex_operation.mutation_add_vertex()

    complex_operation = MutableEdge((PointConv2D((1, 4)),), max_vertices=3)
    assert complex_operation.mutation_add_vertex()
    assert len(complex_operation.vertices_topo_order) == 3
    assert not complex_operation.mutation_add_vertex()


def test_remove_node_success(basic_graph_no_v12, mocker):
    complex_operation, vertex1, vertex2, vertex3, vertex4 = basic_graph_no_v12

    vertex = Vertex()
    edge1 = IdentityOperation()
    edge2 = IdentityOperation()

    vertex1.out_bound_edges.append(edge1)
    edge1.end_vertex = vertex
    vertex.out_bound_edges.append(edge2)
    edge2.end_vertex = vertex2

    complex_operation.sort_vertices()

    mocker.patch('numpy.random.permutation', return_value=[vertex, vertex2])

    assert vertex in complex_operation.vertices_topo_order

    assert complex_operation.mutation_remove_vertex()

    assert vertex2 in complex_operation.vertices_topo_order
    assert vertex not in complex_operation.vertices_topo_order

    assert len(vertex1.out_bound_edges) == 1


def test_remove_node_fail():
    complex_operation = MutableEdge((PointConv2D((1, 4)),))
    assert not complex_operation.mutation_remove_vertex()

    complex_operation.input_vertex.out_bound_edges.clear()

    vertex1 = Vertex()
    vertex2 = Vertex()
    edge1 = IdentityOperation()
    edge2 = IdentityOperation()
    edge3 = IdentityOperation()

    complex_operation.input_vertex.out_bound_edges.append(edge1)
    edge1.end_vertex = vertex1
    vertex1.out_bound_edges.append(edge2)
    edge2.end_vertex = vertex2
    vertex2.out_bound_edges.append(edge3)
    edge3.end_vertex = complex_operation.output_vertex

    complex_operation.sort_vertices()
    assert len(complex_operation.vertices_topo_order) == 4
    assert not complex_operation.mutation_remove_vertex()


def test_max_vertices():
    try:
        MutableEdge((PointConv2D((1, 4)),), max_vertices=1)
        assert False
    except RuntimeError:
        pass

    try:
        MutableEdge((PointConv2D((1, 4)),), max_vertices=2)
    except RuntimeError:
        assert False


def test_deep_copy(basic_graph):
    complex_operation, vertex1, vertex2, vertex3, vertex4 = basic_graph
    higher_level = MutableEdge((complex_operation,),
                               initialize_with_identity=False,
                               max_vertices=10)

    higher_level_copy = cast(MutableEdge, higher_level.deep_copy())
    assert higher_level_copy.max_vertices == 10

    complex_edge = cast(MutableEdge,
                        higher_level.input_vertex.out_bound_edges[0])
    complex_edge_copy = cast(MutableEdge,
                             higher_level_copy.input_vertex.out_bound_edges[0])
    assert (len(complex_edge.vertices_topo_order)
            == len(complex_edge_copy.vertices_topo_order))
    for i, vertex in enumerate(complex_edge.vertices_topo_order):
        for j, edge in enumerate(vertex.out_bound_edges):
            copy_vertex = complex_edge_copy.vertices_topo_order[i]
            copy_edge = copy_vertex.out_bound_edges[j]
            assert copy_edge.end_vertex.order == edge.end_vertex.order
            assert copy_vertex is not vertex
            assert copy_edge is not edge

    assert (len(higher_level_copy.available_operations)
            == len(higher_level.available_operations))

    for i in range(len(higher_level_copy.available_operations)):
        assert (higher_level_copy.available_operations[i]
                is not higher_level.available_operations[i])
