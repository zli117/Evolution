from evolution.base import IdentityOperation


def test_cycle_detection(basic_graph):
    complex_operation, vertex1, vertex2, vertex3, vertex4 = basic_graph
    complex_operation.sort_vertices()

    try:
        complex_operation.sort_vertices()
    except RuntimeError:
        assert False

    edge = IdentityOperation()

    vertex2.out_bound_edges.append(edge)
    edge.end_vertex = vertex1

    try:
        complex_operation.sort_vertices()
        assert False
    except RuntimeError:
        pass


def test_output_reachable_check(basic_graph_no_v12):
    complex_operation, vertex1, vertex2, vertex3, vertex4 = basic_graph_no_v12
    complex_operation.sort_vertices()

    assert complex_operation.check_output_reachable()

    complex_operation.input_vertex.out_bound_edges.clear()

    assert not complex_operation.check_output_reachable()
