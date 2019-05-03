from evolution.encoding.base import IdentityOperation
from evolution.encoding.base import MaxPool2D
from evolution.encoding.base import Vertex
from evolution.encoding.fixed_edge import FixedEdge
from evolution.encoding.mutable_edge import MutableEdge
from evolution.evolve.mutation_strategy import MutateOneLayer


class Layer2(FixedEdge):

    def construct_new_instance(self) -> FixedEdge:
        return Layer2()

    def build_graph(self) -> None:
        vertex = Vertex()
        edge1 = MaxPool2D()
        edge2 = IdentityOperation()
        self.input_vertex.add_edge(edge1, vertex)
        vertex.add_edge(edge2, self.output_vertex)


class TrackMutation(MutableEdge):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.mutated = False

    def mutate(self) -> bool:
        self.mutated = True
        return super().mutate()


def test_mutation_one_layer_fail():
    mutate = MutateOneLayer()

    edge = Layer2()

    assert not mutate(edge)


def test_mutation_success():
    mutate = MutateOneLayer()

    edge = Layer2()
    layer3 = TrackMutation(available_operations=(edge,))
    layer4 = TrackMutation(available_operations=(layer3,))

    layer4.mutation_add_vertex()

    layer4.invalidate_layer_count()

    assert mutate(layer4)
