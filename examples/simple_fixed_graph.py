import numpy as np
from tensorflow import keras

from evolution.base import IdentityOperation
from evolution.base import PointConv2D
from evolution.base import ReLU
from evolution.base import Vertex
from evolution.base import Dense
from evolution.base import Flatten
from evolution.fixed_edge import FixedEdge


class TestEdge1(FixedEdge):
    def __init__(self) -> None:
        super().__init__(name='MiddleLevel')

    def construct_new_instance(self) -> FixedEdge:
        return TestEdge1()

    def build_graph(self) -> None:
        edge1 = PointConv2D((0, 3))
        edge2 = ReLU()
        edge3 = IdentityOperation()
        vertex1 = Vertex(name='V1')

        self.input_vertex.out_bound_edges.append(edge1)
        self.input_vertex.out_bound_edges.append(edge3)
        vertex1.out_bound_edges.append(edge2)
        edge1.end_vertex = vertex1
        edge2.end_vertex = self.output_vertex
        edge3.end_vertex = self.output_vertex


class TestEdge2(FixedEdge):
    def __init__(self) -> None:
        super().__init__(name='HighestLevel')

    def construct_new_instance(self) -> FixedEdge:
        return TestEdge2()

    def build_graph(self) -> None:
        edge1 = TestEdge1()
        edge2 = edge1.deep_copy()
        edge3 = Flatten()
        edge4 = Dense(5)
        vertex1 = Vertex(name='V1')
        vertex2 = Vertex(name='V2')
        vertex3 = Vertex(name='V3')

        self.input_vertex.out_bound_edges.append(edge1)
        vertex1.out_bound_edges.append(edge2)
        edge1.end_vertex = vertex1
        edge2.end_vertex = vertex2
        vertex2.out_bound_edges.append(edge3)
        edge3.end_vertex = vertex3
        vertex3.out_bound_edges.append(edge4)
        edge4.end_vertex = self.output_vertex


edge = TestEdge2()

x = keras.Input(shape=(27, 27, 3))
out = edge.build(x)

model = keras.Model(inputs=x, outputs=out)

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

tensor_board = keras.callbacks.TensorBoard(batch_size=10, write_graph=True)

input_tensor = np.random.random(size=(30, 27, 27, 3))
output_tensor = np.random.random(size=(30, 5))

model.fit(input_tensor, output_tensor, batch_size=10, callbacks=[tensor_board],
          verbose=1)
