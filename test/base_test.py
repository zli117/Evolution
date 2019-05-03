from typing import cast

from evolution.encoding.base import PointConv2D


def test_point_conv2d():
    edge = PointConv2D((0, 100))
    edge_copy = cast(type(edge), edge.deep_copy())

    assert edge.out_channel_range == edge_copy.out_channel_range
