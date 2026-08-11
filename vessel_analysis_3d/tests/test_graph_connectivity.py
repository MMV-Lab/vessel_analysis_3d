import numpy as np

from vessel_analysis_3d.graph.networkx_from_array import get_networkx_graph_from_array


def test_diagonal_voxels_connect_when_not_bridged_by_a_face_neighbor():
    # 3D thinning routinely steps diagonally between medial-axis voxels,
    # especially with anisotropic voxel spacing, so diagonal touches must
    # create edges or the skeleton graph fragments into many disconnected
    # single-voxel components.
    arr = np.zeros((3, 3), dtype=np.uint8)
    arr[0, 0] = 1
    arr[1, 1] = 1
    arr[2, 2] = 1

    graph = get_networkx_graph_from_array(arr)

    edges = {frozenset(e) for e in graph.edges()}
    assert edges == {frozenset({(0, 0), (1, 1)}), frozenset({(1, 1), (2, 2)})}


def test_redundant_diagonal_clique_edge_is_removed():
    # (0,0)-(0,1)-(1,1) are all pairwise adjacent (2 face edges + 1 diagonal),
    # forming a 3-vertex clique. The longer, redundant diagonal edge should be
    # dropped so the graph keeps only the shortest path between the voxels.
    arr = np.zeros((2, 2), dtype=np.uint8)
    arr[0, 0] = 1
    arr[0, 1] = 1
    arr[1, 1] = 1

    graph = get_networkx_graph_from_array(arr)

    edges = {frozenset(e) for e in graph.edges()}
    assert edges == {frozenset({(0, 0), (0, 1)}), frozenset({(0, 1), (1, 1)})}
