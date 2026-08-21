import networkx as nx
import numpy as np

from vessel_analysis_3d.graph.core import GraphObj
from vessel_analysis_3d.graph.networkx_from_array import get_networkx_graph_from_array

# A thick torus with two thin arms attached, mimicking a real, large, busy
# junction (e.g. an anastomosis) that also connects onward to the rest of the
# vessel tree -- the actual real-data scenario (2026-08-11): a 1013-node,
# 199um junction with outside attachments got collapsed to a single isolated
# point, leaving a ~20um+ visual gap in the exported skeleton even though the
# graph stayed "connected" via an artificial long-range edge added on top.
# _collapse_dense_clusters should now keep a real, physically continuous,
# cycle-free path from the junction's center out to each attachment, instead
# of either collapsing to one point (loses continuity) or leaving the
# internal cycles untouched (reintroduces the branch-point explosion the
# original collapse-to-one-point design was built to prevent).
R_TUBE = 12
R_RING = 14  # 2*R_RING=28 < clusterCollapseScale(3.0)*R_TUBE(12)=36 -> collapse-eligible
ARM_RADIUS = 4
ARM_LENGTH = 40
PIXEL_DIMS = [1.0, 1.0, 1.0]
PAD = 6
RING_SPAN = 2 * (R_RING + R_TUBE) + PAD
SHAPE = (2 * R_TUBE + PAD, RING_SPAN, RING_SPAN + ARM_LENGTH)
CENTER = (SHAPE[0] // 2, RING_SPAN // 2, RING_SPAN // 2)

PARAMS = dict(
    pixelDimensions=PIXEL_DIMS,
    pruningScale=1.5,
    lengthLimit=3,
    diaScale=2,
    branchingThreshold=0.25,
    minBranchLengthUm=20.0,
    minBranchRadiusUm=25.0,
    minJunctionRadiusUm=25.0,
    maskOpeningRadiusUm=0.0,
    clusterCollapseScale=3.0,
    maskTipRecoverySteps=1,
    removeEndPtsFromSmallFilaments=False,
)


def _build_torus_with_arm():
    zz, yy, xx = np.indices(SHAPE)
    z = zz - CENTER[0]
    y = yy - CENTER[1]
    x = xx - CENTER[2]
    radial = np.sqrt(y**2 + x**2) - R_RING
    dist_to_core_circle = np.sqrt(radial**2 + z**2)
    torus = dist_to_core_circle <= R_TUBE

    # a thin arm sticking straight out in +x from the ring's rightmost edge
    arm_x0 = CENTER[2] + R_RING + R_TUBE
    arm = (
        (xx >= arm_x0)
        & (np.sqrt((yy - CENTER[1]) ** 2 + (zz - CENTER[0]) ** 2) <= ARM_RADIUS)
    )
    return (torus | arm).astype(np.uint8)


def _build_graph_obj(seg, **overrides):
    from skimage.morphology import skeletonize

    skl = skeletonize(seg > 0, method="lee").astype(np.uint8)
    skl[skl > 0] = 1
    graph = get_networkx_graph_from_array(skl)
    params = dict(PARAMS)
    params.update(overrides)
    return GraphObj(seg, skl, graph, rawSegmentation=seg, **params)


def test_large_junction_with_attachment_stays_connected_and_cycle_free():
    seg = _build_torus_with_arm()
    # This fixture's synthetic radii (R_TUBE=12, ARM_RADIUS=4) are far below
    # PARAMS' production-scale minBranchRadiusUm=25.0 -- fine for every other
    # test here, but after collapse this specific ring+single-arm topology
    # legitimately has no remaining branch point (the ring's shortest-path
    # tree to its one attachment point never itself branches), so
    # _prune's absolute-radius floor for branch-point-free fragments would
    # otherwise remove this whole physically-continuous structure. That
    # floor is intentionally not what this test is checking -- override it
    # off so this test only exercises _collapse_dense_clusters' connectivity
    # preservation, its actual subject.
    gh = _build_graph_obj(seg, minBranchRadiusUm=0.0)

    assert gh.infoDict["clusterCollapse"] >= 1
    # a real junction with an attached arm must keep a physically continuous
    # path, not collapse to a single isolated point
    assert gh.networkxGraph.number_of_nodes() > 5
    # cycle-free: no residual biconnected components of size >= 3 remain
    remaining_cycles = [c for c in nx.biconnected_components(gh.networkxGraph) if len(c) >= 3]
    assert remaining_cycles == []
    assert nx.is_connected(gh.networkxGraph)


def test_simplified_cluster_has_no_edges_beyond_original_adjacency():
    """Every edge left after _collapse_dense_clusters alone must correspond
    to a real, pre-existing edge from the raw skeleton graph -- the new
    design never calls add_edge, so it cannot introduce a connection between
    nodes that were never actually adjacent (the mechanism the 2026-08-11
    investigation traced the original collapse-to-one-point design's failure
    mode to). Checked right after collapse, before _prune / endpoint
    extension run (those legitimately add their own edges later)."""
    from skimage.morphology import skeletonize

    seg = _build_torus_with_arm()
    skl = skeletonize(seg > 0, method="lee").astype(np.uint8)
    skl[skl > 0] = 1
    raw_graph = get_networkx_graph_from_array(skl)
    raw_edges = set(frozenset(e) for e in raw_graph.edges())

    orig_prune = GraphObj._prune
    orig_extend = GraphObj._extend_endpoints_to_raw_boundary
    GraphObj._prune = lambda self, radiusMat: None
    GraphObj._extend_endpoints_to_raw_boundary = lambda self, raw_segmentation: None
    try:
        gh = _build_graph_obj(seg)
    finally:
        GraphObj._prune = orig_prune
        GraphObj._extend_endpoints_to_raw_boundary = orig_extend

    final_edges = set(frozenset(e) for e in gh.networkxGraph.edges())
    assert final_edges <= raw_edges


def test_isolated_loop_with_no_attachments_still_collapses_to_one_point():
    """A closed loop with nothing else attached has no attachment points, so
    there's nothing worth keeping a path to -- it correctly still collapses
    to a single representative, same as before."""
    zz, yy, xx = np.indices((2 * R_TUBE + PAD, RING_SPAN, RING_SPAN))
    center = (zz.shape[0] // 2, RING_SPAN // 2, RING_SPAN // 2)
    z = zz - center[0]
    y = yy - center[1]
    x = xx - center[2]
    radial = np.sqrt(y**2 + x**2) - R_RING
    dist_to_core_circle = np.sqrt(radial**2 + z**2)
    seg = (dist_to_core_circle <= R_TUBE).astype(np.uint8)

    gh = _build_graph_obj(seg)
    assert gh.infoDict["clusterCollapse"] >= 1
    assert gh.networkxGraph.number_of_nodes() == 1


def test_dense_blob_with_chord_edges_terminates_and_stays_cycle_free():
    """A small solid blob (not a clean torus) with two thin arms attached is
    much more likely to skeletonize into a tangle with "chord" edges -- an
    original edge directly connecting two nodes that also end up on the
    kept shortest-path tree via a longer route. If those chords aren't also
    removed, the retained set stays cyclic, biconnected_components() finds
    it again next pass, and an ~1000-node real-data cluster's O(n^2)
    diameter check gets recomputed every pass without making progress (this
    is exactly what caused an hour-long real-data hang that should have
    taken ~6 minutes). This must complete quickly and end cycle-free."""
    import time

    from skimage.morphology import skeletonize

    blob_radius = 10
    arm_radius = 3
    arm_length = 30
    shape = (2 * blob_radius + 4, 2 * blob_radius + 4, 2 * (blob_radius + arm_length) + 4)
    center = (shape[0] // 2, shape[1] // 2, shape[2] // 2)

    zz, yy, xx = np.indices(shape)
    z = zz - center[0]
    y = yy - center[1]
    x = xx - center[2]
    blob = (z**2 + y**2 + x**2) <= blob_radius**2

    arm1 = (xx <= center[2] - blob_radius) & (
        np.sqrt((yy - center[1]) ** 2 + (zz - center[0]) ** 2) <= arm_radius
    )
    arm2 = (xx >= center[2] + blob_radius) & (
        np.sqrt((yy - center[1]) ** 2 + (zz - center[0]) ** 2) <= arm_radius
    )
    seg = (blob | arm1 | arm2).astype(np.uint8)

    skl = skeletonize(seg > 0, method="lee").astype(np.uint8)
    skl[skl > 0] = 1
    graph = get_networkx_graph_from_array(skl)
    params = dict(PARAMS)
    params["clusterCollapseScale"] = 5.0  # generous, to make sure the blob is eligible
    # Same reasoning as test_large_junction_with_attachment_...: these
    # synthetic arm/blob radii are far below the production-scale
    # minBranchRadiusUm=25.0, and this test is specifically about
    # cycle-freedom/connectivity after collapse, not the separate absolute-
    # radius floor for branch-point-free fragments.
    params["minBranchRadiusUm"] = 0.0

    start = time.time()
    gh = GraphObj(seg, skl, graph, rawSegmentation=seg, **params)
    elapsed = time.time() - start

    assert elapsed < 10.0, f"took {elapsed:.1f}s -- likely stuck reprocessing a cyclic cluster"
    remaining_cycles = [c for c in nx.biconnected_components(gh.networkxGraph) if len(c) >= 3]
    assert remaining_cycles == []
    assert nx.is_connected(gh.networkxGraph)
