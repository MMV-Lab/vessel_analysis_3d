import numpy as np
from scipy.ndimage import distance_transform_edt

from vessel_analysis_3d.graph.core import GraphObj
from vessel_analysis_3d.graph.networkx_from_array import get_networkx_graph_from_array

# A thick torus: its skeleton (after Lee thinning) traces a single large
# circle -- one big biconnected component (a genuine anatomical loop, e.g. an
# anastomosis), not a small thinning artifact. The tube is thick enough
# (R_TUBE) that the OLD _collapse_dense_clusters check
# (cluster_diameter_um >= clusterCollapseScale * max_radius) is satisfied,
# because the ring's geometric diameter (~2*R_RING) stays below
# clusterCollapseScale * R_TUBE, even though the loop actually has dozens of
# skeleton nodes. This is exactly the bug found on real data: a real full
# volume run produced a NetworkX component of 1698 nodes spanning 8 distinct,
# spatially disjoint scipy.ndimage.label regions, because a large genuine
# cycle got collapsed (and, cascading, chained further collapses) purely
# because a single node in the cluster happened to have a large local EDT
# radius. See GraphObj._MAX_COLLAPSE_CLUSTER_NODES.
R_TUBE = 12
R_RING = 14  # 2*R_RING=28 < clusterCollapseScale(3.0)*R_TUBE(12)=36 -> would pass the old check
PIXEL_DIMS = [1.0, 1.0, 1.0]
PAD = 6
SHAPE = (2 * R_TUBE + PAD, 2 * (R_RING + R_TUBE) + PAD, 2 * (R_RING + R_TUBE) + PAD)
CENTER = (SHAPE[0] // 2, SHAPE[1] // 2, SHAPE[2] // 2)

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


def _build_torus():
    zz, yy, xx = np.indices(SHAPE)
    z = zz - CENTER[0]
    y = yy - CENTER[1]
    x = xx - CENTER[2]
    radial = np.sqrt(y**2 + x**2) - R_RING
    dist_to_core_circle = np.sqrt(radial**2 + z**2)
    torus = (dist_to_core_circle <= R_TUBE).astype(np.uint8)
    return torus


def _build_graph_obj(seg, **overrides):
    from skimage.morphology import skeletonize

    skl = skeletonize(seg > 0, method="lee").astype(np.uint8)
    skl[skl > 0] = 1
    graph = get_networkx_graph_from_array(skl)
    params = dict(PARAMS)
    params.update(overrides)
    return GraphObj(seg, skl, graph, rawSegmentation=seg, **params)


def test_large_genuine_loop_is_not_collapsed():
    seg = _build_torus()
    gh = _build_graph_obj(seg)

    # a genuine anatomical loop must survive as a real cycle with many nodes,
    # not get erased into a single representative point
    assert gh.networkxGraph.number_of_nodes() > GraphObj._MAX_COLLAPSE_CLUSTER_NODES
    assert gh.infoDict["clusterCollapse"] == 0


def test_without_the_node_cap_the_loop_would_be_collapsed():
    """Negative control: confirms the phantom actually exercises the bug
    being guarded against, not just a geometry that happens to survive
    regardless."""
    seg = _build_torus()
    orig_cap = GraphObj._MAX_COLLAPSE_CLUSTER_NODES
    GraphObj._MAX_COLLAPSE_CLUSTER_NODES = 10**9
    try:
        gh = _build_graph_obj(seg)
    finally:
        GraphObj._MAX_COLLAPSE_CLUSTER_NODES = orig_cap

    assert gh.infoDict["clusterCollapse"] >= 1
    assert gh.networkxGraph.number_of_nodes() <= GraphObj._MAX_COLLAPSE_CLUSTER_NODES
