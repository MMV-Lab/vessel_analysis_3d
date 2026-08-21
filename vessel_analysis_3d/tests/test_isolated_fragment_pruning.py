import networkx as nx
import numpy as np
from skimage.morphology import skeletonize

from vessel_analysis_3d.graph.core import GraphObj
from vessel_analysis_3d.graph.networkx_from_array import get_networkx_graph_from_array

# A standalone, straight tube with no branching at all -- the topology class
# _prune's branch-point-triggered noise check never evaluates, since its
# walk from an endpoint only tests thinness/shortness once it hits a real
# branch point (degree > 2). Confirmed on real data (segmentation_v30,
# 2026-08-19): an isolated 47um-long, 17.68um-diameter fragment survived
# untouched at roughly a third of the configured 50um vessel-diameter floor.
PIXEL_DIMS = [1.0, 1.0, 1.0]
LENGTH = 60
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


def _build_straight_tube(radius, length=LENGTH):
    pad = radius + 4
    shape = (2 * pad, 2 * pad, length + 2 * pad)
    zz, yy, xx = np.indices(shape)
    center_y, center_z = shape[1] // 2, shape[0] // 2
    in_tube = (
        (xx >= pad)
        & (xx < pad + length)
        & (np.sqrt((yy - center_y) ** 2 + (zz - center_z) ** 2) <= radius)
    )
    return in_tube.astype(np.uint8)


def _build_graph_obj(seg):
    skl = skeletonize(seg > 0, method="lee").astype(np.uint8)
    skl[skl > 0] = 1
    graph = get_networkx_graph_from_array(skl)
    return GraphObj(seg, skl, graph, rawSegmentation=seg, **PARAMS)


def test_isolated_thin_fragment_below_radius_floor_is_removed():
    """A standalone tube with no branch point, radius well below
    minBranchRadiusUm, must be pruned entirely -- not just shifted or left
    as an orphaned endpoint."""
    seg = _build_straight_tube(radius=8)  # diameter 16um, well under the 50um floor
    gh = _build_graph_obj(seg)
    assert gh.networkxGraph.number_of_nodes() == 0


def test_isolated_thick_fragment_above_radius_floor_survives():
    """A standalone tube with a real, plausible vessel radius must survive
    untouched -- the new check must not remove every branch-point-free
    fragment indiscriminately, only ones failing the physical-plausibility
    floor."""
    seg = _build_straight_tube(radius=30)  # diameter 60um, above the 50um floor
    gh = _build_graph_obj(seg)
    assert gh.networkxGraph.number_of_nodes() > 0
    assert nx.is_connected(gh.networkxGraph)
