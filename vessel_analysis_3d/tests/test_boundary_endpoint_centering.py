import numpy as np
from skimage.morphology import skeletonize

from vessel_analysis_3d.graph.core import GraphObj
from vessel_analysis_3d.graph.networkx_from_array import get_networkx_graph_from_array

# A tube that runs straight, then bends through a moderate arc, cut by the
# imaged volume's boundary partway through the arc -- close enough to the
# bend that the endpoint-extension tangent estimate (backward lookback over
# the last few skeleton nodes) is a blend of the pre-bend and post-bend
# directions, and so does not point at the true cross-sectional centroid.
#
# Real-data motivation (2026-08-11/12): Simon flagged a real endpoint sitting
# at the edge of a border-cut cross-section instead of centered on it,
# measured 84.3um off the true cut-face centroid. GraphObj._boundary_centroid_voxel
# fixes this by looking at the raw mask directly on the boundary plane and
# redirecting the walk's final approach there, instead of trusting the
# tangent estimate all the way to the edge.
PIXEL_DIMS = [1.0, 1.0, 1.0]
R = 5           # tube radius
RC = 20         # bend radius (centerline curvature) -- gentle enough that
                # the raw skeleton endpoint still reaches the true boundary
Z0 = 20
STRAIGHT_LEN = 30
N_ARC = 60
Y_AXIS = 10     # the straight segment's fixed y
SHAPE_Z, SHAPE_X = 40, 70

PARAMS = dict(
    pixelDimensions=PIXEL_DIMS,
    pruningScale=1.5,
    lengthLimit=3,
    diaScale=2,
    branchingThreshold=0.25,
    minBranchLengthUm=1.0,
    minBranchRadiusUm=1.0,
    minJunctionRadiusUm=1.0,
    maskOpeningRadiusUm=0.0,
    clusterCollapseScale=3.0,
    maskTipRecoverySteps=1,
    removeEndPtsFromSmallFilaments=False,
)


def _centerline():
    pts = [(x, Y_AXIS) for x in np.linspace(0, STRAIGHT_LEN, 60)]
    cx, cy = STRAIGHT_LEN, Y_AXIS + RC
    for t in np.linspace(-np.pi / 2, 0, N_ARC):
        pts.append((cx + RC * np.cos(t), cy + RC * np.sin(t)))
    return np.array(pts)


CENTERLINE = _centerline()


def _build_curved_tube(y_cut):
    # the array itself ends exactly at y_cut -- a real array-boundary cut,
    # not a mask that merely stops early inside a larger array
    shape = (SHAPE_Z, y_cut + 1, SHAPE_X)
    zz, yy, xx = np.indices(shape)
    dist = np.full(shape, np.inf)
    for cx, cy in CENTERLINE:
        d = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2 + (zz - Z0) ** 2)
        dist = np.minimum(dist, d)
    return (dist <= R).astype(np.uint8)


def _true_centroid_at(y_cut):
    """Ground truth (z, x) centroid of the tube's true cross-section at
    y=y_cut, computed directly from the geometry, independent of the code
    under test."""
    zz, xx = np.indices((SHAPE_Z, SHAPE_X))
    dist = np.full((SHAPE_Z, SHAPE_X), np.inf)
    for cx, cy in CENTERLINE:
        d = np.sqrt((xx - cx) ** 2 + (y_cut - cy) ** 2 + (zz - Z0) ** 2)
        dist = np.minimum(dist, d)
    mask = dist <= R
    zs, xs = np.nonzero(mask)
    return np.array([zs.mean(), xs.mean()])


def _extended_tip_at(y_cut, redirect_enabled):
    seg = _build_curved_tube(y_cut)
    skl = skeletonize(seg > 0, method="lee").astype(np.uint8)
    skl[skl > 0] = 1
    graph = get_networkx_graph_from_array(skl)

    orig = GraphObj._boundary_centroid_voxel
    if not redirect_enabled:
        GraphObj._boundary_centroid_voxel = lambda self, *a, **k: None
    try:
        gh = GraphObj(seg, skl, graph, rawSegmentation=seg, **PARAMS)
    finally:
        GraphObj._boundary_centroid_voxel = orig

    endpoints = [n for n, d in gh.networkxGraph.degree() if d == 1]
    tips = [n for n in endpoints if n[1] == y_cut]
    assert len(tips) == 1, f"expected exactly one tip at y={y_cut}, got {tips}"
    return tips[0]


def test_endpoint_lands_on_true_centroid_at_a_bent_boundary_cut():
    y_cut = 32
    true_c = _true_centroid_at(y_cut)
    tip = _extended_tip_at(y_cut, redirect_enabled=True)
    offset = np.linalg.norm(np.array([tip[0], tip[2]], dtype=float) - true_c)
    assert offset < 0.5, f"endpoint {tip} is {offset:.2f} voxels off the true centroid {true_c}"


def test_without_the_redirect_the_endpoint_is_measurably_off_center():
    """Negative control: confirms the phantom and cut position actually
    exercise the bug being guarded against, not just a geometry that lands
    on-center regardless."""
    y_cut = 32
    true_c = _true_centroid_at(y_cut)
    tip = _extended_tip_at(y_cut, redirect_enabled=False)
    offset = np.linalg.norm(np.array([tip[0], tip[2]], dtype=float) - true_c)
    assert offset >= 0.5, (
        f"expected the un-redirected tangent walk to miss center at this cut, "
        f"but it landed within {offset:.2f} voxels -- phantom no longer exercises the bug"
    )
