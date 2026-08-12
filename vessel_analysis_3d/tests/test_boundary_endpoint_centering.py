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


def test_nearby_separate_vessel_does_not_pull_the_centroid_off_target():
    """Real-data regression, tested directly against _boundary_centroid_voxel
    rather than through the full skeletonize+extension pipeline (Lee's
    algorithm's own tip-retraction behavior on a flat-cut cylinder made the
    full pipeline unreliable to construct a clean synthetic reproduction
    for in the time available -- this isolates exactly the logic that
    matters instead).

    Two circular cross-sections on the same boundary plane, close enough
    together that their raw masks are CONNECTED there (as scipy.ndimage.label
    would report), but real, physically distinct vessels. On real data, two
    endpoints ~21um apart shared one 1101-voxel connected region on their
    boundary plane and both got redirected to the same, meaningless centroid,
    landing 54-85um off -- toward each other's territory. The fix restricts
    the centroid search to a local physical window around each seed, sized
    off its own local radius, so the other vessel's pixels never contribute
    even though the two masks are connected on this specific plane.
    """
    r = 15
    gap = 10  # edge-to-edge distance between the two circles -- comfortably
              # beyond search_radius_um = max(10, 1.5*r) - r = 12.5 from
              # either circle's own edge, so a correctly-local search must
              # not reach the other one
    y_a = 20
    y_b = y_a + 2 * r + gap
    shape = (10, y_b + r + 5, 5)

    zz, yy, xx = np.indices(shape)
    circle_a = ((yy - y_a) ** 2 + (zz - 5) ** 2) <= r**2
    circle_b = ((yy - y_b) ** 2 + (zz - 5) ** 2) <= r**2
    # connect the two circles' near edges only on this one boundary plane,
    # simulating the real case where two otherwise-separate vessels' raw
    # masks happen to touch right at a boundary cut
    bridge = (yy >= y_a + r - 1) & (yy <= y_b - r + 1) & (zz >= 4) & (zz <= 6)
    plane_mask = (circle_a | circle_b | bridge)
    raw_segmentation = np.broadcast_to(plane_mask, shape).astype(np.uint8).copy()

    gh = GraphObj.__new__(GraphObj)
    gh.pixelDims = PIXEL_DIMS

    for seed_y, true_center_y in [(y_a, y_a), (y_b, y_b)]:
        seed = (5, seed_y, 4)
        exit_voxel = (5, seed_y, 5)  # one step further in x -- out of array bounds
        target = gh._boundary_centroid_voxel(raw_segmentation, seed, 2, exit_voxel, float(r))
        assert target is not None
        offset = abs(target[1] - true_center_y)
        assert offset <= 1.5, (
            f"seed at y={seed_y} redirected to {target}, {offset:.1f} voxels from its own "
            f"true center (y={true_center_y}) -- likely pulled toward the other vessel"
        )
        midpoint = (y_a + y_b) / 2
        assert abs(target[1] - midpoint) > 1.5, (
            f"seed at y={seed_y} redirected to {target}, suspiciously close to the midpoint "
            f"between the two vessels ({midpoint}) -- centroid may be merging them"
        )
