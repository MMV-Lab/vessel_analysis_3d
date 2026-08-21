import numpy as np

from vessel_analysis_3d.graph.core import GraphObj

# Reproduces the exact case a skeleton-local-radius check misses (2026-08-20,
# Simon): a vessel running roughly parallel to a volume face, clipped along
# its side rather than at a tip. Its skeleton recenters within whatever
# cross-section is actually visible and never itself touches the boundary --
# but the raw mask plainly does. GraphObj._flag_boundary_clipped_segments
# attributes boundary-touching raw-mask patches to the nearest skeleton node
# (across the whole graph) instead of relying on any single segment's own
# local radius.
PIXEL_DIMS = [1.0, 1.0, 1.0]
SHAPE = (30, 30, 30)


def _make_shim(raw, segments_dict, margin_um=3.0, radii=None):
    shim = object.__new__(GraphObj)
    shim.rawSegmentation = raw
    shim.pixelDims = PIXEL_DIMS
    shim.boundaryClipMarginUm = margin_um
    shim.segmentsDict = {0: segments_dict}
    shim.segStatsDict = {0: {key: {} for key in segments_dict}}
    radius_matrix = np.zeros(raw.shape, dtype=float)
    for node, r in (radii or {}).items():
        radius_matrix[node] = r
    shim.radiusMatrix = radius_matrix
    return shim


def test_parallel_vessel_gets_flagged_without_its_skeleton_touching_boundary():
    raw = np.zeros(SHAPE, dtype=np.uint8)
    # the visible sliver of a vessel truncated by the x=0 face -- its own
    # true center lies outside the volume, so only this strip was captured
    raw[5:25, 18:23, 0] = 1

    tube_a = [(z, 15, 15) for z in range(5, 25)]  # deep interior, no face contact
    tube_b = [(z, 20, 1) for z in range(5, 25)]  # one voxel off the x=0 face

    key_a = (tube_a[0], tube_a[-1])
    key_b = (tube_b[0], tube_b[-1])
    shim = _make_shim(raw, {key_a: tube_a, key_b: tube_b})

    GraphObj._flag_boundary_clipped_segments(shim)

    assert shim.segStatsDict[0][key_b].get("boundaryClipped") is True
    assert shim.segStatsDict[0][key_a].get("boundaryClipped") is not True


def test_small_distant_boundary_patch_does_not_misattribute_to_an_unrelated_segment():
    raw = np.zeros(SHAPE, dtype=np.uint8)
    # a tiny, isolated patch on a different face, far from any skeleton --
    # the signature of the (separate, known) "no skeleton reaches here at
    # all" case. Its closest node is much farther than a few times its own
    # size, so it must be left unattributed rather than blamed on whichever
    # segment happens to be nearest.
    raw[0, 5:7, 5:7] = 1

    tube_a = [(z, 15, 15) for z in range(5, 25)]
    tube_b = [(z, 20, 1) for z in range(5, 25)]
    key_a = (tube_a[0], tube_a[-1])
    key_b = (tube_b[0], tube_b[-1])
    shim = _make_shim(raw, {key_a: tube_a, key_b: tube_b})

    GraphObj._flag_boundary_clipped_segments(shim)

    assert shim.segStatsDict[0][key_a].get("boundaryClipped") is not True
    assert shim.segStatsDict[0][key_b].get("boundaryClipped") is not True


def test_branch_point_shared_by_two_segments_flags_both():
    raw = np.zeros(SHAPE, dtype=np.uint8)
    # small patch centered right at the branch point (not spanning the
    # whole tube like the other tests) so the nearest skeleton node to it
    # genuinely is the shared branch point, not some other point along
    # tube_b's length
    raw[4:7, 18:23, 0] = 1

    branch_point = (5, 20, 1)
    tube_b = [(z, 20, 1) for z in range(5, 25)]  # contains the branch point
    tube_c = [branch_point, (5, 21, 2), (5, 22, 3)]  # a second branch off it

    key_b = (tube_b[0], tube_b[-1])
    key_c = (tube_c[0], tube_c[-1])
    shim = _make_shim(raw, {key_b: tube_b, key_c: tube_c})

    GraphObj._flag_boundary_clipped_segments(shim)

    assert shim.segStatsDict[0][key_b].get("boundaryClipped") is True
    assert shim.segStatsDict[0][key_c].get("boundaryClipped") is True


def test_mask_within_margin_but_not_touching_exact_edge_still_flags():
    # Reproduces the specific gap Simon flagged (2026-08-20): mask-opening
    # erosion/smoothing routinely leaves a real edge-truncated vessel a few
    # voxels short of literally touching the array boundary. This blob's
    # nearest voxel to the x=0 face is at x=5 -- under the original
    # exact-voxel check it would never have been flagged at all. Kept
    # >25 units from every OTHER face too (unlike an earlier version of
    # this test), so only the x=0 proximity is actually being exercised.
    shape = (100, 100, 100)
    raw = np.zeros(shape, dtype=np.uint8)
    raw[49:52, 49:52, 5:7] = 1  # min x=5; z,y both deep interior (~50)

    tube_near = [(50 + dz, 50, 6) for dz in range(-5, 6)]  # x=6, just off x=0
    tube_far = [(50 + dz, 50, 50) for dz in range(-5, 6)]  # x=50, far from every face

    key_near = (tube_near[0], tube_near[-1])
    key_far = (tube_far[0], tube_far[-1])
    shim = _make_shim(
        raw, {key_near: tube_near, key_far: tube_far}, margin_um=25.0
    )

    GraphObj._flag_boundary_clipped_segments(shim)

    assert shim.segStatsDict[0][key_near].get("boundaryClipped") is True
    assert shim.segStatsDict[0][key_far].get("boundaryClipped") is not True


def test_many_vessels_bridged_into_one_raw_blob_are_each_flagged_independently():
    # The actual bug Simon caught on real data (2026-08-20): a prior version
    # grouped all near-boundary foreground into connected-component blobs
    # first and attributed each whole blob to its single nearest skeleton
    # node. On segmentation_v30, one such blob alone was 399,194 of 485,412
    # total near-boundary voxels (82%) -- dozens of unrelated vessels
    # fused into "one" region by thin bridges within the margin slab -- so
    # only whichever single vessel was nearest the blob's averaged-out
    # centroid got flagged; every other real vessel sharing that blob,
    # including the one Simon was pointing at, was silently skipped.
    #
    # This reproduces that shape directly: three well-separated tubes near
    # x=0, all touching one continuous raw-mask strip that spans the full
    # y range (so under the old blob-based design this would score as one
    # mega-blob). All three must still be flagged independently -- the
    # current design never groups foreground into blobs at all, so there
    # is no single "nearest" vessel to silently win at the others' expense.
    shape = (30, 60, 10)
    raw = np.zeros(shape, dtype=np.uint8)
    raw[5:25, :, 0] = 1  # one bridge touching x=0, spanning the entire y range

    tube_1 = [(z, 5, 1) for z in range(5, 25)]
    tube_2 = [(z, 30, 1) for z in range(5, 25)]
    tube_3 = [(z, 55, 1) for z in range(5, 25)]
    key_1 = (tube_1[0], tube_1[-1])
    key_2 = (tube_2[0], tube_2[-1])
    key_3 = (tube_3[0], tube_3[-1])
    shim = _make_shim(raw, {key_1: tube_1, key_2: tube_2, key_3: tube_3})

    GraphObj._flag_boundary_clipped_segments(shim)

    assert shim.segStatsDict[0][key_1].get("boundaryClipped") is True
    assert shim.segStatsDict[0][key_2].get("boundaryClipped") is True
    assert shim.segStatsDict[0][key_3].get("boundaryClipped") is True


def test_thick_vessel_with_flat_centered_skeleton_still_flags():
    # A second real bug, found immediately after fixing the first one
    # (2026-08-20, same session): even with blob grouping removed, a node
    # was only checked if ITS OWN coordinate was within the margin of a
    # face -- reintroducing a version of the very first problem this
    # check exists to solve. Confirmed on segmentation_v30: a
    # 166um-diameter vessel's skeleton sat flat at z=10-11 (~83-91um from
    # the z=0 face, past a 25um margin) for its whole length, while its
    # raw mask was solid foreground all the way to z=1 -- a thick
    # vessel's medial axis naturally centers roughly its own radius away
    # from the wall it's actually touching. The pre-filter now reaches
    # margin + the node's own local radius, not margin alone.
    shape = (60, 60, 60)
    raw = np.zeros(shape, dtype=np.uint8)
    # solid mask right at the z=0 face -- what the vessel's cross-section
    # actually looks like where it's captured, well within the thick
    # vessel's own radius of its (flat, centered) skeleton nodes below
    raw[0:5, 28:33, 28:33] = 1

    tube_thick = [(15, 30, 30), (15, 31, 30)]  # dist to z=0 face = 15
    tube_thin = [(15, 30, 40), (15, 31, 40)]  # same depth, far in x, thin

    key_thick = (tube_thick[0], tube_thick[-1])
    key_thin = (tube_thin[0], tube_thin[-1])
    radii = {}
    for node in tube_thick:
        radii[node] = 15.0  # margin(5) + radius(15) = 20 >= dist(15)
    for node in tube_thin:
        radii[node] = 1.0  # margin(5) + radius(1) = 6 < dist(15)

    shim = _make_shim(
        raw,
        {key_thick: tube_thick, key_thin: tube_thin},
        margin_um=5.0,
        radii=radii,
    )

    GraphObj._flag_boundary_clipped_segments(shim)

    assert shim.segStatsDict[0][key_thick].get("boundaryClipped") is True
    assert shim.segStatsDict[0][key_thin].get("boundaryClipped") is not True
