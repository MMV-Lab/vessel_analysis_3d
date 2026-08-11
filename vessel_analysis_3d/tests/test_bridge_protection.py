import numpy as np
import pytest
from scipy.ndimage import binary_closing, binary_dilation, binary_erosion
from skimage.morphology import skeletonize

from vessel_analysis_3d.processing_pipeline import (
    Pipeline3D,
    _anisotropic_ellipsoid_footprint,
)
from vessel_analysis_3d.graph.filament import Filament

# Two T-junction "bars" (each with two ~70um arms of its own, radius 8), joined
# by a thinner, shorter connecting bridge (radius 4, length ~33um) -- an "H"
# shape. Each bar's own branch point has substantial structure on both sides
# of the bridge (two long arms), so the bridge is the sole connection holding
# two real sub-trees together, not a dead-end spur.
#
# This phantom exists because a real full-volume run produced exactly this
# topology by accident: a short segment connecting two branch points got
# deleted by the "segment length <= lengthLimit" rule (which has no concept
# of what it's connecting), silently splitting one continuous, clearly-wide
# vessel arm into an orphaned fragment that never reached the tissue's true
# edge. See Filament._is_bridge_between_substantial_structure.
SHAPE = (170, 100, 75)
RADIUS = 8
BRIDGE_RADIUS = 4
Y0 = 50
BAR_A_X0 = 15
BAR_B_X0 = 50
Z_RANGE = (10, 150)
BRIDGE_Z = 80
PIXEL_DIMS = [1.0, 1.0, 1.0]

PARAMS = {
    "pixelDimensions": PIXEL_DIMS,
    "pruningScale": 1.5,
    # deliberately raised from the production default (3um) to reliably
    # flag this phantom's ~33um bridge as "short": the fix under test
    # doesn't care which criterion flagged a segment for removal, only
    # whether removing it would sever two otherwise-independent structures,
    # so this is a clean, reliable way to exercise it on a known geometry.
    "lengthLimit": 40,
    "diaScale": 2,
    "branchingThreshold": 0.25,
    "minBranchLengthUm": 20.0,
    "minBranchRadiusUm": 25.0,
    "minJunctionRadiusUm": 25.0,
    "maskOpeningRadiusUm": 3.0,
    "clusterCollapseScale": 3.0,
    "maskTipRecoverySteps": 1,
    "removeEndPtsFromSmallFilaments": False,
}


def _build_h_phantom():
    zz, yy, xx = np.indices(SHAPE)
    bar_a = (
        (zz >= Z_RANGE[0]) & (zz <= Z_RANGE[1])
        & ((yy - Y0) ** 2 + (xx - BAR_A_X0) ** 2 <= RADIUS**2)
    )
    bar_b = (
        (zz >= Z_RANGE[0]) & (zz <= Z_RANGE[1])
        & ((yy - Y0) ** 2 + (xx - BAR_B_X0) ** 2 <= RADIUS**2)
    )
    bridge = (
        (xx >= BAR_A_X0) & (xx <= BAR_B_X0)
        & ((yy - Y0) ** 2 + (zz - BRIDGE_Z) ** 2 <= BRIDGE_RADIUS**2)
    )
    seg = np.zeros(SHAPE, dtype=np.uint8)
    seg[bar_a | bar_b | bridge] = 1
    return seg


def _run_pipeline(seg_raw):
    footprint = _anisotropic_ellipsoid_footprint(3.0, PIXEL_DIMS)
    seg_mask = seg_raw > 0
    eroded = binary_erosion(seg_mask, structure=footprint)
    opened = binary_dilation(eroded, structure=footprint, iterations=1, mask=seg_mask)
    seg = binary_closing(opened, structure=footprint).astype(np.uint8)

    skl = skeletonize(seg > 0, method="lee").astype(np.uint8)
    skl[skl > 0] = 1

    pipe = Pipeline3D()
    return pipe.process_one_file(
        seg, skl, dict(PARAMS), save_path=None, basename="bridge", raw_segmentation=seg_raw
    )


def test_short_bridge_between_two_substantial_subtrees_is_kept():
    seg_raw = _build_h_phantom()
    _, brPts, endPts, reports = _run_pipeline(seg_raw)
    seg_stats, fil_stats, *_ = reports

    # full H-shape preserved: 4 long arms + 1 bridge segment, 2 branch points,
    # 4 terminal tips
    assert len(seg_stats) == 5
    assert len(brPts) == 2
    assert len(endPts) == 4

    # the short segment itself must still be present and reported
    short_segments = seg_stats[seg_stats["length"] < 40]
    assert len(short_segments) == 1


def test_without_the_safeguard_the_bridge_would_be_cut():
    """Negative control: confirms the phantom and PARAMS actually exercise the
    bug being guarded against, not just a geometry that happens to survive
    regardless. With the safeguard disabled, the same phantom must collapse
    into two plain, unbranched lines."""
    seg_raw = _build_h_phantom()
    orig = Filament._is_bridge_between_substantial_structure
    Filament._is_bridge_between_substantial_structure = lambda self, path: False
    try:
        _, brPts, endPts, reports = _run_pipeline(seg_raw)
    finally:
        Filament._is_bridge_between_substantial_structure = orig
    seg_stats, fil_stats, *_ = reports

    assert len(seg_stats) == 2
    assert len(brPts) == 0
