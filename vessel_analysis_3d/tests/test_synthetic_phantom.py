import numpy as np
import pytest
from scipy.ndimage import binary_closing, binary_dilation, binary_erosion
from skimage.morphology import skeletonize

from vessel_analysis_3d.processing_pipeline import (
    Pipeline3D,
    _anisotropic_ellipsoid_footprint,
)

# Synthetic T-junction phantom with exactly known ground truth (isotropic 1um
# voxels, so 1 voxel == 1um everywhere):
#   - trunk: cylinder along Z, axis at (y=50, x=15), radius 8, z in [10, 150]
#   - branch: cylinder along X, axis at (y=50, z=80), radius 8, x in [15, 85]
#   - branch attaches perpendicularly to the trunk at its midpoint (z=80)
# Ground truth: 1 branch point, 3 endpoints, 3 segments, diameter ~16um,
# each arm length ~70um, one ~90 degree branch angle. Arms are deliberately
# long relative to diameter -- an earlier, shorter draft of this phantom was
# (correctly) discarded whole by the pipeline's own "terminal segment shorter
# than diaScale*diameter" anti-stub-noise rule, which is real pruning
# behavior, not a bug.
SHAPE = (170, 100, 100)
RADIUS = 8
Y0, X0 = 50, 15
TRUNK_Z = (10, 150)
JUNCTION_Z = 80
BRANCH_X = (15, 85)
PIXEL_DIMS = [1.0, 1.0, 1.0]

PARAMS = {
    "pixelDimensions": PIXEL_DIMS,
    "pruningScale": 1.5,
    "lengthLimit": 3,
    "diaScale": 2,
    "branchingThreshold": 0.25,
    "minBranchLengthUm": 20.0,
    "minBranchRadiusUm": 25.0,
    "minJunctionRadiusUm": 25.0,
    "maskOpeningRadiusUm": 3.0,
    "clusterCollapseScale": 3.0,
    "maskTipRecoverySteps": 1,
    # this phantom deliberately has only 3 segments, below the pipeline's
    # default <5-segment "small filament" endpoint-suppression heuristic;
    # disabled here to validate raw endpoint counting on its own
    "removeEndPtsFromSmallFilaments": False,
}


def _build_t_junction_phantom():
    zz, yy, xx = np.indices(SHAPE)
    trunk = (zz >= TRUNK_Z[0]) & (zz <= TRUNK_Z[1]) & (
        (yy - Y0) ** 2 + (xx - X0) ** 2 <= RADIUS**2
    )
    branch = (xx >= BRANCH_X[0]) & (xx <= BRANCH_X[1]) & (
        (yy - Y0) ** 2 + (zz - JUNCTION_Z) ** 2 <= RADIUS**2
    )
    seg = np.zeros(SHAPE, dtype=np.uint8)
    seg[trunk | branch] = 1
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
        seg, skl, dict(PARAMS), save_path=None, basename="phantom", raw_segmentation=seg_raw
    )


@pytest.fixture(scope="module")
def phantom_result():
    seg_raw = _build_t_junction_phantom()
    skl_final, brPts, endPts, reports = _run_pipeline(seg_raw)
    seg_stats, fil_stats, br_stats, summary, profiles = reports
    return {
        "seg_raw": seg_raw,
        "brPts": brPts,
        "endPts": endPts,
        "seg_stats": seg_stats,
        "summary": summary,
        "profiles": profiles,
    }


def test_topology_matches_ground_truth(phantom_result):
    assert len(phantom_result["brPts"]) == 1
    assert len(phantom_result["endPts"]) == 3
    assert len(phantom_result["seg_stats"]) == 3


def test_endpoints_reach_true_tips(phantom_result):
    expected_tips = {(150, Y0, X0), (10, Y0, X0), (JUNCTION_Z, Y0, BRANCH_X[1])}
    measured_tips = set(phantom_result["endPts"])
    for expected in expected_tips:
        closest = min(
            measured_tips,
            key=lambda p: sum((a - b) ** 2 for a, b in zip(p, expected)),
        )
        dist = sum((a - b) ** 2 for a, b in zip(closest, expected)) ** 0.5
        assert dist <= 2.0, f"no measured endpoint within 2 voxels of true tip {expected}"


def test_diameter_matches_ground_truth(phantom_result):
    summary = phantom_result["summary"]
    true_diameter = 2 * RADIUS
    assert abs(summary["meanDiameter_um"].iloc[0] - true_diameter) < 1.5


def test_total_length_matches_ground_truth(phantom_result):
    true_length = 2 * (JUNCTION_Z - TRUNK_Z[0]) + (BRANCH_X[1] - X0)
    # arms: bottom (80-10=70), top (150-80=70), branch (85-15=70) -> 210um
    measured = phantom_result["summary"]["totalVesselLength_mm"].iloc[0] * 1000
    assert abs(measured - true_length) < 5.0


def test_vessel_volume_fraction_matches_voxel_count_exactly(phantom_result):
    seg_raw = phantom_result["seg_raw"]
    true_fraction = seg_raw.sum() / seg_raw.size
    measured = phantom_result["summary"]["vesselVolumeFraction"].iloc[0]
    assert measured == pytest.approx(true_fraction, abs=1e-9)


def test_diameter_profile_matches_ground_truth(phantom_result):
    profiles = phantom_result["profiles"]
    seg_stats = phantom_result["seg_stats"]

    # one profile per segment, matching segStats one-to-one
    assert profiles["segmentID"].nunique() == len(seg_stats)

    for segment_id, group in profiles.groupby("segmentID"):
        group = group.sort_values("pointIndex")
        # path position starts at the segment's first node and only grows
        assert group["position_um"].iloc[0] == pytest.approx(0.0, abs=1e-9)
        assert (group["position_um"].diff().dropna() >= 0).all()

        # profile should reach close to the segment's own reported length
        matching_stat = seg_stats[seg_stats["segmentID"] == segment_id]
        assert len(matching_stat) == 1
        stat_length = matching_stat["length"].iloc[0]
        assert group["position_um"].iloc[-1] == pytest.approx(stat_length, abs=2.0)

        # every diameter along the path should be within the segment's own
        # reported min/max range
        stat_min = matching_stat["minDiameter"].iloc[0]
        stat_max = matching_stat["maxDiameter"].iloc[0]
        assert (group["diameter_um"] >= stat_min - 1e-6).all()
        assert (group["diameter_um"] <= stat_max + 1e-6).all()
