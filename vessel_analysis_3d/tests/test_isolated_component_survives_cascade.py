import numpy as np
import networkx as nx

from vessel_analysis_3d.graph.core import GraphObj

# A standalone (not connected to anything else), genuinely branched
# structure -- a central branch point with 4 short arms, each individually
# thin/short enough to read as "weak" against GraphObj._prune's absolute
# thresholds -- reproducing a real bug (2026-08-20, segmentation_v30): an
# isolated 99-node structure with 4 endpoints and a 356um bounding-box
# diagonal (mean radius 23.2um, just under the 25um floor) was cascaded
# away completely by _prune's per-branch removal loop, which assumes every
# branch it evaluates belongs to a much larger trunk -- an assumption that
# doesn't hold for a standalone component with nowhere else to fall back
# on. _prune now restores the most substantial path through any originally
# multi-branch component that its own cascade would otherwise eliminate
# entirely.
PIXEL_DIMS = [1.0, 1.0, 1.0]
CENTER = (50, 50, 50)
ARM_LENGTH = 12
MIN_BRANCH_RADIUS_UM = 25.0
ARM_RADIUS_UM = 10.0  # deliberately below MIN_BRANCH_RADIUS_UM on every arm


def _build_four_armed_star():
    graph = nx.Graph()
    radii = {}
    radii[CENTER] = ARM_RADIUS_UM
    directions = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0)]
    for dz, dy, dx in directions:
        prev = CENTER
        for s in range(1, ARM_LENGTH + 1):
            node = (CENTER[0] + dz * s, CENTER[1] + dy * s, CENTER[2] + dx * s)
            graph.add_edge(prev, node)
            radii[node] = ARM_RADIUS_UM
            prev = node
    return graph, radii


def test_standalone_multi_branch_structure_survives_prune():
    graph, radii = _build_four_armed_star()
    radius_matrix_shape = (200, 200, 200)
    radius_matrix = np.zeros(radius_matrix_shape, dtype=float)
    for node, r in radii.items():
        radius_matrix[node] = r

    shim = object.__new__(GraphObj)
    shim.pixelDims = PIXEL_DIMS
    shim.networkxGraph = graph
    shim.prunScale = 1.5
    shim.minBranchLengthUm = 20.0
    shim.minBranchRadiusUm = MIN_BRANCH_RADIUS_UM
    shim.minJunctionRadiusUm = MIN_BRANCH_RADIUS_UM
    shim.runTimeDict = {"pruning": 0}
    shim.infoDict = {"pruning": 0, "isolatedComponentsRestored": 0}
    shim.nodeDegreeDict = dict(nx.degree(graph))

    # sanity check the phantom actually reproduces the failure mode:
    # every arm should individually read as weak by the existing rule
    assert ARM_RADIUS_UM < MIN_BRANCH_RADIUS_UM
    arm_length_um = ARM_LENGTH * min(PIXEL_DIMS)
    assert arm_length_um < shim.prunScale * ARM_RADIUS_UM, (
        "phantom arms should also be short relative to the branch point radius, "
        "matching the real cascading-removal trigger"
    )

    GraphObj._prune(shim, radius_matrix)

    assert shim.networkxGraph.number_of_nodes() > 0, (
        "a standalone, genuinely branched structure should never be cascaded "
        "away to nothing just because none of its own branches meets a "
        "threshold tuned for pruning noise off a much larger trunk"
    )
    assert shim.infoDict["isolatedComponentsRestored"] == 1
    # the restored path should span the structure's own longest throughline
    # (opposite arms), not just a single leftover node
    assert shim.networkxGraph.number_of_nodes() >= 2 * ARM_LENGTH + 1
    assert nx.is_connected(shim.networkxGraph)
