import numpy as np
import networkx as nx

from vessel_analysis_3d.graph.core import GraphObj

# A tube that runs straight, then curves sharply (170 degrees) starting
# right where a pre-existing skeleton endpoint sits -- reproducing the
# geometry of a real "hook" bug (2026-08-20): compounding per-step
# direction corrections during endpoint extension, with no bound on total
# rotation, could follow a locally-ambiguous ridge signal far enough to
# turn the walk back on itself, landing the final endpoint well inside the
# vessel instead of at its tip. _extend_endpoints_to_raw_boundary now bounds
# cumulative rotation from the walk's own established outward direction
# (default max_direction_deviation_deg=90) and stops once a correction
# would exceed it, rather than continuing into a physically implausible
# reversal. This phantom's arc turns twice that (170 degrees) specifically
# to confirm the walk actually stops partway around it instead of
# following the curve all the way, whether or not real anatomy would.
#
# All coordinates/tuples here are (z, y, x), matching how the rest of this
# module indexes radiusMatrix[node] and every array in the codebase.
RADIUS = 6
STRAIGHT_LEN = 30
TURN_DEG = 170.0
TURN_RADIUS = 12
SHAPE = (40, 100, 90)  # (z, y, x)
Z0, Y0, X0 = 20, 60, 10
# Deliberately anisotropic and non-round -- the extension's per-axis voxel
# step is step_phys/pixel_dims[axis], where step_phys=min(pixel_dims)/2,
# which is *exactly* 0.5 for whichever axis has the minimum value. With
# isotropic pixel_dims that hits every moving axis at once, landing
# repeatedly on exact .5 voxel boundaries where Python's round-half-to-even
# can stall the walk (round(28.5) == 28). Z never moves in this phantom, so
# it's set as the minimum -- the two axes that actually move (Y, X) get
# distinct, non-round values instead. Real anisotropic data (this
# dataset's own pixel sizes differ ~2.4x between Z and XY) would never hit
# this by coincidence; it's purely a synthetic-phantom construction detail.
PIXEL_DIMS = [0.9, 1.3, 1.7]


def _build_hook_tube():
    n_straight = STRAIGHT_LEN * 2
    straight = [(float(Z0), float(Y0), X0 + s / 2.0) for s in range(n_straight)]
    arc_center_y = Y0 - TURN_RADIUS
    arc_center_x = X0 + STRAIGHT_LEN
    n_arc = int(TURN_DEG * 2)
    arc = []
    for k in range(n_arc + 1):
        theta = np.radians(TURN_DEG) * k / n_arc
        x = arc_center_x + TURN_RADIUS * np.sin(theta)
        y = arc_center_y + TURN_RADIUS * np.cos(theta)
        arc.append((float(Z0), y, x))
    centerline = straight + arc

    mask = np.zeros(SHAPE, dtype=np.uint8)
    zz, yy, xx = np.indices(SHAPE)
    for cz, cy, cx in centerline:
        dist2 = (zz - cz) ** 2 + (yy - cy) ** 2 + (xx - cx) ** 2
        mask[dist2 <= RADIUS**2] = 1
    return mask


def _build_pre_existing_chain():
    """A straight skeleton chain covering just the straight part, ending
    exactly where the sharp curve begins -- as if Lee's thinning had
    already produced this much and extension is about to walk forward
    into the curved, geometry-ambiguous region."""
    graph = nx.Graph()
    nodes = [(Z0, Y0, X0 + s) for s in range(0, STRAIGHT_LEN, 2)]
    for a, b in zip(nodes, nodes[1:]):
        graph.add_edge(a, b)
    return graph, nodes


def test_extension_stops_within_direction_bound_instead_of_hooking():
    raw_mask = _build_hook_tube()
    graph, chain_nodes = _build_pre_existing_chain()
    endpoint = chain_nodes[-1]  # sits right where the sharp curve begins

    shim = object.__new__(GraphObj)
    shim.pixelDims = PIXEL_DIMS
    shim.networkxGraph = graph
    shim.radiusMatrix = np.zeros(SHAPE, dtype=float)
    for n in chain_nodes:
        shim.radiusMatrix[n] = float(RADIUS)
    shim.runTimeDict = {"endpointExtension": 0}
    shim.infoDict = {"endpointsExtended": 0, "endpointsRecentered": 0}
    shim.nodeDegreeDict = dict(nx.degree(graph))
    shim.recenteredEndpoints = set()

    GraphObj._extend_endpoints_to_raw_boundary(shim, raw_mask)

    base_nodes = set(chain_nodes)
    added = [n for n in shim.networkxGraph.nodes() if n not in base_nodes]
    assert added, "extension should have walked forward into the curve at all"

    # walk the newly-added chain in order from the original endpoint,
    # always preferring a neighbor that was actually added by extension
    # (picking an arbitrary neighbor could walk backward into the
    # pre-existing chain instead of following the new one)
    prev, cur = None, endpoint
    ordered = []
    while True:
        neighbors = [n for n in shim.networkxGraph.neighbors(cur) if n != prev]
        candidates = [n for n in neighbors if n not in base_nodes]
        if not candidates:
            break
        prev, cur = cur, candidates[0]
        ordered.append(cur)

    assert len(ordered) >= 2, "expected the walk to add more than one voxel"
    initial_direction = np.array([0.0, 0.0, 1.0])  # the straight chain's own +x tangent, (z,y,x) order
    final_direction = np.array(ordered[-1], dtype=float) - np.array(ordered[-2], dtype=float)
    final_direction = final_direction / np.linalg.norm(final_direction)
    cos_dev = float(np.dot(final_direction, initial_direction))
    deviation_deg = np.degrees(np.arccos(np.clip(cos_dev, -1.0, 1.0)))

    assert deviation_deg <= 90.0 + 1e-6, (
        f"walk rotated {deviation_deg:.1f} degrees from its own outward direction -- "
        "should have stopped at the max_direction_deviation_deg bound instead of hooking"
    )
    # and confirm this is a real, non-trivial check: the tube's own arc
    # goes to 170 degrees, so a walk that actually followed it all the way
    # (the bug's own behavior) would fail the assertion above
    assert deviation_deg < TURN_DEG - 30.0
