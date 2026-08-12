import os
from collections import defaultdict
import time
import numpy as np
from typing import Union
from pathlib import Path
import csv

from scipy.ndimage import center_of_mass, distance_transform_edt
import networkx as nx
import dask.array as da

# from dask.distributed import Client
from .filament import Filament
from .measurement import getLength


class distance_transform_edt_dask:
    def __init__(self, sampling):
        self.sampling = sampling

    def compute_distance_transform(self, im):
        return distance_transform_edt(im.astype(float), sampling=self.sampling)


class GraphObj:
    """
    Find statistics on a graph of a skeleton

    Parameters
    ----------
    graph : networkx graph
        networkx graph of a skeleton
    """

    def __init__(
        self,
        segmentation,
        skeleton,
        networkxGraph,
        pixelDimensions,
        pruningScale,
        lengthLimit,
        diaScale,
        branchingThreshold,
        removeBorderEndPts: bool = True,
        removeEndPtsFromSmallFilaments: bool = True,
        interpolate: bool = True,
        cut_neighbor_brpt_segs: bool = True,
        infoFile: Path = None,
        expFlag: bool = False,
        smallRAMmode: bool = False,
        fileName: Union[str, Path] = None,
        minBranchLengthUm: float = 4.0,
        minBranchRadiusUm: float = 1.0,
        minJunctionRadiusUm: float = 1.0,
        maskOpeningRadiusUm: float = 0.75,
        clusterCollapseScale: float = 3.0,
        maskTipRecoverySteps: int = 1,
        rawSegmentation=None,
    ):
        self.skeleton = skeleton
        self.rawSegmentation = (
            rawSegmentation if rawSegmentation is not None else segmentation
        )
        self.networkxGraph = networkxGraph
        self.pixelDims = pixelDimensions
        self.prunScale = pruningScale
        self.lengthLim = lengthLimit
        self.diaScale = diaScale
        self.branchingThresh = branchingThreshold
        self.minBranchLengthUm = minBranchLengthUm
        self.minBranchRadiusUm = minBranchRadiusUm
        self.minJunctionRadiusUm = minJunctionRadiusUm
        # consumed as mask preprocessing in Pipeline3D.run() before this class is
        # constructed; kept here only so the analysis-parameters dict can be
        # forwarded to this constructor unfiltered
        self.maskOpeningRadiusUm = maskOpeningRadiusUm
        self.maskTipRecoverySteps = maskTipRecoverySteps
        self.clusterCollapseScale = clusterCollapseScale
        self.removeBorderEndPts = removeBorderEndPts
        self.removeEndPtsFromSmallFilaments = removeEndPtsFromSmallFilaments
        self.interpolate = interpolate
        self.cut_neighbor_brpt_segs = cut_neighbor_brpt_segs

        self.expFlag = expFlag
        assert (
            not expFlag
        ), "experiment_flag is a legacy parameter, and can only be False"

        self.smallRAMmode = smallRAMmode
        assert not smallRAMmode, "smallRAMmode is not supported yet"

        self.fileName = fileName
        assert (
            fileName is None
        ), "fileName will only be used after smallRAMmode is added"

        self.infoFile = infoFile
        assert infoFile is None, "infoFile is a legacy parameter, and can only be None"

        self.segmentsDict = defaultdict(dict)
        self.countSegmentsDict = {}
        self.branchPointsDict = {}
        self.endPointsDict = {}
        self.borderEndPointsDict = {}
        self.countBranchPointsDict = {}
        self.countEndPointsDict = {}
        self.compTime = 0
        self.postProcessTime = 0
        self.endPtsTopVsBottom = 0
        self.nodesFinal = set()
        self.runTimeDict = {
            "distTransformation": 0,
            "clusterCollapse": 0,
            "pruning": 0,
            "endpointExtension": 0,
            "dfsComp": 0,
            "postProcessing": 0,
            "statCalculation": 0,
        }
        self.infoDict = {
            "clusterCollapse": 0,
            "pruning": 0,
            "endpointsExtended": 0,
            "filaments": 0,
            "segments": 0,
            "postProcBranches": 0,
            "postProcEndPts": 0,
        }
        self.nodeDegreeDict = dict(nx.degree(self.networkxGraph))
        # dictionaries containing all filament, segment and branch point statistics
        self.segStatsDict = defaultdict(dict)
        self.segmentProfilesDict = defaultdict(dict)
        self.filStatsDict = defaultdict(dict)
        self.branchesBrPtDict = defaultdict(dict)

        # calculate distance transform matrix
        self.initTime = time.time()
        if not self.smallRAMmode:
            self.distTransf = distance_transform_edt(
                segmentation, sampling=self.pixelDims
            )
        else:
            im_dask = da.from_array(segmentation, chunks=(128, 128, 128))
            if (
                segmentation.shape[0] * segmentation.shape[1] * segmentation.shape[2]
                > 16777216
            ):
                im_dask = da.from_array(segmentation, chunks="auto")
            edt_func = distance_transform_edt_dask(sampling=self.pixelDims)
            self.distTransf = da.map_overlap(
                edt_func.compute_distance_transform,
                im_dask,
                dtype="float",
                depth=15,
                boundary="reflect",
            )
        self.radiusMatrix = self.distTransf * self.skeleton
        self.runTimeDict["distTransformation"] = round(time.time() - self.initTime, 3)

        # pruning: delete branches with length below the distance to its closest border
        if self.smallRAMmode:
            self.radiusMatrix.to_zarr(
                "tmp_zarr" + os.sep + self.fileName + "_radiusMatrix.zarr"
            )
            radiusMat = da.from_zarr(
                "tmp_zarr" + os.sep + self.fileName + "_radiusMatrix.zarr"
            )
        else:
            radiusMat = self.radiusMatrix
        self._collapse_dense_clusters(radiusMat)
        self._prune(radiusMat)
        self._extend_endpoints_to_raw_boundary(self.rawSegmentation)

        self.filaments = list(self.connected_component_subgraphs(self.networkxGraph))
        # print("found {} filaments".format(len(self.filaments)))

    def prune_and_analyze(self, return_final_skel: bool = True):
        """
        Set the statistics of a networkx graph
        1) go through each subgraph
        2) calculate all end points and set one end point as beginning point
        3) extract an adjacency dictionary of networkx subgraph
        4) calculate the subgraphs statistics with the class Filament
        5) save statistics for each filament

        Parameters
        ----------
        Graph : networkx graph
            networkx graph of a skeleton
        """
        # statistic calculation
        startTime = time.time()
        self.infoDict["filaments"] = len(self.filaments)
        for ithDisjointGraph, subGraphSkeleton in enumerate(self.filaments):
            nodeDegreeDict = dict(nx.degree(subGraphSkeleton))
            endPoints = [k for (k, v) in nodeDegreeDict.items() if v == 1]
            if endPoints:
                start = endPoints[0]  # take random end point as beginning
                adjacencyDict = nx.to_dict_of_lists(subGraphSkeleton)
                filament = Filament(
                    adjacencyDict,
                    start,
                    self.radiusMatrix,
                    self.pixelDims,
                    self.lengthLim,
                    self.diaScale,
                    self.branchingThresh,
                    self.expFlag,
                    self.smallRAMmode,
                    self.fileName,
                    self.removeBorderEndPts,
                    self.removeEndPtsFromSmallFilaments,
                    self.interpolate,
                    self.cut_neighbor_brpt_segs,
                    self.minBranchLengthUm,
                    self.minBranchRadiusUm,
                    self.minJunctionRadiusUm,
                )
                filament.dfs_iterative()
                self.segmentsDict[ithDisjointGraph] = filament.segmentsDict

                # save nodes of filament after processing
                nodes = list(self.segmentsDict[ithDisjointGraph].values())
                nodes = set([item for sublist in nodes for item in sublist])
                self.nodesFinal.update(nodes)

                # filament may have no segments left after postprocessing
                if len(self.segmentsDict[ithDisjointGraph]) != 0:
                    self.countSegmentsDict[ithDisjointGraph] = len(
                        self.segmentsDict[ithDisjointGraph]
                    )
                    self.branchPointsDict[ithDisjointGraph] = filament.brPtsDict
                    self.endPointsDict[ithDisjointGraph] = filament.endPtsList
                    self.borderEndPointsDict[ithDisjointGraph] = (
                        filament.borderEndPts
                    )
                    self.countBranchPointsDict[ithDisjointGraph] = len(
                        filament.brPtsDict
                    )
                    self.countEndPointsDict[ithDisjointGraph] = len(filament.endPtsList)
                    self.infoDict["segments"] += len(
                        self.segmentsDict[ithDisjointGraph]
                    )
                    self.runTimeDict["dfsComp"] += filament.compTime
                    self.runTimeDict["postProcessing"] += filament.postprocessTime
                    self.infoDict["postProcBranches"] += filament.postprocBranches
                    self.infoDict["postProcEndPts"] += filament.postprocEndPts

                    # fill dictionaries containing all filament, segment and branch
                    # point statistics
                    self.segStatsDict[ithDisjointGraph] = filament.segmentStats
                    self.segmentProfilesDict[ithDisjointGraph] = (
                        filament.segmentProfiles
                    )
                    self.filStatsDict[ithDisjointGraph]["TerminalPoints"] = (
                        self.countEndPointsDict[ithDisjointGraph]
                    )
                    self.filStatsDict[ithDisjointGraph]["BranchPoints"] = (
                        self.countBranchPointsDict[ithDisjointGraph]
                    )
                    self.filStatsDict[ithDisjointGraph]["Segments"] = (
                        self.countSegmentsDict[ithDisjointGraph]
                    )
                    self.branchesBrPtDict[ithDisjointGraph] = filament.brPtsDict
                else:
                    self.infoDict["filaments"] -= 1
        self.runTimeDict["statCalculation"] = round(time.time() - startTime, 3)

        if self.expFlag:
            self.endPtsTopVsBottom = self.top_endPts_vs_bottom_endPts()

        if self.infoFile:
            self._writeInfoFile()

        if return_final_skel:
            # remove nodes which were removed in postprocessing of filaments
            # in the overall graph
            nodes_graph = set(self.networkxGraph.nodes)
            nodes_to_remove = nodes_graph - self.nodesFinal
            for node in nodes_to_remove:
                self.networkxGraph.remove_node(node)
            self.skeleton = self._get_final_skeleton()

            return self.skeleton

    def _get_final_skeleton(self):
        skel = np.zeros(self.skeleton.shape)
        for ind in list(self.networkxGraph.nodes):
            skel[ind] = 1
        return skel

    def _writeInfoFile(self):
        with open(self.infoFile, "w", newline="") as csvfile:
            fieldnames = ["step", "time(s)", "comment"]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            writer.writeheader()
            writer.writerow(
                {
                    "step": "Distance transformation",
                    "time(s)": str(self.runTimeDict["distTransformation"]),
                }
            )
            writer.writerow(
                {
                    "step": "Pruning",
                    "time(s)": str(self.runTimeDict["pruning"]),
                    "comment": "pruned " + str(self.infoDict["pruning"]) + " branches",
                }
            )
            writer.writerow(
                {
                    "step": "DFS calculations",
                    "time(s)": str(round(self.runTimeDict["dfsComp"], 3)),
                }
            )
            writer.writerow(
                {
                    "step": "Postprocessing",
                    "time(s)": str(round(self.runTimeDict["postProcessing"], 3)),
                    "comment": "removed "
                    + str(self.infoDict["postProcBranches"])
                    + " branches and "
                    + str(self.infoDict["postProcEndPts"])
                    + " end points",
                }
            )
            writer.writerow(
                {
                    "step": "Statistic calculation",
                    "time(s)": str(self.runTimeDict["statCalculation"]),
                    "comment": "filaments="
                    + str(self.infoDict["filaments"])
                    + " segments="
                    + str(self.infoDict["segments"]),
                }
            )

    def _collapse_dense_clusters(self, radiusMat):
        """
        Simplify small, locally cyclic or plate-like voxel clusters down to a
        thin, cycle-free connecting structure.

        Lee's topological thinning only guarantees topology preservation, not a
        1-voxel-wide curve everywhere: at a roughly blob-shaped confluence (e.g. a
        3-way junction where the branches meet at comparable widths) or where
        boundary roughness crosses itself, it can leave a small tangle of
        interconnected voxels, or even a residual 2D surface patch, instead of a
        single point. These form cycles in the skeleton graph and are invisible to
        the endpoint-based pruning in `_prune` (a cycle has no degree-1 node to
        start a walk from).

        For each such cluster (as long as its physical extent is small relative
        to the local vessel radius, so a genuine, larger-scale vascular loop is
        left untouched), every node is dropped *except* those lying on a
        shortest-path tree from the cluster's highest-radius node -- the best
        proxy for the true anatomical junction centre -- out to each of the
        cluster's attachment points (the nodes that actually touch the rest of
        the skeleton). This eliminates the cycles that made the cluster "dense"
        in the first place (a single-root shortest-path tree is cycle-free by
        construction), while keeping a real, physically continuous voxel path
        through the junction -- unlike collapsing to one point, this doesn't
        leave a visual gap in the exported skeleton, and unlike the earlier
        collapse-to-one-point design, it never adds a graph edge that didn't
        already exist (attachment points keep their original edges to their
        outside neighbors untouched), so it can't create a connection between
        nodes that were never actually adjacent.
        """
        startTime = time.time()
        total_simplified = 0
        changed = True
        while changed:
            changed = False
            for component in list(nx.biconnected_components(self.networkxGraph)):
                if len(component) < 3:
                    continue  # a single edge is not a cycle
                if not all(self.networkxGraph.has_node(n) for n in component):
                    # overlaps a cluster simplified earlier in this pass via a
                    # shared articulation point; revisit on the next pass
                    continue

                nodes = list(component)
                radii = [float(radiusMat[n]) for n in nodes]
                max_radius = max(radii)
                coords = np.array(nodes, dtype=float) * np.array(self.pixelDims)
                diffs = coords[:, None, :] - coords[None, :, :]
                cluster_diameter_um = float(np.sqrt((diffs**2).sum(-1)).max())

                if cluster_diameter_um >= self.clusterCollapseScale * max_radius:
                    continue  # likely a genuine larger-scale vascular loop

                representative = nodes[int(np.argmax(radii))]
                component_set = set(nodes)
                attachment_points = {
                    n
                    for n in nodes
                    if any(
                        neighbor not in component_set
                        for neighbor in self.networkxGraph.neighbors(n)
                    )
                }

                cluster_subgraph = self.networkxGraph.subgraph(nodes)
                paths = nx.single_source_shortest_path(cluster_subgraph, representative)
                nodes_to_keep = {representative}
                for ap in attachment_points:
                    nodes_to_keep.update(paths[ap])

                tree_edges = {
                    frozenset((paths[m][-2], paths[m][-1]))
                    for m in nodes_to_keep
                    if m != representative
                }

                for n in nodes:
                    if n not in nodes_to_keep:
                        self.networkxGraph.remove_node(n)
                # Nodes kept on the tree can still be joined by an original
                # "chord" edge that isn't part of the tree (e.g. two branches
                # of the shortest-path tree that also happened to be directly
                # adjacent in the raw cluster). Left in place, that chord
                # keeps the retained set cyclic, so the next pass's
                # biconnected_components() finds it again -- reprocessing an
                # ~1000-node cluster's O(n^2) diameter check every pass
                # without making progress. Tree_edges alone already connect
                # every kept node, so dropping chords is always safe.
                for u, v in list(self.networkxGraph.subgraph(nodes_to_keep).edges()):
                    if frozenset((u, v)) not in tree_edges:
                        self.networkxGraph.remove_edge(u, v)

                total_simplified += 1
                changed = True
        self.nodeDegreeDict = dict(nx.degree(self.networkxGraph))
        self.runTimeDict["clusterCollapse"] = round(time.time() - startTime, 3)
        self.infoDict["clusterCollapse"] = total_simplified

    def _prune(self, radiusMat):
        """
        This implements a conservative pruning step for weak, noise-like branches.

        The old heuristic only removed branches based on a short path-length to the
        first branch point. For imperfect, weakly connected liver segmentations, this
        can incorrectly keep thin bridge-like artifacts. We now prune a branch if it is
        short in physical units and/or its local radius is below a physical threshold.

        This runs to a fixed point: removing a spur can drop its branch point's
        degree to 2 (it stops being a branch point, correctly merging into a single
        through-segment) or to 1 (it becomes a new end point). A rough, few-voxel
        bridge in the segmentation often produces a small cascade of such nearby
        branch points, so a single pass only cleans the outermost one. Looping until
        no branch is removed catches the whole cascade instead of leaving the inner
        spurious branch points behind.
        """
        startTime = time.time()
        total_pruned = 0
        while True:
            endPtsList = [k for (k, v) in self.nodeDegreeDict.items() if v == 1]
            branchesToRemove = []
            for endPt in endPtsList:
                visited, stack = set(), [endPt]
                branch = []
                while stack:
                    vertex = stack.pop()
                    if vertex not in visited:
                        visited.add(vertex)
                        branch.append(vertex)
                        neighbors = [n for n in self.networkxGraph.neighbors(vertex)]
                        for neighbor in neighbors:
                            if neighbor not in visited:
                                stack.append(neighbor)
                        if len(neighbors) > 2:  # branch point found
                            branch_length_um = getLength(branch, self.pixelDims)
                            branch_point_radius_um = float(radiusMat[vertex])
                            branch_mean_radius_um = float(
                                np.mean([radiusMat[n] for n in branch[:-1]])
                            )
                            is_short_branch = branch_length_um < self.minBranchLengthUm
                            is_junction_thin = (
                                branch_point_radius_um < self.minJunctionRadiusUm
                            )
                            is_branch_thin = branch_mean_radius_um < self.minBranchRadiusUm
                            is_weak_branch = is_junction_thin or is_branch_thin

                            # Absolute thresholds assume a locally cylindrical vessel,
                            # which does not hold at bulges or where two vessels touch.
                            # As a scale-independent complement, also prune a spur whose
                            # length is short relative to the trunk radius it attaches
                            # to (pruningScale): a real bifurcation's branches are
                            # normally at least on the order of the trunk's own radius,
                            # while a noise spur off a rough boundary is much shorter
                            # than the trunk it sits on regardless of absolute size.
                            is_short_relative_to_trunk = (
                                branch_length_um
                                < self.prunScale * branch_point_radius_um
                            )

                            # For liver-vessel masks, only the branch-point and the
                            # interior support of the branch are reliable. Short and
                            # weakly supported branches are removed; true vessel tips
                            # are preserved.
                            if (is_short_branch and is_weak_branch) or is_short_relative_to_trunk:
                                branchesToRemove.append(branch)
                            break
            if not branchesToRemove:
                break
            # delete branches from adjacency matrix besides branch points
            for branch in branchesToRemove:
                for node in branch[:-1]:
                    if self.networkxGraph.has_node(node):
                        self.networkxGraph.remove_node(node)
            total_pruned += len(branchesToRemove)
            self.nodeDegreeDict = dict(nx.degree(self.networkxGraph))
        self.runTimeDict["pruning"] = round(time.time() - startTime, 3)
        self.infoDict["pruning"] = total_pruned

    def _boundary_centroid_voxel(
        self, raw_segmentation, seed_voxel, boundary_axis, exit_voxel, local_radius_um
    ):
        """
        Find the true centroid of the vessel's cross-section where it is cut
        by the imaged volume's boundary, on the plane perpendicular to
        `boundary_axis`.

        A straight tangent walk (see `_extend_endpoints_to_raw_boundary`)
        estimates its direction from a handful of interior skeleton nodes;
        at a border-cut cross-section, that estimate does not have to point
        at the section's centroid, so the walk can land on a side wall
        instead -- 84.3um off-center in one real example.

        This looks at the raw mask directly on the boundary plane, but only
        within a local physical window around `seed_voxel`, sized off the
        vessel's own local radius (`local_radius_um`) -- taking the centroid
        of the *whole* connected foreground region there is not safe: on
        real data, two skeleton endpoints ~21um apart were found sharing one
        1101-voxel connected region at their cut plane (their two vessels'
        raw masks happened to touch there), and both got redirected to the
        same, meaningless averaged centroid, landing 54-85um off -- worse
        than not correcting them at all. Restricting to a local window means
        a nearby, separate vessel's pixels never contribute to the centroid
        in the first place, regardless of whether the two masks are
        connected on this specific plane.
        """
        shape = raw_segmentation.shape
        boundary_index = 0 if exit_voxel[boundary_axis] < 0 else shape[boundary_axis] - 1

        slicer = [slice(None)] * 3
        slicer[boundary_axis] = boundary_index
        plane = raw_segmentation[tuple(slicer)] > 0
        if not plane.any():
            return None

        other_axes = [a for a in range(3) if a != boundary_axis]
        other_scale = np.array([self.pixelDims[a] for a in other_axes])
        seed_2d = np.array([seed_voxel[a] for a in other_axes], dtype=float)

        search_radius_um = max(10.0, 1.5 * local_radius_um)
        yy, xx = np.indices(plane.shape)
        dist_um = np.sqrt(
            ((yy - seed_2d[0]) * other_scale[0]) ** 2
            + ((xx - seed_2d[1]) * other_scale[1]) ** 2
        )
        local_mask = plane & (dist_um <= search_radius_um)

        if not local_mask.any():
            # seed missed the mask by a hair (e.g. right at a tapering
            # edge) -- fall back to the single nearest foreground pixel,
            # never the whole (possibly unrelated) connected region
            coords_2d = np.argwhere(plane)
            dists = np.sum((coords_2d - seed_2d) ** 2, axis=1)
            centroid_2d = coords_2d[np.argmin(dists)]
        else:
            centroid_2d = center_of_mass(local_mask)

        target = [0, 0, 0]
        target[boundary_axis] = boundary_index
        for a, v in zip(other_axes, centroid_2d):
            target[a] = int(round(v))
        return tuple(target)

    def _extend_endpoints_to_raw_boundary(
        self, raw_segmentation, tangent_lookback: int = 5, max_extension_um: float = 50.0
    ):
        """
        Extend true skeleton endpoints outward to the boundary of the raw
        (pre-opening) segmentation, along each endpoint's local tangent.

        Mask preprocessing (see Pipeline3D.run()) trims a few voxels off every
        tapering vessel tip along with the noise it is meant to suppress, and
        widening that opening was shown to reconnect most of the noise instead
        of just recovering the tips. Since each endpoint here is already final
        (after cluster collapse and pruning), it is walked straight outward, in
        physical space, through the untouched raw mask until it leaves the
        vessel. This only ever extends an existing leaf along its own
        established direction, so it cannot create a new branch point, and a
        suppressed noise spike -- not collinear with the segment -- is not
        picked up.
        """
        startTime = time.time()
        pixel_dims = np.array(self.pixelDims, dtype=float)
        step_phys = float(pixel_dims.min()) / 2.0
        max_steps = int(round(max_extension_um / step_phys))
        shape = raw_segmentation.shape

        endpoints = [n for n, d in self.networkxGraph.degree() if d == 1]
        extended = 0
        for endpoint in endpoints:
            # walk backward through the single-neighbor chain to estimate the
            # local tangent direction at this endpoint
            path = [endpoint]
            prev, cur = None, endpoint
            for _ in range(tangent_lookback):
                neighbors = [n for n in self.networkxGraph.neighbors(cur) if n != prev]
                if not neighbors:
                    break
                prev, cur = cur, neighbors[0]
                path.append(cur)
                if self.networkxGraph.degree(cur) != 2:
                    break  # reached a branch point or another endpoint
            if len(path) < 2:
                continue

            start_phys = np.array(path[-1], dtype=float) * pixel_dims
            end_phys = np.array(path[0], dtype=float) * pixel_dims
            direction = end_phys - start_phys
            norm = np.linalg.norm(direction)
            if norm < 1e-9:
                continue
            direction = direction / norm

            radius_here = float(self.radiusMatrix[endpoint])
            pos_phys = end_phys.copy()
            last_voxel = endpoint
            did_extend = False
            # None until the walk is redirected toward a boundary-plane
            # centroid (see below); then counts down so the walk stops
            # exactly there instead of continuing past it in a straight
            # line for the rest of its step budget.
            steps_until_target = None
            for _ in range(max_steps):
                next_pos_phys = pos_phys + step_phys * direction
                voxel = tuple(int(round(v)) for v in next_pos_phys / pixel_dims)
                out_of_bounds_axis = next(
                    (
                        axis
                        for axis, (v, s) in enumerate(zip(voxel, shape))
                        if v < 0 or v >= s
                    ),
                    None,
                )
                if out_of_bounds_axis is not None:
                    if steps_until_target is not None:
                        break  # already redirected once and still off the array -- give up
                    # Leaving the imaged volume, not the vessel -- this is a
                    # border cut, not a real tip. Redirect the remaining
                    # approach at the true cross-sectional centroid on this
                    # boundary plane instead of trusting the tangent
                    # estimate all the way to the edge, which can drift
                    # toward a side wall.
                    target = self._boundary_centroid_voxel(
                        raw_segmentation, last_voxel, out_of_bounds_axis, voxel, radius_here
                    )
                    if target is None:
                        break
                    target_phys = np.array(target, dtype=float) * pixel_dims
                    remaining = target_phys - pos_phys
                    remaining_dist = np.linalg.norm(remaining)
                    if remaining_dist < 1e-9:
                        break
                    direction = remaining / remaining_dist
                    steps_until_target = max(1, int(np.ceil(remaining_dist / step_phys)))
                    continue  # retry from the same position with the corrected direction
                pos_phys = next_pos_phys
                if not raw_segmentation[voxel]:
                    break
                if voxel != last_voxel:
                    self.networkxGraph.add_edge(last_voxel, voxel)
                    self.radiusMatrix[voxel] = radius_here
                    last_voxel = voxel
                    did_extend = True
                if steps_until_target is not None:
                    # count every iteration once redirected (even a
                    # zero-progress one, where voxel == last_voxel), so a
                    # sub-voxel-sized remaining distance still terminates
                    # the walk at the target instead of continuing on
                    steps_until_target -= 1
                    if steps_until_target <= 0:
                        break
            if did_extend:
                extended += 1

        if extended:
            self.nodeDegreeDict = dict(nx.degree(self.networkxGraph))
        self.runTimeDict["endpointExtension"] = round(time.time() - startTime, 3)
        self.infoDict["endpointsExtended"] = extended

    def connected_component_subgraphs(self, G):
        for c in nx.connected_components(G):
            yield G.subgraph(c)

    def top_endPts_vs_bottom_endPts(self, gap=15):
        z = self.radiusMatrix.shape[0] - 1
        top_endPts = 0
        bottom_endPts = 0
        for filament in self.endPointsDict:
            if self.countSegmentsDict[filament] > 4:
                for endPts in self.endPointsDict[filament]:
                    if endPts[0] < gap:
                        bottom_endPts += 1
                    if endPts[0] >= z - gap:
                        top_endPts += 1
        if bottom_endPts != 0:
            ratio = round(top_endPts / bottom_endPts, 4)
        else:
            ratio = "NULL"
        return ratio
