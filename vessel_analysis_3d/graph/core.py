import os
from collections import defaultdict
import time
import numpy as np
from typing import Union
from pathlib import Path
import csv

from scipy.ndimage import distance_transform_edt, maximum_filter
from scipy.spatial import cKDTree
from skimage.morphology import skeletonize
import networkx as nx
import dask.array as da

# from dask.distributed import Client
from .filament import Filament
from .measurement import getLength
from .networkx_from_array import get_networkx_graph_from_array


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
        maskSmoothingSigmaUm: float = 10.0,
        boundaryClipMarginUm: float = 25.0,
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
        self.maskSmoothingSigmaUm = maskSmoothingSigmaUm
        # see _flag_boundary_clipped_segments: how close (in physical units)
        # raw-mask foreground has to run to a volume face before the
        # segment it belongs to is flagged "boundaryClipped". 25um matches
        # minBranchRadiusUm -- the same "no vessel below 50um diameter"
        # domain floor already used elsewhere, so a vessel at least that
        # wide is never more than one of its own radii from a face it's
        # actually touching before this catches it.
        self.boundaryClipMarginUm = boundaryClipMarginUm
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
            "offCenterSpurPruning": 0,
            "endpointExtension": 0,
            "dfsComp": 0,
            "postProcessing": 0,
            "statCalculation": 0,
        }
        self.infoDict = {
            "clusterCollapse": 0,
            "pruning": 0,
            "isolatedComponentsRestored": 0,
            "offCenterSpursRemoved": 0,
            "endpointsExtended": 0,
            "endpointsRecentered": 0,
            "filaments": 0,
            "segments": 0,
            "postProcBranches": 0,
            "postProcEndPts": 0,
        }
        self.nodeDegreeDict = dict(nx.degree(self.networkxGraph))
        # endpoints whose approach was cut back and re-walked by
        # _find_trustworthy_anchor / _extend_endpoints_to_raw_boundary: the
        # resulting terminal segment can legitimately be shorter than its
        # own diameter (that's often exactly why the walk stopped there --
        # see _is_locally_elongated), so it must not be judged by
        # Filament's ordinary short-stub-relative-to-diameter noise rule,
        # which assumes a short, thick terminal segment is a raw
        # skeletonization artifact rather than a deliberately-truncated,
        # already-verified-centered one
        self.recenteredEndpoints = set()
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
        if not self.smallRAMmode:
            # Skeleton topology comes from a possibly-smoothed mask (see
            # maskSmoothingSigmaUm in processing_pipeline.py), so a
            # skeleton voxel can occasionally sit a voxel or two outside
            # the true, unsmoothed segmentation at a bend -- there,
            # distTransf (computed on the true mask) is exactly 0,
            # producing a physically impossible "vessel pinches to zero
            # width" reading instead of the node's real local radius.
            # Confirmed directly on real data (segmentation_v30): 42 of
            # 33701 skeleton voxels (0.12%) sit just outside the true
            # mask, corrupting minDiameter for 14 segments (one down to a
            # literal 0.0um mid-path, bracketed by 95um and 31um
            # neighbors). Correct only those exact degenerate nodes with
            # the nearest real tissue's distance-transform value, rather
            # than touching skeleton topology/connectivity at all -- the
            # rest of this pipeline extensively validates connectivity
            # and this must not risk it.
            degenerate = (self.skeleton > 0) & (self.distTransf == 0)
            if degenerate.any():
                nearby_max = maximum_filter(self.distTransf, size=5)
                self.radiusMatrix = np.where(
                    degenerate, nearby_max, self.radiusMatrix
                )
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
        # Re-enabled 2026-08-17 with a redundancy signal added: an earlier
        # version pruned any sufficiently off-center terminal branch and
        # cascaded into removing 28% of a real filament's nodes, including
        # its legitimate main line, because "off-center" alone can't tell
        # a redundant parasitic path (should prune) from a real, singular
        # path Lee's algorithm just didn't perfectly center on a wide
        # vessel (must keep) -- both look identical under that check
        # alone. The current version only prunes a flagged branch if a
        # trusted (not itself flagged) nearby path already covers the same
        # physical territory near its true medial-axis ridge -- see the
        # method's own docstring for the two-pass design.
        self._prune_off_center_spurs(self.rawSegmentation)
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
                    self.recenteredEndpoints,
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

        self._flag_boundary_clipped_segments()

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

    def _flag_boundary_clipped_segments(self):
        """
        Flags every segment with raw (unsmoothed) mask foreground within
        `self.boundaryClipMarginUm` of BOTH a volume face AND one of its
        own skeleton nodes -- not just segments with an endpoint at the
        edge, but also a vessel running roughly parallel to a face that
        never has any skeleton node touch the boundary itself, only its
        side.

        Ground truth is the raw mask, not a skeleton-derived radius
        estimate: a skeleton-local-radius check, tried first, misses the
        parallel case -- the medial axis of a partially-cropped
        cross-section simply describes the visible slice, it doesn't
        know anything is missing.

        This is deliberately a purely LOCAL, per-node test -- for each
        skeleton node close enough to a face to be worth checking, crop
        a small window of the raw mask around that node (spanning the
        margin band on the face-normal axis, +-margin on the two
        in-plane axes) and ask only "is there foreground in here at
        all". Every node whose crop finds foreground flags its own
        segment(s); a node shared by several segments (a branch point)
        flags all of them, since a junction that close to the edge means
        any of its branches could be the one being truncated.

        CHANGELOG:

        2026-08-20 v1: only looked at the exact boundary voxel layer
        (face_index 0 / shape-1). Mask-opening erosion/smoothing
        routinely leaves real edge-truncated vessels a few voxels short
        of literally touching the array boundary, so this missed them
        (Simon: "count everything within 25um of the boundary as
        boundary clipping").

        2026-08-20 v2: widened to a margin-thick slab per face, but
        grouped ALL foreground in that slab into connected-component
        "blobs" and attributed each blob to its single nearest skeleton
        node. This broke down on real, densely-vascularized data: a
        margin slab commonly contains many physically distinct vessels
        that happen to touch or bridge within that thin band, so 3D
        connected-component labeling silently fused them into one mega
        blob (confirmed on segmentation_v30: one blob alone was 399,194
        of 485,412 total near-boundary voxels, 82%, obviously dozens of
        unrelated vessels merged together). That blob's single centroid
        then got attributed to exactly one nearby segment -- every other
        real, distinct, genuinely boundary-touching vessel folded into
        the same merged blob was silently left unflagged, which is
        exactly what Simon caught by hovering a visibly boundary-hugging
        vessel that stayed unflagged.

        v3 removed blob grouping entirely -- there is no longer any
        "which vessel does this merged region belong to" question to get
        wrong, because nothing is ever merged across vessels: each node
        only ever looks at the raw mask immediately around itself. But
        v3's cheap pre-filter (skip a node if its own coordinate is
        farther than the margin from every face) reintroduced a version
        of the *original* problem this whole check exists to solve: a
        node's own position is not a reliable stand-in for how close the
        raw mask actually gets. Confirmed directly on segmentation_v30: a
        166um-diameter vessel's skeleton sat flat at z=10-11 (~83-91um
        from the z=0 face) the entire length Simon was looking at, well
        past a 25um margin -- while its raw mask was solid foreground
        all the way to z=1. A vessel that thick simply has its medial
        axis centered roughly its own radius away from the wall it's
        actually touching, the same reason the very first version of
        this check missed a vessel running parallel to a face, just
        along the face-normal axis instead of laterally this time.

        v4 (this version) widens the pre-filter's reach by each node's
        own local radius (`self.radiusMatrix[node]`, free -- already
        computed for every node in this pipeline): a node is worth
        cropping if its distance to a face is within margin + its own
        radius, not margin alone. A thick vessel gets a proportionally
        wider net; the crop step itself (which does the actual, precise
        mask check) is unchanged.
        """
        if self.rawSegmentation is None or not self.segmentsDict:
            return
        raw = self.rawSegmentation
        ndims = raw.ndim
        pixel_dims = np.asarray(self.pixelDims, dtype=float)
        margin = float(self.boundaryClipMarginUm)

        # every skeleton node belonging to a final segment, mapped to the
        # segment(s) that contain it -- shared only at branch/end points
        node_to_segments = defaultdict(list)
        for fil_idx, segs in self.segmentsDict.items():
            for seg_key, nodes in segs.items():
                for node in nodes:
                    node_to_segments[node].append((fil_idx, seg_key))
        if not node_to_segments:
            return

        margin_voxels = [
            max(1, min(int(np.ceil(margin / pixel_dims[a])), raw.shape[a]))
            for a in range(ndims)
        ]

        flagged = set()
        for node in node_to_segments:
            owners = node_to_segments[node]
            if all(o in flagged for o in owners):
                continue
            # a thick vessel's medial axis sits roughly its own radius away
            # from its true wall -- the same reason the very first version
            # of this check missed a vessel running parallel to a face,
            # just along the face-normal axis this time. Confirmed directly
            # on segmentation_v30: a 166um-diameter vessel's skeleton sat
            # flat at z=10-11 (~83-91um from the z=0 face, past a 25um
            # margin) while its raw mask was solid foreground all the way
            # to z=1 -- a plain distance-to-face pre-filter skipped it
            # entirely. Reach = margin + local radius so a thick vessel
            # gets a proportionally wider net to still be worth cropping.
            local_radius_um = float(self.radiusMatrix[node])
            reach = margin + local_radius_um
            for axis in range(ndims):
                dist_low = node[axis] * pixel_dims[axis]
                dist_high = (raw.shape[axis] - 1 - node[axis]) * pixel_dims[axis]
                for side, dist_to_face in ((0, dist_low), (1, dist_high)):
                    if dist_to_face > reach:
                        continue
                    slices = []
                    for a in range(ndims):
                        if a == axis:
                            if side == 0:
                                slices.append(slice(0, margin_voxels[a]))
                            else:
                                slices.append(
                                    slice(raw.shape[a] - margin_voxels[a], raw.shape[a])
                                )
                        else:
                            lo = max(0, node[a] - margin_voxels[a])
                            hi = min(raw.shape[a], node[a] + margin_voxels[a] + 1)
                            slices.append(slice(lo, hi))
                    if np.any(raw[tuple(slices)] > 0):
                        flagged.update(owners)
                        break
                else:
                    continue
                break

        for fil_idx, seg_key in flagged:
            self.segStatsDict[fil_idx][seg_key]["boundaryClipped"] = True

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

        # This cascading removal assumes every branch it evaluates belongs
        # to a much larger, well-supported trunk -- "a real bifurcation's
        # branches are normally at least on the order of the trunk's own
        # radius" only makes sense when there IS a trunk. A standalone
        # component with no connection to anything else has no such
        # fallback: if each of its own terminal branches happens to
        # individually read as weak against these same absolute
        # thresholds, the cascade can strip them one by one, each removal
        # legitimately thinning the branch point(s) left behind and
        # inviting the next one, until a genuinely substantial, real,
        # multi-branch vessel is reduced to nothing -- confirmed directly
        # on real data (segmentation_v30): an isolated 99-node structure
        # with 4 endpoints and a 356um bounding-box diagonal, mean radius
        # 23.2um (just under the 25um floor), cascaded away completely,
        # while a comparable isolated structure elsewhere survived only
        # because its own branches happened not to all read as weak.
        # Snapshotting which components have a real branch point before
        # any of this runs -- and restoring the single most substantial
        # path through one if the cascade below eliminates it outright --
        # protects exactly this case without touching the cascade itself,
        # so a branch/spur that legitimately has a large network to fall
        # back on is pruned exactly as before.
        substantial_components = [
            self.networkxGraph.subgraph(component).copy()
            for component in nx.connected_components(self.networkxGraph)
            if any(self.networkxGraph.degree(n) > 2 for n in component)
        ]

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
                else:
                    # Stack exhausted without ever finding a branch point:
                    # this endpoint's entire connected component is a bare
                    # path (or a single node) with no branching at all, so
                    # it's never reached by the branch-point-triggered logic
                    # above. A standalone fragment too thin to be a real
                    # vessel by the same minBranchRadiusUm floor used
                    # everywhere else in this function would otherwise
                    # survive completely untouched, however implausible its
                    # radius (confirmed on real data, segmentation_v30: an
                    # isolated 47um-long, 17.68um-diameter fragment survived
                    # at roughly a third of the configured 50um floor).
                    # Length isn't checked here, unlike the branch-point
                    # case -- there's no branch point to be short *relative
                    # to*, and a standalone fragment failing the absolute
                    # radius floor is implausible regardless of its length.
                    # Requiring the far end to also be a real endpoint
                    # (degree==1) rather than just "no branch point found"
                    # excludes a single isolated node (e.g. a closed loop
                    # collapsed to one representative point by
                    # _collapse_dense_clusters, which has degree 0 and no
                    # real endpoints at all -- not what this rule is about).
                    if branch and self.networkxGraph.degree(branch[-1]) == 1:
                        branch_mean_radius_um = float(
                            np.mean([radiusMat[n] for n in branch])
                        )
                        if branch_mean_radius_um < self.minBranchRadiusUm:
                            branchesToRemove.append(branch)
            if not branchesToRemove:
                break
            # delete branches from adjacency matrix besides branch points.
            # The branch's last node is only preserved if it's still a real
            # branch point (degree > 2) at delete time -- for a bare
            # endpoint-to-endpoint fragment (see the "no branch point
            # found" case above), the last node is just the far endpoint,
            # not something to keep, so the whole fragment is removed.
            for branch in branchesToRemove:
                keep_last = (
                    self.networkxGraph.has_node(branch[-1])
                    and self.networkxGraph.degree(branch[-1]) > 2
                )
                nodes_to_remove = branch[:-1] if keep_last else branch
                for node in nodes_to_remove:
                    if self.networkxGraph.has_node(node):
                        self.networkxGraph.remove_node(node)
            total_pruned += len(branchesToRemove)
            self.nodeDegreeDict = dict(nx.degree(self.networkxGraph))

        # Restore the single most substantial path through any originally-
        # branched component the cascade above eliminated completely (see
        # the comment where substantial_components is built). "Most
        # substantial" is the path between whichever pair of the
        # component's own original endpoints sit physically farthest
        # apart -- a real vessel's longest throughline -- rather than
        # reintroducing every branch that was individually flagged weak;
        # this keeps the fix narrowly scoped to "don't let a genuine,
        # sizeable vessel disappear entirely" without undoing the
        # cascade's actual noise-pruning judgment on structures that do
        # have somewhere else to fall back on.
        pixel_dims = np.array(self.pixelDims, dtype=float)
        components_restored = 0
        for snapshot in substantial_components:
            if any(self.networkxGraph.has_node(n) for n in snapshot.nodes()):
                continue  # some of it survived; leave the cascade's result as-is
            endpoints = [n for n, d in snapshot.degree() if d == 1]
            if len(endpoints) < 2:
                continue  # shouldn't happen for a component with a branch point, but be safe
            ends_phys = np.array(endpoints, dtype=float) * pixel_dims
            dists = np.linalg.norm(ends_phys[:, None, :] - ends_phys[None, :, :], axis=-1)
            i, j = np.unravel_index(np.argmax(dists), dists.shape)
            path = nx.shortest_path(snapshot, endpoints[i], endpoints[j])
            self.networkxGraph.add_edges_from(zip(path[:-1], path[1:]))
            components_restored += 1
        if components_restored:
            self.nodeDegreeDict = dict(nx.degree(self.networkxGraph))
        self.infoDict["isolatedComponentsRestored"] = components_restored

        self.runTimeDict["pruning"] = round(time.time() - startTime, 3)
        self.infoDict["pruning"] = total_pruned

    def _prune_off_center_spurs(
        self,
        raw_segmentation,
        offset_threshold: float = 0.5,
        min_bad_fraction: float = 0.6,
        num_samples: int = 8,
        min_redundancy_fraction: float = 0.6,
        redundancy_lateral_scale: float = 0.6,
    ):
        """
        Lee's thinning has no centering guarantee inside a wide vessel, for
        the same reason `_find_trustworthy_anchor` and `_local_raw_edt_ridge`
        document: with enough cross-sectional room, the algorithm can trace
        an entire secondary path running close to one wall instead of
        through the true center, not just a single off-center point. Unlike
        a genuine noise spur -- thin, short, well below the trunk's own
        radius, already caught by `_prune`'s length/diameter checks -- this
        kind of spur can be long (100-200um, confirmed on real data) and
        properly tapering, since it's following a real physical boundary
        the whole way, just the wrong one (the vessel wall, not the medial
        axis). So neither absolute length nor length-relative-to-trunk-
        radius can distinguish it; `_prune` correctly leaves it alone.

        What does distinguish it: measured directly on two confirmed real
        cases, points along such a spur show offset/radius (the same
        metric `_extend_endpoints_to_raw_boundary` already uses) at 0.5-0.94
        throughout almost the entire branch -- because the branch's own
        stored radius (computed on the thinned skeleton) is deflated purely
        by hugging a wall; the raw mask's true local radius there (found via
        the same ridge search) is consistently and substantially larger
        (confirmed: 55-67um true vs. 41um stored at several points). A
        genuine branch's skeleton, by contrast, tracks close to the true
        centerline throughout.

        Off-center alone is not enough to prune, though: a real, singular
        path that Lee's algorithm just failed to perfectly center on a wide
        vessel looks *identical* under this test to a redundant parasitic
        twin path -- both fail the same offset check for the same reason
        (there's room to wander). Pruning on the offset test alone was
        tried and reverted after it cascaded into removing 28% of a real
        filament, including its legitimate main line (see the dev log for
        2026-08-17). The missing piece is a redundancy signal: is there
        already another, trustworthy path nearby covering the same
        physical territory this one claims to?

        This runs in two passes per outer iteration to get that signal
        without two candidates being able to validate each other into
        removal:

        Pass A walks every terminal chain and flags it as an off-center
        *candidate* (same offset test as before) without removing anything.

        Pass B builds a KD-tree of every node that is *not* part of any
        flagged candidate from Pass A -- the "trusted" set, i.e. the parts
        of the skeleton this same test didn't already object to -- and only
        prunes a candidate if most of its own off-axis sample points have a
        trusted node sitting near their true ridge point (the medial-axis
        location `_local_raw_edt_ridge` already found while measuring the
        offset -- no extra search needed). A real-but-imperfect singular
        path has no such trusted twin nearby and survives; a redundant
        parasitic path, by definition, has the real trunk running right
        alongside it and gets pruned. Two mutually off-center spurs can no
        longer clear each other, since both are excluded from the trusted
        set before either is tested.

        Runs after `_prune` (so ordinary noise spurs are already gone) and
        before `_extend_endpoints_to_raw_boundary` (so an off-center spur
        never gets the chance to be recentered/extended as if it were a
        real vessel tip -- it should be discarded outright, not repositioned).

        `redundancy_lateral_scale` default of 0.6 (rather than a wider
        1.0) reflects the addition of upstream mask smoothing
        (`maskSmoothingSigmaUm`, see processing_pipeline.py): once the
        raw mask's boundary roughness is smoothed away before
        skeletonization, this function has far less work left to do
        (confirmed on `segmentation_test`: the pre-off-center-pruning
        checkpoint drops from 126 raw endpoints unsmoothed to 30 with
        smoothing on), so the wider, more aggressive search is no longer
        worth its added risk -- 0.6 vs. 1.0 only differs by ~4 spurs on
        smoothed data (6 vs. 10 removed), while 0.6 keeps more margin.
        Without smoothing (`maskSmoothingSigmaUm=0`), a wider search was
        separately validated safe up to at least 1.0-1.5x local radius --
        widening the trusted-twin search from 0.6x to 1.0x raised the
        catch rate 30->49 removed spurs on unsmoothed data, with every
        individual removed component staying within the same size
        envelope confirmed to be genuine wall-hugging spurs (max 47
        nodes / 210um bounding-box diagonal at 1.0x, vs. max 41 nodes /
        171um at 0.6x -- no evidence of the redundancy gate letting a
        merged/cascaded removal through as the search widens); revisit
        that wider setting if smoothing is ever disabled.
        """
        startTime = time.time()
        pixel_dims = np.array(self.pixelDims, dtype=float)
        slab_half_um = max(float(pixel_dims.min()) / 2.0, float(pixel_dims.min()))
        total_removed = 0
        changed = True
        while changed:
            changed = False
            endpoints = [n for n, d in self.networkxGraph.degree() if d == 1]

            # Pass A -- flag candidates, remove nothing yet.
            candidates = []
            for endpoint in endpoints:
                if not self.networkxGraph.has_node(endpoint):
                    continue
                # walk the terminal chain back to the nearest structural node
                chain = [endpoint]
                prev, cur = None, endpoint
                while True:
                    neighbors = [n for n in self.networkxGraph.neighbors(cur) if n != prev]
                    if not neighbors:
                        break
                    prev, cur = cur, neighbors[0]
                    chain.append(cur)
                    if self.networkxGraph.degree(cur) != 2:
                        break
                # too short to sample meaningfully, or ends at another
                # endpoint (an isolated short filament, not a spur off a
                # larger structure -- out of scope for this check)
                if len(chain) < 4 or self.networkxGraph.degree(chain[-1]) < 3:
                    continue

                interior = chain[:-1]  # exclude the branch point itself
                # skip the quarter of the branch closest to its own tip:
                # any tapering branch, real or spurious, is trivially
                # "centered" right at its own point -- there is no room to
                # be off-axis there -- so including it dilutes the real
                # signal, which lives in the middle/base of the branch
                # (confirmed directly: a real off-center spur's bad
                # fraction came out to exactly 0.50, just under the 0.6
                # cutoff, purely from its two tip-most samples both
                # reading 0.00; excluding them raised it to 0.83)
                skip = max(0, int(0.25 * len(interior)))
                sample_start = min(skip, len(interior) - 1)
                sample_idx = sorted(set(
                    np.linspace(sample_start, len(interior) - 1, min(num_samples, len(interior) - sample_start)).astype(int)
                ))
                offsets = []
                bad_samples = []  # (ridge_point, edt_radius) for offset >= threshold
                for k in sample_idx:
                    lo_i, hi_i = max(0, k - 1), min(len(interior) - 1, k + 1)
                    if lo_i == hi_i:
                        continue
                    direction = np.array(interior[lo_i], dtype=float) - np.array(interior[hi_i], dtype=float)
                    norm = np.linalg.norm(direction)
                    if norm < 1e-9:
                        continue
                    direction = direction / norm
                    node = interior[k]
                    pos_phys = np.array(node, dtype=float) * pixel_dims
                    radius_here = float(self.radiusMatrix[node])
                    window_um = max(6.0, 1.2 * radius_here) if radius_here > 0 else 20.0
                    ridge = self._local_raw_edt_ridge(
                        raw_segmentation, pos_phys, pixel_dims, window_um, direction, slab_half_um
                    )
                    if ridge is None:
                        continue
                    ridge_point, edt_radius, extent_radius = ridge
                    if edt_radius < 1e-6:
                        continue
                    lateral = pos_phys - ridge_point
                    lateral = lateral - np.dot(lateral, direction) * direction
                    offset = np.linalg.norm(lateral) / edt_radius
                    offsets.append(offset)
                    if offset >= offset_threshold:
                        bad_samples.append((ridge_point, edt_radius))

                if len(offsets) < 3:
                    continue
                bad_fraction = sum(1 for o in offsets if o >= offset_threshold) / len(offsets)
                if bad_fraction >= min_bad_fraction:
                    candidates.append({
                        "chain": chain,
                        "chain_set": set(interior),
                        "bad_samples": bad_samples,
                    })

            if not candidates:
                break

            # Pass B -- only prune a candidate if a trusted (non-candidate)
            # node already sits near its true ridge point.
            candidate_node_union = set()
            for c in candidates:
                candidate_node_union |= c["chain_set"]
            trusted_nodes = [
                n for n in self.networkxGraph.nodes() if n not in candidate_node_union
            ]
            if not trusted_nodes:
                break  # nothing trustworthy to compare against; leave the rest alone
            trusted_coords = np.array(trusted_nodes, dtype=float) * pixel_dims
            tree = cKDTree(trusted_coords)

            for c in candidates:
                if not c["bad_samples"]:
                    continue
                redundant_hits = 0
                for ridge_point, edt_radius in c["bad_samples"]:
                    search_radius = max(
                        edt_radius * redundancy_lateral_scale, float(pixel_dims.min())
                    )
                    if tree.query_ball_point(ridge_point, r=search_radius):
                        redundant_hits += 1
                redundancy_fraction = redundant_hits / len(c["bad_samples"])
                if redundancy_fraction >= min_redundancy_fraction:
                    for node in c["chain"][:-1]:
                        if self.networkxGraph.has_node(node):
                            self.networkxGraph.remove_node(node)
                    total_removed += 1
                    changed = True

        if total_removed:
            self.nodeDegreeDict = dict(nx.degree(self.networkxGraph))
        self.runTimeDict["offCenterSpurPruning"] = round(time.time() - startTime, 3)
        self.infoDict["offCenterSpursRemoved"] = total_removed

    def _local_raw_edt_ridge(
        self, raw_segmentation, pos_phys, pixel_dims, window_um, direction, slab_half_um
    ):
        """
        Locate the true medial-axis ridge point in a thin cross-sectional disk
        (radius `window_um`, half-thickness `slab_half_um` along `direction`)
        centered at `pos_phys`: the position of maximum distance to the raw
        mask's own boundary (its local EDT), computed on a crop padded well
        beyond the disk so boundary voxels near the disk's own edge don't
        bias it.

        This replaces an earlier design that pulled toward the raw mask's
        local *centroid* in the same disk. A centroid is a reasonable center
        estimate for a wall that is still surrounding the walk on all sides,
        but is not the medial axis, and the two increasingly diverge exactly
        where centering matters most: near a bend, taper, or blunt cap.

        Also reports the raw mask's own in-plane half-width at this
        cross-section (`extent_radius`) alongside the EDT-based radius
        (`edt_radius`). The two agree while following a wall that is still
        expanding or contracting steadily -- a cylinder, or a genuine taper
        of any steepness, all the way down to its point. They diverge at a
        focal-collapse point: a rounded/bulbous cap where the nearest raw-
        mask boundary point has stopped being "the surrounding wall" and
        become "the cap's own pole up ahead" -- a smoothly curving boundary
        pulls the EDT down toward that single far point faster than the
        cross-section itself is actually narrowing. See
        `_extend_endpoints_to_raw_boundary` for how the resulting ratio is
        used to stop the walk before it overshoots past a cap's geometric
        center (the mathematically correct terminus of its medial axis --
        provably one full local radius short of a hemispherical cap's
        visible tip) while still letting a straight taper, however steep,
        reach all the way to its point.

        Returns (ridge_point_phys, edt_radius_um, extent_radius_um), or None
        if the disk has too little raw-mask support to trust.
        """
        pad_um = 0.5 * window_um
        crop_half_um = window_um + pad_um
        radius_vox = np.ceil(crop_half_um / pixel_dims).astype(int)
        center_vox = np.round(pos_phys / pixel_dims).astype(int)
        lo = np.maximum(center_vox - radius_vox, 0)
        hi = np.minimum(center_vox + radius_vox + 1, raw_segmentation.shape)
        if np.any(hi <= lo):
            return None
        crop = raw_segmentation[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]
        if not crop.any():
            return None
        # a crop with almost no real background gives distance_transform_edt
        # nothing genuine to measure distance to -- verified directly: on an
        # all-foreground array it still returns finite distances, shaped by
        # the crop's own boundary, not real anatomy. Across 40 sampled real
        # endpoints, normal vessel cross-sections left 30-90% of the crop as
        # background; a structure far wider than the search window (a real
        # vessel flaring into a support/anchor structure, or the volume's
        # own edge) left under 1%. 2% is a large, well-separated margin
        # between those two populations, not a tight cutoff.
        if crop.mean() > 0.98:
            return None
        edt_crop = distance_transform_edt(crop, sampling=pixel_dims)

        idx = np.argwhere(crop)
        phys = idx * pixel_dims + lo * pixel_dims
        rel = phys - pos_phys
        along = rel @ direction
        lateral = rel - along[:, None] * direction
        lateral_dist = np.linalg.norm(lateral, axis=1)
        in_disk = (np.abs(along) <= slab_half_um) & (lateral_dist <= window_um)
        if in_disk.sum() < 3:
            return None

        edt_vals = edt_crop[tuple(idx[in_disk].T)]
        along_in_disk = along[in_disk]
        lateral_in_disk = lateral_dist[in_disk]
        max_val = edt_vals.max()
        near_max = edt_vals >= max_val - 1e-6
        # tie-break: near any blunt boundary (flat cut or a rounding cap), a
        # whole disk of off-axis points can sit exactly as far from the
        # boundary as the true on-axis ridge point (every point within that
        # disk is equidistant from a flat cap, or from a hemispherical one's
        # pole) -- an axis-agnostic argmax can latch onto a point far off to
        # one side, and since it also steers next step's direction, that one
        # bad pick sends the whole walk spiralling off-axis. Prefer the tied
        # point closest to the search axis first; along-distance and
        # forward-preference only break finer, genuine single-voxel ties.
        cand = np.flatnonzero(near_max)
        order = np.lexsort((
            -along_in_disk[near_max],
            np.abs(along_in_disk[near_max]),
            lateral_in_disk[near_max],
        ))
        best = cand[order[0]]
        ridge_point = phys[in_disk][best]
        return ridge_point, float(edt_vals[best]), float(lateral_in_disk.max())

    def _is_locally_elongated(
        self, raw_segmentation, pos_phys, pixel_dims, direction, lateral_radius_um,
        check_half_um: float = 150.0, elongation_ratio: float = 2.0,
    ):
        """
        Distinguishes a genuine vessel cross-section, however wide, from a
        roughly round chamber: a tube continues much further along its own
        axis than across it, while a chamber is comparably sized in every
        direction. On this dataset, `extent_radius` saturating `window_um`
        (see `_local_raw_edt_ridge`) turned out to be true for the large
        majority of perfectly ordinary steps -- skeleton radius here has a
        133um dataset-wide maximum and a 42um *median*, so "the local search
        window filled up" cannot by itself distinguish a wide but normal
        vessel from a genuinely non-vessel-shaped structure (verified
        directly: sampled real endpoints saturate their search window on
        ~57% of steps). Elongation is the actual distinguishing geometry,
        and needs a crop sized independently of `window_um` to measure --
        otherwise a saturated lateral search would trivially saturate this
        check too. Verified against a real, confirmed pathological chamber
        (ratio 1.33) and 30 sampled ordinary endpoints (minimum ratio 2.06,
        median 5.7): a threshold of 2.0 separates them cleanly.

        Returns True (assume elongated -- don't block a legitimate walk) if
        there isn't enough local support to tell either way, e.g. near the
        volume's own edge.
        """
        radius_vox = np.ceil(check_half_um / pixel_dims).astype(int)
        center_vox = np.round(pos_phys / pixel_dims).astype(int)
        lo = np.maximum(center_vox - radius_vox, 0)
        hi = np.minimum(center_vox + radius_vox + 1, raw_segmentation.shape)
        if np.any(hi <= lo):
            return True
        crop = raw_segmentation[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]
        if not crop.any():
            return True
        idx = np.argwhere(crop)
        phys = idx * pixel_dims + lo * pixel_dims
        rel = phys - pos_phys
        along = rel @ direction
        lateral = rel - along[:, None] * direction
        lateral_dist = np.linalg.norm(lateral, axis=1)
        # restrict to a cylinder around the axis matching this cross-
        # section's own lateral scale, so unrelated structure elsewhere in
        # the crop can't be mistaken for this vessel continuing
        near_axis = lateral_dist <= max(lateral_radius_um, 1.0)
        if near_axis.sum() < 3:
            return True
        along_extent = along[near_axis].max() - along[near_axis].min()
        return along_extent >= elongation_ratio * 2.0 * lateral_radius_um

    def _find_trustworthy_anchor(
        self, raw_segmentation, pixel_dims, endpoint, slab_half_um,
        offset_threshold: float = 0.5, min_stable_run: int = 3, max_lookback_um: float = 150.0,
    ):
        """
        Lee's thinning algorithm reliably centers its path through a tube,
        but has no such guarantee inside a wide, non-tube-shaped region --
        confirmed directly on a real flagged endpoint sitting deep in a
        ~300um chamber: offset/radius (measured the same way as everywhere
        else in this module, via `_local_raw_edt_ridge`) was 0.61 at the
        endpoint itself and still 0.43-0.44 ten *existing* skeleton steps
        further back, well before `_extend_endpoints_to_raw_boundary`'s own
        walk ever begins. Extending or stopping precisely at that endpoint
        can't fix this -- the pre-existing path leading up to it is already
        wrong.

        Walks backward from `endpoint` through its existing chain,
        measuring each node's own offset/radius the same way. Returns
        (anchor_node, nodes_to_remove): the last trustworthy node found (a
        `min_stable_run`-long run of well-centered nodes starts there) and
        the poorly-centered nodes between it and `endpoint` (inclusive,
        ordered endpoint-first) to discard before re-walking outward from
        the anchor with the normal, already-validated extension logic --
        which then re-covers that stretch itself, ridge-corrected, and
        continues into new territory exactly as it would from any other
        endpoint. If `endpoint` is already well-centered, or no anchor is
        found within `max_lookback_um` (or before hitting a branch point or
        another endpoint), returns (endpoint, []): a no-op, so a normal,
        already-well-centered endpoint is completely unaffected.
        """
        chain = [endpoint]
        prev, cur = None, endpoint
        while True:
            neighbors = [n for n in self.networkxGraph.neighbors(cur) if n != prev]
            if not neighbors:
                break
            prev, cur = cur, neighbors[0]
            chain.append(cur)
            if self.networkxGraph.degree(cur) != 2:
                break  # reached a branch point or another endpoint
            cumulative_um = np.linalg.norm(
                (np.array(cur, dtype=float) - np.array(endpoint, dtype=float)) * pixel_dims
            )
            if cumulative_um > max_lookback_um:
                break

        offsets = []
        for i in range(len(chain) - 1):
            node = chain[i]
            behind = chain[i + 1]
            pos_phys = np.array(node, dtype=float) * pixel_dims
            behind_phys = np.array(behind, dtype=float) * pixel_dims
            direction = pos_phys - behind_phys
            norm = np.linalg.norm(direction)
            if norm < 1e-9:
                offsets.append(None)
                continue
            direction = direction / norm
            radius_here = float(self.radiusMatrix[node])
            window_um = max(6.0, 1.2 * radius_here) if radius_here > 0 else 12.0
            ridge = self._local_raw_edt_ridge(
                raw_segmentation, pos_phys, pixel_dims, window_um, direction, slab_half_um
            )
            if ridge is None:
                offsets.append(None)
                continue
            ridge_point, edt_radius, extent_radius = ridge
            if edt_radius < 1e-6:
                offsets.append(None)
                continue
            lateral = pos_phys - ridge_point
            lateral = lateral - np.dot(lateral, direction) * direction
            offsets.append(float(np.linalg.norm(lateral) / edt_radius))

        for i in range(len(offsets)):
            run = offsets[i:i + min_stable_run]
            if len(run) < min_stable_run:
                break
            if all(o is not None and o < offset_threshold for o in run):
                return chain[i], chain[:i]

        return endpoint, []

    def _extend_endpoints_to_raw_boundary(
        self,
        raw_segmentation,
        tangent_lookback: int = 5,
        max_extension_um: float = 50.0,
        lookback_fraction: float = 0.1,
        decline_fraction: float = 0.12,
        correction_blend: float = 0.3,
        max_direction_deviation_deg: float = 90.0,
    ):
        """
        Extend true skeleton endpoints outward through the raw (pre-opening)
        segmentation, following each endpoint's local vessel centerline out
        to the medial axis' own true terminus -- which is not always the
        visible mask boundary (see below).

        Mask preprocessing (see Pipeline3D.run()) trims a few voxels off every
        tapering vessel tip along with the noise it is meant to suppress, and
        widening that opening was shown to reconnect most of the noise instead
        of just recovering the tips. Since each endpoint here is already final
        (after cluster collapse and pruning), it is walked outward, in
        physical space, through the untouched raw mask, one fixed step at a
        time. At every step, `_local_raw_edt_ridge` finds the true medial-axis
        point in the cross-section just stepped into (steering the next
        step's direction) and reports two radii for that cross-section: the
        raw mask's true distance-to-boundary (`edt_radius`) and its plain
        in-plane half-width (`extent_radius`).

        For a vessel that is still a tube or a genuine taper -- however steep
        -- these two radii track each other, so the walk simply continues
        until it runs off the edge of the raw mask, same as before. But at a
        rounded/bulbous cap, the two provably diverge before the visible
        surface: past a hemispherical cap's equator, the nearest raw-mask
        boundary point stops being the surrounding wall and becomes the
        cap's own pole, so `edt_radius` collapses toward that single far
        point faster than the cross-section (`extent_radius`) is actually
        narrowing. This divergence keeps *trending downward* only while the
        boundary keeps curving in on itself; a straight taper (even one with
        a corner where it begins, which produces its own brief, non-trending
        settling dip in the ratio from simple Euclidean geometry) shows no
        such sustained trend once past that settling distance. So the walk
        compares the edt/extent ratio across two consecutive lookback windows
        (each `lookback_fraction` of the endpoint's own local radius) and
        only stops -- discarding the cross-section that triggered it -- once
        the ratio has declined by more than `decline_fraction` in *both*
        windows in a row; a one-window dip alone is not enough to stop the
        walk. This is what keeps a rounded cap's endpoint from being pushed
        past the cap's own geometric center -- the medial axis' true, and
        for a hemispherical cap provably correct, terminus -- while still
        letting every genuine taper reach all the way to its point.

        A flat, perpendicular cut produces this same declining-ratio
        signature (the nearest boundary point ahead pulls `edt_radius` down
        exactly the way a rounding cap does), but unlike a rounding cap its
        `extent_radius` stays essentially full width right up to the cut
        instead of visibly narrowing on approach -- so the stop also
        requires `extent_radius` to have measurably shrunk over the same
        window, or a flat-cut endpoint would get pulled back from its tip
        for no physical reason.

        `correction_blend` is deliberately conservative. Real, wide junction
        chambers (as opposed to a tube or a taper) have no single well-
        defined cross-sectional ridge -- several nearby points can sit
        within noise of the same maximum EDT -- so a per-step argmax can
        legitimately jump between them. A high blend chases each jump and
        the walk's direction visibly spins step to step without settling; a
        lower blend still drifts over a long walk (there is no stable target
        to converge to inside a chamber) but far more slowly, which is the
        best a purely local, single-path walk can do here short of
        detecting "I am inside a chamber, not a vessel" outright.

        Even a conservative per-step blend has no memory of where the walk
        started, though -- nothing previously stopped many small, similarly-
        biased corrections from *compounding* over the course of one walk.
        Near an ambiguous or blunt tip, exactly where the ridge search can
        legitimately keep drifting to one side step after step (the
        `correction_blend` paragraph above), that compounding can rotate
        the walk's direction past 90 degrees and even further -- confirmed
        directly on a real flagged endpoint, a small branch whose extension
        visibly hooked ~180 degrees back on itself a few steps before
        terminating, landing the endpoint well inside the vessel instead of
        at its tip. A real vessel extending outward over a mere
        `max_extension_um` budget essentially never needs to turn that
        sharply -- that would mean curving tighter than the vessel's own
        typical radius within this short a distance -- so
        `max_direction_deviation_deg` bounds cumulative rotation from the
        walk's own established outward direction, not just the per-step
        delta: once a correction would rotate past this bound, that's the
        same "local geometry is no longer trustworthy" situation this
        function already stops for elsewhere, so it stops here too, keeping
        the endpoint at the last position it actually had reliable local
        evidence for, instead of continuing into a physically implausible
        reversal.
        """
        startTime = time.time()
        pixel_dims = np.array(self.pixelDims, dtype=float)
        step_phys = float(pixel_dims.min()) / 2.0
        max_steps = int(round(max_extension_um / step_phys))
        slab_half_um = max(step_phys, float(pixel_dims.min()))
        shape = raw_segmentation.shape

        # the ridge search actively hunts for the locally thickest nearby
        # point, unlike the plain mask centroid it replaced -- near a large,
        # densely packed junction that can pull a step sideways into the
        # junction's own bulk, landing on a node that's already part of the
        # graph (whether original, or added moments ago by a *different*
        # endpoint's walk in this same loop) and splicing an unrelated
        # branch onto this one -- an incorrect topology change, not just an
        # imprecise position. `self.networkxGraph.has_node` is checked live
        # (not a set snapshotted before the loop) so it also catches nodes
        # added by earlier iterations of this very loop.
        endpoints = [n for n, d in self.networkxGraph.degree() if d == 1]
        extended = 0
        recentered = 0
        for endpoint in endpoints:
            # Lee's thinning has no centering guarantee inside a wide,
            # non-tube-shaped region (see _find_trustworthy_anchor), so an
            # endpoint can already be poorly centered before any of this
            # function's own walk begins. Cut back to the last trustworthy
            # point first, if any -- a no-op for the large majority of
            # endpoints where the existing path is already well-centered.
            anchor, nodes_to_remove = self._find_trustworthy_anchor(
                raw_segmentation, pixel_dims, endpoint, slab_half_um
            )
            # _find_trustworthy_anchor fires on any offset/radius >= 0.5,
            # from the one confirmed ~123um chamber-scale correction down
            # to a single rough-surface voxel nudging the path by a few um
            # -- both are worth recentering (Lee's thinning has no
            # centering guarantee either way), but only a *substantial*
            # cutback means the resulting short terminal segment is short
            # because we deliberately discarded real distance, rather than
            # just being an intrinsically short, thin noise spur that the
            # ordinary stub rule should still catch. Measured directly:
            # confirmed genuine chamber-scale corrections cluster at
            # 65-123um and >=1.1x the local radius; the bulk of ordinary
            # rough-surface corrections are under 20um and often under 1x
            # the local radius -- exempting everything that merely went
            # through recentering (as originally implemented) let a wave
            # of those small corrections' resulting stubs survive as if
            # they were confirmed-real, inflating short-thick terminal
            # segments 7x (6 -> 42) on this dataset.
            exemption_eligible = False
            if nodes_to_remove:
                cutback_um = float(np.linalg.norm(
                    (np.array(anchor, dtype=float) - np.array(endpoint, dtype=float)) * pixel_dims
                ))
                radius_at_anchor = float(self.radiusMatrix[anchor])
                exemption_eligible = (
                    cutback_um >= 50.0 and cutback_um >= radius_at_anchor
                )
                for node in nodes_to_remove:
                    self.networkxGraph.remove_node(node)
                endpoint = anchor
                recentered += 1

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
            # snapshot of the walk's own established outward direction,
            # before any per-step ridge correction -- see
            # max_direction_deviation_deg above
            initial_direction = direction
            min_direction_cos = float(np.cos(np.radians(max_direction_deviation_deg)))

            radius_here = float(self.radiusMatrix[endpoint])
            half_lookback = max(1, int(round(lookback_fraction * radius_here / step_phys)))
            pos_phys = end_phys.copy()
            last_voxel = endpoint
            did_extend = False
            ratio_history = []
            extent_history = []
            non_elongated_streak = 0
            for _ in range(max_steps):
                # recomputed every step from the current radius_here (not
                # fixed to the endpoint's starting radius) so the search
                # disk grows if the walk enters a genuinely wider structure
                # -- e.g. a real vessel flaring into a support/anchor
                # structure or the volume's own edge, both observed in
                # practice at a scale far beyond any normal vessel cross-
                # section in this dataset. Recomputing every step (rather
                # than only ever growing) tracks real anatomy more tightly
                # in the overwhelmingly common case where the disk stays at
                # ordinary vessel scale; the trade-off is that very close to
                # a rounding cap, where radius_here is dropping quickly,
                # this can occasionally interrupt the sustained-decline
                # pattern `_extend_endpoints_to_raw_boundary` looks for.
                window_um = max(6.0, 1.2 * radius_here)
                new_pos = pos_phys + step_phys * direction
                voxel = tuple(int(round(v)) for v in new_pos / pixel_dims)
                if any(v < 0 or v >= s for v, s in zip(voxel, shape)):
                    break
                if not raw_segmentation[voxel]:
                    break
                if voxel != last_voxel and self.networkxGraph.has_node(voxel):
                    break  # never splice back into already-existing structure

                ridge = self._local_raw_edt_ridge(
                    raw_segmentation, new_pos, pixel_dims, window_um, direction, slab_half_um
                )
                if ridge is None:
                    # no trustworthy local geometry (see _local_raw_edt_ridge
                    # -- most notably a crop with almost no real background,
                    # meaning this structure is far wider than any normal
                    # vessel and the search has lost the ability to see a
                    # true boundary in any direction). Previously this fell
                    # back to a plain, uncorrected forward step and kept
                    # going for the rest of the budget -- which, on a real
                    # oversized structure (a vessel flaring into a support
                    # structure, or the volume's own edge), just walks
                    # blindly deep into it with no way to tell "centered"
                    # from "wherever we happened to end up". Stopping here
                    # instead keeps the endpoint at the last position we
                    # actually had reliable local evidence for.
                    break
                ridge_point, edt_radius, extent_radius = ridge
                if extent_radius >= 0.98 * window_um:
                    # the lateral search has hit its own search radius --
                    # on this dataset that alone happens on the large
                    # majority of perfectly ordinary steps (median
                    # skeleton radius here is 42um, up to 133um), so it
                    # cannot by itself mean anything is wrong. But it does
                    # mean we should confirm this is still a genuine,
                    # elongated vessel before trusting edt_radius/extent
                    # further -- a real chamber (confirmed on this
                    # dataset: nearly cube-shaped, ~290x297x359um, close
                    # to the volume's own edge) saturates in every
                    # direction at once, not just laterally. A real,
                    # merely-curving vessel can occasionally look
                    # under-elongated over one 150um straight-line window
                    # too (measured directly: ~20% of individual saturated
                    # steps on this dataset), so -- like focal-collapse --
                    # this requires a *sustained* run of failures, not one,
                    # or the compounding per-step false-positive rate over
                    # a long walk would stop the large majority of walks
                    # prematurely (measured: 48% of endpoints on a single-
                    # failure trigger, almost all spurious).
                    if self._is_locally_elongated(
                        raw_segmentation, new_pos, pixel_dims, direction, extent_radius
                    ):
                        non_elongated_streak = 0
                    else:
                        non_elongated_streak += 1
                        if non_elongated_streak >= 3:
                            break  # not vessel-shaped here; stop at the last trusted position
                else:
                    non_elongated_streak = 0
                ratio = edt_radius / extent_radius if extent_radius > 1e-9 else 1.0
                if len(ratio_history) >= 2 * half_lookback:
                    old_ratio = ratio_history[-2 * half_lookback]
                    mid_ratio = ratio_history[-half_lookback]
                    ratio_declining = (
                        old_ratio > 1e-9
                        and mid_ratio < (1.0 - decline_fraction) * old_ratio
                        and mid_ratio > 1e-9
                        and ratio < (1.0 - decline_fraction) * mid_ratio
                    )
                    # a flat, perpendicular cut produces the same kind of
                    # declining edt/extent ratio as a rounding cap (the
                    # nearest boundary point ahead pulls edt_radius down
                    # exactly the same way) but the cross-section itself
                    # stays essentially full width right up to the cut,
                    # unlike a rounding cap's, which visibly narrows on
                    # approach -- so also require the mask's own in-plane
                    # extent to have measurably shrunk before treating
                    # this as focal collapse, or a flat-cut endpoint gets
                    # pulled back from its tip for no physical reason
                    extent_old = extent_history[-2 * half_lookback]
                    extent_shrinking = (
                        extent_old > 1e-9 and extent_radius < 0.9 * extent_old
                    )
                    if ratio_declining and extent_shrinking:
                        break  # focal collapse: discard this cross-section
                ratio_history.append(ratio)
                extent_history.append(extent_radius)
                radius_here = edt_radius
                lateral = ridge_point - new_pos
                lateral = lateral - np.dot(lateral, direction) * direction
                lateral_norm = np.linalg.norm(lateral)
                if lateral_norm > 1e-9:
                    new_direction = direction + correction_blend * (lateral / lateral_norm)
                    new_direction_norm = np.linalg.norm(new_direction)
                    if new_direction_norm > 1e-9:
                        new_direction = new_direction / new_direction_norm
                        # Bound cumulative rotation from the walk's own
                        # outward direction (see max_direction_deviation_deg
                        # docstring above). This step's own ridge point is
                        # what's pulling the direction implausibly far, so
                        # -- same as the focal-collapse and no-ridge breaks
                        # above -- discard this cross-section too rather
                        # than committing it: it's the first position whose
                        # own local geometry is no longer trustworthy, not
                        # just the last position before one that would be.
                        if float(np.dot(new_direction, initial_direction)) < min_direction_cos:
                            break
                        direction = new_direction

                pos_phys = new_pos
                if voxel != last_voxel:
                    self.networkxGraph.add_edge(last_voxel, voxel)
                    self.radiusMatrix[voxel] = radius_here
                    last_voxel = voxel
                    did_extend = True
            if did_extend:
                extended += 1
            if exemption_eligible:
                # whatever node this walk actually ended on -- last_voxel
                # if it managed to extend further from the anchor, or the
                # anchor itself if the very next cross-section was already
                # untrustworthy -- is a deliberately truncated, already-
                # verified-centered terminus, not a raw skeletonization
                # artifact; protect it from the short-stub-relative-to-
                # diameter noise rule downstream in Filament. Only earned
                # by a substantial cutback (see above) -- a trivial
                # recentering nudge gets no such protection and is judged
                # by the ordinary rule like any other terminal segment.
                self.recenteredEndpoints.add(last_voxel)

        if extended or recentered:
            self.nodeDegreeDict = dict(nx.degree(self.networkxGraph))
        self.runTimeDict["endpointExtension"] = round(time.time() - startTime, 3)
        self.infoDict["endpointsExtended"] = extended
        self.infoDict["endpointsRecentered"] = recentered

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
