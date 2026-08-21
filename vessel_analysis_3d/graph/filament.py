import numpy as np
import time
from collections import defaultdict
import dask.array as da
import os
import math
from scipy import interpolate
from geomdl import knotvector

from .measurement import (
    getLength,
    getVolumeCylinder,
    getSurfaceAreaCylinder,
    getVolume,
    getRadius,
    getRadiusMinMax,
    get_z_angle,
    touchesVolumeBoundary,
)


class Filament:
    """
    Find statistics on a graph of a skeleton

    Parameters
    ----------
    graph : dictionary of lists (adjacency list) with key as node and value = list with
            its neighboring nodes

    start : list of coordinates
        beginning point for DFS (must be an end point)
        3D: [z, y, x]   2D: [y, x]

    skelRadii : array containing the distance to the closest background point (=radius)

    pixelDimensions : list of pixel dimensions [z, y, x]

    lengthLimit : float
        minimum length (all branches below this length will be removed)

    Notes
    --------
    straightness = curveDisplacement / curveLength
    """

    def __init__(
        self,
        graph,
        start,
        skelRadii,
        pixelDimensions,
        lengthLimit,
        diaScale,
        branchingThreshold,
        expFlag,
        smallRAMmode,
        fileName,
        removeBorderEndPts,
        removeEndPtsFromSmallFilaments,
        interpolate,
        cut_neighbor_brpt_segs,
        minBranchLengthUm,
        minBranchRadiusUm,
        minJunctionRadiusUm,
        recenteredEndpoints=frozenset(),
    ):
        self.graph = graph
        self.start = start
        self.skelRadii = skelRadii
        self.pixelDims = pixelDimensions
        self.lengthLim = lengthLimit
        self.diaScale = diaScale
        self.branchingThr = branchingThreshold
        self.expFlag = expFlag
        self.smallRAMmode = smallRAMmode
        self.fileName = fileName
        self.removeBorderEndPts = removeBorderEndPts
        self.removeEndPtsFromSmallFilaments = removeEndPtsFromSmallFilaments
        self.interpolate = interpolate
        self.cut_neighbor_brpt_segs = cut_neighbor_brpt_segs
        self.minBranchLengthUm = minBranchLengthUm
        self.minBranchRadiusUm = minBranchRadiusUm
        self.minJunctionRadiusUm = minJunctionRadiusUm
        self.recenteredEndpoints = recenteredEndpoints
        self.endPtsList = []
        # true endpoints excluded from endPtsList by removeBorderEndPts
        # (touch the volume boundary, so likely a field-of-view crop
        # artifact rather than a real vessel terminus -- kept here, not
        # discarded, so a viewer can show them separately without having to
        # recompute anything)
        self.borderEndPts = []
        self.brPtsDict = {}
        self.segmentsDict = {}
        self._predDict = {}
        self.compTime = 0
        self.postprocessTime = 0
        self.postprocBranches = 0
        self.postprocProtectedBridges = 0
        self.postprocEndPts = 0
        # dictionary containing all segment statistics
        self.segmentStats = defaultdict(dict)
        # per-segment diameter profile: key -> list of
        # (position_um, diameter_um, z, y, x) along the segment's real,
        # possibly-curved skeleton path (not the aggregate mean/min/max in
        # segmentStats). Always kept in sync with segmentStats: both are set
        # together in _setSegStats and removed together in _removeSegments.
        self.segmentProfiles = {}

    def dfs_iterative(self):
        """
        Iterate over the graph in a depth-first-search. If a branch or end point is
        found, retrieve the segment and calculate its statistics
        """
        startTime = time.time()
        visited, stack = set(), [self.start]
        while stack:
            vertex = stack.pop()
            if vertex not in visited:
                visited.add(vertex)
                for neighbor in self.graph[vertex]:
                    if neighbor not in visited:
                        self._predDict[neighbor] = vertex
                        stack.append(neighbor)
                    elif (
                        neighbor in visited and not self._predDict[vertex] == neighbor
                    ):  # cycle found
                        if len(self.graph[neighbor]) > 2:  # neighbor is branch point
                            oldPred = self._predDict[neighbor]
                            self._predDict[neighbor] = (
                                vertex  # change predecessor to get segment of cycle
                            )
                            segment = self._getSegment(neighbor)
                            self._predDict[neighbor] = (
                                oldPred  # change back to old predecessor
                            )
                            self._setSegStats(segment, interpolate=self.interpolate)
                if len(self.graph[vertex]) == 1:  # end point found
                    self.endPtsList.append(vertex)
                    if vertex != self.start:
                        segment = self._getSegment(vertex)
                        self._setSegStats(segment, interpolate=self.interpolate)
                elif len(self.graph[vertex]) > 2:  # branch point found
                    self.brPtsDict[vertex] = len(self.graph[vertex])
                    segment = self._getSegment(vertex)
                    self._setSegStats(segment, interpolate=self.interpolate)
        self.compTime = time.time() - startTime

        # postprocessing
        postprocStart = time.time()
        self._removeSmallAndSegmentsBelowDiameterLengthRatio(
            self.cut_neighbor_brpt_segs
        )

        if self.removeBorderEndPts:
            # remove image border points from end points list
            self._removeBorderPtsFromEndPts()

        if self.removeEndPtsFromSmallFilaments:
            if len(self.segmentsDict) < 5:
                self.endPtsList = []

        self.postprocessTime = time.time() - postprocStart

        # add number of terminal and branch points to segments dictionary
        for seg in self.segmentStats:
            self.segmentStats[seg]["terminal Points"] = 0
            self.segmentStats[seg]["branching Points"] = 0
            if seg[0] in self.endPtsList:
                self.segmentStats[seg]["terminal Points"] += 1
            if seg[0] in self.brPtsDict.keys():
                self.segmentStats[seg]["branching Points"] += 1
            if seg[1] in self.endPtsList:
                self.segmentStats[seg]["terminal Points"] += 1
            if seg[1] in self.brPtsDict.keys():
                self.segmentStats[seg]["branching Points"] += 1

    ##############################
    # ## Segment Interpolation ###
    ##############################
    # Build interpolation delta
    def _delta_calc(self, num_verts, vis_radius):
        # base = 3 if num_verts > 50 else 2
        delta = max(3, math.ceil(num_verts / math.log(num_verts, 2)))
        if num_verts > 100 or (vis_radius > 3 and num_verts > 20):
            delta = int(delta / 2)
        return delta

    def _seg_interpolate(self, point_coords, vis_radius):
        # Find basis-spline (BSpline) of points to smooth jaggedness of skeleton.
        # The resources I used to learn about BSplines can be examined below:
        # https://web.mit.edu/hyperbook/Patrikalakis-Maekawa-Cho/node17.html
        # http://learnwebgl.brown37.net/07_cameras/points_along_a_path.html
        # http://www.independent-software.com/determining-coordinates-on-a-html-canvas-bezier-curve.html

        # Set appropriate degree of our BSpline
        num_verts = point_coords.shape[0]
        if num_verts > 4:
            spline_degree = 3
        else:
            spline_degree = max(1, num_verts - 1)

        # The optimal number of interpolated segment points for visualization was
        # determined emperically as a trade-off value between ground-truth length
        # and computational costs.
        delta = self._delta_calc(num_verts, vis_radius)

        # Find the segment length based on our cubic BSpline.
        # https://github.com/kawache/Python-B-spline-examples
        u = np.linspace(0, 1, delta, endpoint=True)  # U

        # Scipy knot vector format
        knots = knotvector.generate(spline_degree, num_verts)  # Knotvector
        tck = [
            knots,
            [point_coords[:, 0], point_coords[:, 1], point_coords[:, 2]],
            spline_degree,
        ]

        coords_list = np.array(interpolate.splev(u, tck)).T
        return coords_list

    def _getSegment(self, node):
        """
        Find the segment of a branch or end node by iterating over its predecessors

        Parameters
        ----------
        node : list of node coordinates
            3D: [z, y, x]   2D: [y, x]

        Returns
        -------
        segmentList : list of nodes in the segment
        """
        segmentList = [node]
        while True:
            node = self._predDict.get(node)
            if node is None:
                # may happen due to postprocessing removing predecessors of old
                # branching points
                return None
            segmentList.insert(0, node)
            if len(self.graph[node]) == 1 or len(self.graph[node]) > 2:
                break
        return segmentList

    def _getBranchingDegree(self, segment, factor=0.25):
        """
        Calculates the branching degree of a segment

        Parameters
        ----------
        segment : list
            list of nodes in the segment

        factor : float
            length factor used for calculating the vectors of the segments

        Returns
        -------
        degree : float
            degree between the vectors of the segment and it's predecessor
        """
        segPredList = self._getSegment(segment[0])
        if segPredList is None:
            return "Null"
        segPredList.reverse()
        # take neighboring points from branching point
        if factor <= 0:
            segPredPt = segPredList[1]
            segPt = segment[1]
        # take half of the segments from branching point
        elif factor == 0.5:
            segPredPt = segPredList[int(len(segPredList) * factor)]
            segPt = segment[int(len(segment) * factor)]
        # take the whole segment from branching point
        elif factor >= 1:
            segPredPt = segPredList[len(segPredList) - 1]
            if (
                segment[len(segment) - 1] == segment[0]
            ):  # in case of a circle, take pre-last point
                segPt = segment[len(segment) - 2]
            else:
                segPt = segment[len(segment) - 1]
        # take points according to the factor of the segment lengths
        else:
            # determine point of predecessor
            if round(len(segPredList) * factor) == 0 or len(segPredList) == 2:
                segPredPt = segPredList[1]
            else:
                segPredPt = segPredList[round(len(segPredList) * factor)]
            # determine point of segment
            if round(len(segment) * factor) == 0 or len(segment) == 2:
                segPt = segment[1]
            else:
                if (
                    segment[round(len(segment) * factor)] == segment[0]
                ):  # in case of a circle, take pre-last point
                    segPt = segment[len(segment) - 2]
                else:
                    segPt = segment[round(len(segment) * factor)]
        segPredVect = [j - i for i, j in zip(segPredPt, segment[0])]
        segPredVect = [a * b for a, b in zip(segPredVect, self.pixelDims)]
        segVect = [j - i for i, j in zip(segment[0], segPt)]
        segVect = [a * b for a, b in zip(segVect, self.pixelDims)]

        segPredNorm = np.linalg.norm(segPredVect)
        segNorm = np.linalg.norm(segVect)
        if segPredNorm <= 1e-12 or segNorm <= 1e-12:
            return 0.0

        cosine_angle = np.dot(segPredVect, segVect) / (segPredNorm * segNorm)
        cosine_angle = float(np.clip(cosine_angle, -1.0, 1.0))
        angle = np.arccos(round(cosine_angle, 4))

        return round(np.degrees(angle), 4)

    def _computeDiameterProfile(self, segment, radiusMat):
        """
        Diameter at every skeleton voxel along a segment's real (possibly
        curved) path, with cumulative physical distance from the segment's
        first node -- unlike the mean/min/max in segmentStats, this is not
        aggregated away, so a viewer can show diameter at a chosen spacing
        along the actual path instead of one value for the whole segment.

        Returns a list of (position_um, diameter_um, z, y, x) tuples, one per
        node in `segment`, in order.
        """
        profile = []
        cumDist = 0.0
        prevPt = None
        for pt in segment:
            if prevPt is not None:
                vect = [
                    (a - b) * d for a, b, d in zip(pt, prevPt, self.pixelDims)
                ]
                cumDist += float(np.linalg.norm(vect))
            diameter = 2.0 * float(radiusMat[pt])
            profile.append((cumDist, diameter, int(pt[0]), int(pt[1]), int(pt[2])))
            prevPt = pt
        return profile

    def _setSegStats(self, segment, interpolate):
        """
        Sets the statistics for a segment

        Parameters
        ----------
        segment : list
            list of nodes in the segment
        """
        self.segmentsDict[segment[0], segment[len(segment) - 1]] = segment
        if self.smallRAMmode:
            radius = getRadius(
                da.from_zarr(
                    "tmp_zarr" + os.sep + self.fileName + "_radiusMatrix.zarr"
                ),
                segment,
            )
        else:
            radius = getRadius(self.skelRadii, segment)

        # min/max radius across the segment's own skeleton nodes (not the
        # interpolated/smoothed path) -- same basis as the mean radius above
        if self.smallRAMmode:
            minRadius, maxRadius = getRadiusMinMax(
                da.from_zarr(
                    "tmp_zarr" + os.sep + self.fileName + "_radiusMatrix.zarr"
                ),
                segment,
            )
        else:
            minRadius, maxRadius = getRadiusMinMax(self.skelRadii, segment)

        # Placeholder here -- whether this segment is boundary-clipped is
        # decided later, globally, by GraphObj._flag_boundary_clipped_segments
        # (core.py). That check needs the raw mask and every filament's
        # final segments at once, neither of which is available yet at
        # this per-filament, per-segment point in the pipeline. An earlier
        # version checked only this segment's own skeleton-local radius
        # against its distance to the nearest face, but that misses a
        # vessel running roughly parallel to a face: its skeleton simply
        # recenters within whatever cross-section is visible and never
        # reads as "reaching" the boundary, even though the raw mask
        # plainly shows it being cut off along its side, not just its tip.
        boundaryClipped = False

        # full per-voxel diameter profile along the segment's real skeleton
        # path (see self.segmentProfiles docstring in __init__)
        if self.smallRAMmode:
            radiusMatForProfile = da.from_zarr(
                "tmp_zarr" + os.sep + self.fileName + "_radiusMatrix.zarr"
            )
        else:
            radiusMatForProfile = self.skelRadii
        self.segmentProfiles[segment[0], segment[len(segment) - 1]] = (
            self._computeDiameterProfile(segment, radiusMatForProfile)
        )

        if interpolate:
            coords = self._seg_interpolate(np.asarray(segment), radius)
        else:
            coords = segment

        # calculate straightness
        vect = [j - i for i, j in zip(segment[0], segment[len(segment) - 1])]
        vect = [
            a * b for a, b in zip(vect, self.pixelDims)
        ]  # multiply pixel length with pixel dimension
        curveDisplacement = np.linalg.norm(vect)

        # calculate segment length. Always measured along the real, raw
        # skeleton path (`segment`), never the interpolated/smoothed
        # B-spline curve (`coords`, still computed above for the
        # smallRAMmode volume calc below) -- a B-spline fit through a
        # decimated set of control points legitimately cuts corners at
        # sharp bends, understating true length by 20-30% on real
        # high-tortuosity segments (confirmed directly on segmentation_v30).
        # This feeds straightness/volume/surfaceArea too, so all of them
        # now reflect the vessel's actual physical length, not a smoothed
        # approximation of it.
        segLength = getLength(segment, self.pixelDims)
        if segLength <= 0:
            straightness = 1.0
        else:
            straightness = curveDisplacement / segLength

        # calculate segment volume
        if interpolate:
            if self.smallRAMmode:
                volume = getVolume(
                    da.from_zarr(
                        "tmp_zarr" + os.sep + self.fileName + "_radiusMatrix.zarr"
                    ),
                    coords,
                    self.pixelDims,
                )
            else:
                volume = getVolumeCylinder(radius, segLength)
        else:
            volume = getVolumeCylinder(radius, segLength)

        # calculate segment surface area (lateral surface of the cylinder
        # approximation, same model as volume above)
        surfaceArea = getSurfaceAreaCylinder(radius, segLength)

        # calculate branching degree
        branchingDegree = self._getBranchingDegree(segment, self.branchingThr)

        # fill dictionary for csv file containing all segment statistics
        self.segmentStats[segment[0], segment[len(segment) - 1]]["diameter"] = (
            radius * 2
        )
        self.segmentStats[segment[0], segment[len(segment) - 1]]["minDiameter"] = (
            minRadius * 2
        )
        self.segmentStats[segment[0], segment[len(segment) - 1]]["maxDiameter"] = (
            maxRadius * 2
        )
        self.segmentStats[segment[0], segment[len(segment) - 1]]["straightness"] = (
            straightness
        )
        self.segmentStats[segment[0], segment[len(segment) - 1]]["length"] = segLength
        self.segmentStats[segment[0], segment[len(segment) - 1]]["volume"] = volume
        self.segmentStats[segment[0], segment[len(segment) - 1]][
            "surfaceArea"
        ] = surfaceArea
        self.segmentStats[segment[0], segment[len(segment) - 1]][
            "branchingAngle"
        ] = branchingDegree
        self.segmentStats[segment[0], segment[len(segment) - 1]][
            "boundaryClipped"
        ] = boundaryClipped

        # experimental statistics
        if self.expFlag == 1:
            zAngle = get_z_angle(segment, self.pixelDims)
            self.segmentStats[segment[0], segment[len(segment) - 1]]["zAngle"] = zAngle

    def _removeBorderPtsFromEndPts(self):
        """
        Removes end points whose own local vessel radius already reaches a
        face of the imaged volume from the end points list.

        Uses touchesVolumeBoundary: an endpoint is border-excluded if the
        vessel is locally wider than its distance to the nearest volume
        face, meaning the true cross-section almost certainly continues
        past the crop edge -- this is a field-of-view artifact, not a
        real anatomical terminus. This single-point radius-vs-distance
        test is reliable specifically *for endpoints*, unlike for a
        whole segment: _extend_endpoints_to_raw_boundary already pushes
        every endpoint as far as the raw mask allows, so an endpoint
        that's still short of the volume face by less than its own
        radius really does mean "the mask ends here, not the vessel".
        The equivalent per-segment "boundaryClipped" flag (see
        GraphObj._flag_boundary_clipped_segments in core.py) needs a
        different, mask-attribution-based test instead, because a
        vessel merely running close to a face never has any single
        skeleton point read as "reaching" it that way -- see that
        method's docstring for why.

        CHANGELOG: originally (2026-08-18) this only caught an endpoint
        touching the *exact* boundary voxel, so a vessel truncated near
        the crop edge whose segmentation fell just short of literally
        reaching it (mask-opening erosion, signal falloff) was
        indistinguishable from a genuine anatomical endpoint -- in
        practice this meant the border-excluded-endpoints viewer layer
        came back essentially empty on real data. Widened 2026-08-20 to
        the physical radius-vs-distance test above, which catches that
        case directly and needs no extra tunable margin.

        KNOWN LIMITATION (2026-08-20, still open): a thick, short vessel
        cut off by the volume boundary can sometimes get no endpoint here
        to even classify, because Lee's thinning never produced any
        skeleton branch into it at all (confirmed directly: zero raw
        skeleton voxels near several such regions, despite the raw mask
        being fully solid there, at 16.6um -- 2 voxels -- from the Z
        edge). This function has nothing to work with in that case; the
        gap is upstream of endpoint classification entirely. Fixing it
        would mean detecting large solid regions touching the boundary
        with no nearby skeleton and synthesizing a stub endpoint there
        (which could then flow through this same border-endpoint path),
        rather than anything this function itself could do -- a real new
        capability, scoped but deliberately not built yet.
        """
        if self.smallRAMmode:
            distTrans = da.from_zarr(
                "tmp_zarr" + os.sep + self.fileName + "_radiusMatrix.zarr"
            )
        else:
            distTrans = self.skelRadii
        endPtsToRemove = [
            endPt
            for endPt in self.endPtsList
            if touchesVolumeBoundary(
                [endPt], distTrans, self.pixelDims, self.skelRadii.shape
            )
        ]
        self.postprocEndPts += len(endPtsToRemove)
        for endPt in endPtsToRemove:
            self.endPtsList.remove(endPt)
        self.borderEndPts.extend(endPtsToRemove)

    def _deletePath(self, path):
        for i in range(len(path) - 1):
            self._removeEdge(path[i], path[i + 1])

    def _removeEdge(self, u, v):
        self.graph.get(u).remove(v)
        self.graph.get(v).remove(u)

    def _is_bridge_between_substantial_structure(self, path):
        """
        Check whether removing this segment would sever two otherwise-
        independent, non-trivial parts of the network apart, as opposed to
        simply pruning a genuine dead-end leaf (the normal, intended case).

        A segment's own interior nodes are private to it (degree 2, not
        shared with anything else), so removing them is always safe in
        isolation -- the only question is its two endpoint nodes. If either
        endpoint has no other connections left once this segment's edges are
        gone, that endpoint (and everything beyond it) simply *was* this
        segment, and removing it is the normal leaf-pruning outcome. But if
        BOTH endpoints still have other structure attached, and there is no
        alternate path between them, this segment is the sole bridge holding
        two real sub-trees together -- removing it would silently orphan
        one of them (this is what "length <= lengthLim" and similar
        thin/short heuristics have no way of noticing on their own, since
        they only look at the segment itself, never at what it connects).

        Returns True if removal is unsafe (a real bridge), False if safe.
        """
        u, v = path[0], path[-1]
        if u == v:
            return False  # a closed loop; removing it only opens the cycle

        edges = list(zip(path[:-1], path[1:]))
        for a, b in edges:
            if b in self.graph.get(a, []):
                self.graph[a].remove(b)
            if a in self.graph.get(b, []):
                self.graph[b].remove(a)
        try:
            if not self.graph.get(u) or not self.graph.get(v):
                return False  # one side has nothing else -- a genuine leaf tip
            visited, stack = {u}, [u]
            while stack:
                node = stack.pop()
                if node == v:
                    return False  # still connected via another path -- safe
                for neighbor in self.graph.get(node, []):
                    if neighbor not in visited:
                        visited.add(neighbor)
                        stack.append(neighbor)
            return True  # both sides have other structure and no other path
        finally:
            for a, b in edges:
                self.graph.setdefault(a, [])
                self.graph.setdefault(b, [])
                if b not in self.graph[a]:
                    self.graph[a].append(b)
                if a not in self.graph[b]:
                    self.graph[b].append(a)

    def _removeSegments(self, keys):
        brPtCandidates = set()
        actuallyRemoved = 0
        for key in keys:
            path = self.segmentsDict[key]
            if self._is_bridge_between_substantial_structure(path):
                self.postprocProtectedBridges += 1
                continue
            actuallyRemoved += 1
            self._deletePath(path)
            del self.segmentsDict[key]
            del self.segmentStats[key]
            del self.segmentProfiles[key]
            if key[0] in self.endPtsList:
                self.endPtsList.remove(key[0])
            if key[1] in self.endPtsList:
                self.endPtsList.remove(key[1])
            # add branch points to possible deletable candidates
            if key[0] in self.brPtsDict:
                brPtCandidates.add(key[0])
            if key[1] in self.brPtsDict:
                brPtCandidates.add(key[1])
        self.postprocBranches = actuallyRemoved
        # check if branch points still remain branch points after postprocessing
        for brPt in brPtCandidates:
            # branch point becomes normal point connecting two segments together
            if len(self.graph[brPt]) == 2:
                del self.brPtsDict[brPt]
                # find both segments connected by the branch pt
                segments = [
                    v
                    for k, v in self.segmentsDict.items()
                    if k[0] == brPt or k[1] == brPt
                ]
                # delete old segments from segment dictionaries (either one
                # segment if circle otherwise 2 segments)
                if len(segments) > 0:
                    segKey1 = (segments[0][0], segments[0][-1])
                    del self.segmentsDict[segKey1]
                    del self.segmentStats[segKey1]
                    del self.segmentProfiles[segKey1]
                    if (
                        len(segments) != 1
                    ):  # if segment is not a circle delete second segment from
                        # dictionaries
                        segKey2 = (segments[1][0], segments[1][-1])
                        del self.segmentsDict[segKey2]
                        del self.segmentStats[segKey2]
                        del self.segmentProfiles[segKey2]
                        # combine both segments to one segment and calculate its
                        # statistics
                        if segments[0][-1] == brPt and segments[1][0] == brPt:
                            combSegments = segments[0] + segments[1][1:]
                        elif segments[0][0] == brPt and segments[1][-1] == brPt:
                            combSegments = segments[1] + segments[0][1:]
                        # in this case the predecessor of the brPt has been deleted
                        elif segments[0][0] == brPt and segments[1][0] == brPt:
                            combSegments = segments[0][::-1] + segments[1][1:]
                        elif segments[0][-1] == brPt and segments[1][-1] == brPt:
                            combSegments = segments[0][:-1] + segments[1][::-1]
                        self._setSegStats(combSegments, interpolate=self.interpolate)
            # branch point becomes end point
            elif len(self.graph[brPt]) == 1:
                del self.brPtsDict[brPt]
                self.endPtsList.append(brPt)
            # all branches of a branch point were removed => delete branch point from
            # dict
            elif len(self.graph[brPt]) == 0:
                del self.brPtsDict[brPt]

    def _removeSmallAndSegmentsBelowDiameterLengthRatio(self, cut_neighbor_brpt_segs):
        """
        Removes all branches with a length below a specified limit and length below the
        scaled diameter and removes their statistics from the dictionaries. After branch
        removal branch points are reassigned (either they remain branch point, become
        normal point or end point) and statistics of new segments are recalculated.
        """
        # find segments with endpoint(s) in lengthDict which are below length/diameter
        # ratio
        keysToRemoveList = []
        for segKey in self.segmentStats:
            seg_length = self.segmentStats[segKey]["length"]
            seg_diameter = self.segmentStats[segKey]["diameter"]
            seg_is_terminal = segKey[0] in self.endPtsList or segKey[1] in self.endPtsList
            is_short_by_length = seg_length <= self.lengthLim
            is_short_by_branch_rule = seg_length < self.minBranchLengthUm
            is_thin_by_diameter = seg_diameter < self.minBranchRadiusUm * 2
            # a segment ending at a deliberately re-centered endpoint (see
            # GraphObj._find_trustworthy_anchor) was intentionally truncated
            # right where its own local geometry stopped being trustworthy
            # -- that's often exactly why it's short relative to its own
            # diameter, not evidence of a raw skeletonization artifact,
            # which is what this specific rule exists to catch. The other
            # weak-branch rules (absolute shortness, thin-and-short) still
            # apply -- this only exempts the diameter-relative one.
            is_recentered = (
                segKey[0] in self.recenteredEndpoints
                or segKey[1] in self.recenteredEndpoints
            )
            is_short_stub = (
                seg_length < self.diaScale * seg_diameter
                and seg_is_terminal
                and not is_recentered
            )

            if cut_neighbor_brpt_segs:
                if (
                    is_short_stub
                    or is_short_by_length
                    or (is_short_by_branch_rule and is_thin_by_diameter)
                ):
                    keysToRemoveList.append(segKey)
            # dont cut segments which are 2 branching points with length below pixel
            # dimension
            else:
                if not (
                    segKey[0] in self.brPtsDict
                    and segKey[1] in self.brPtsDict
                    and is_short_by_length
                ):
                    if (
                        is_short_stub
                        or is_short_by_length
                        or (is_short_by_branch_rule and is_thin_by_diameter)
                    ):
                        keysToRemoveList.append(segKey)
        # _removeSegments sets self.postprocBranches itself, to the count
        # actually removed (a candidate is skipped, not removed, if it's the
        # sole bridge between two otherwise-independent parts of the tree --
        # see _is_bridge_between_substantial_structure)
        self._removeSegments(keysToRemoveList)
