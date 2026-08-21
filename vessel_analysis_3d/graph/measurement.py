import numpy as np
import math


def getLength(path, dimensions):
    """
    Find length of a path as distance between nodes in it

    Parameters
    ----------
    path : list
        list of nodes in the path

    dimensions : list
        list with pixel dimensions in desired unit (e.g. microns)
        3D: [z, y, x]   2D: [y, x]

    Returns
    -------
    length : float
        Length of path
    """
    length = 0
    for index, item in enumerate(path):
        if index + 1 != len(path):
            item2 = path[index + 1]
        vect = [j - i for i, j in zip(item, item2)]
        vect = [
            a * b for a, b in zip(vect, dimensions)
        ]  # multiply pixel length with original length
        length += np.linalg.norm(vect)
    return length


def getRadius(distTrans, segment):
    if len(segment) == 0:
        return 0.0
    sumRadii = 0
    for skelPt in segment:
        sumRadii += float(distTrans[skelPt])
    radius = sumRadii / len(segment)
    if not np.isfinite(radius):
        return 0.0
    return radius


def getRadiusMinMax(distTrans, segment):
    if len(segment) == 0:
        return 0.0, 0.0
    radii = [float(distTrans[skelPt]) for skelPt in segment]
    radii = [r for r in radii if np.isfinite(r)]
    if not radii:
        return 0.0, 0.0
    return min(radii), max(radii)


def getVolumeCylinder(radius, segLength):
    return math.pi * radius**2 * segLength


def getSurfaceAreaCylinder(radius, segLength):
    return 2 * math.pi * radius * segLength


def getVolume(skelRadii, segment, dimensions):
    volume = 0
    for index, skelPt in enumerate(segment):
        if index + 1 != len(segment):
            vect = [j - i for i, j in zip(skelPt, segment[index + 1])]
            vect = [
                a * b for a, b in zip(vect, dimensions)
            ]  # multiply with pixel dimensions
            rad = skelRadii[int(skelPt[0]), int(skelPt[1]), int(skelPt[2])]
            volume += math.pi * rad**2 * np.linalg.norm(vect)
    return volume


def touchesVolumeBoundary(segment, distTrans, dimensions, volumeShape):
    """
    True if any point of the segment lies close enough to a face of the
    imaged volume that the vessel's own local radius there already
    reaches past that edge -- i.e. the true cross-section is wider than
    what was captured, not merely that the skeleton happens to run near
    the edge. This is the same physical test whether it's applied to a
    whole segment (does this vessel get clipped along its length, e.g.
    running roughly parallel to the boundary) or to a single endpoint
    (is this a real terminus or a field-of-view crop artifact).
    """
    for skelPt in segment:
        radius = float(distTrans[skelPt])
        if not np.isfinite(radius):
            continue
        boundaryDist = min(
            min(coord, extent - 1 - coord) * dim
            for coord, extent, dim in zip(skelPt, volumeShape, dimensions)
        )
        if boundaryDist < radius:
            return True
    return False


def get_z_angle(segment, pixelDims):
    zVector = [1, 0, 0]
    v1 = segment[0]

    if (
        segment[len(segment) - 1] == segment[0]
    ):  # in case of a circle, take pre-last point
        v2 = segment[len(segment) - 2]
    else:
        v2 = segment[len(segment) - 1]

    dist_v1_z = np.linalg.norm(v1[1:])
    dist_v2_z = np.linalg.norm(v2[1:])

    if dist_v1_z < dist_v2_z:
        segVector = [j - i for i, j in zip(v1, v2)]  # v2-v1
    else:
        segVector = [j - i for i, j in zip(v2, v1)]  # v1-v2

    segVector = [a * b for a, b in zip(segVector, pixelDims)]

    cosine_angle = np.dot(zVector, segVector) / (
        np.linalg.norm(zVector) * np.linalg.norm(segVector)
    )
    angle = np.arccos(round(cosine_angle, 4))

    return round(np.degrees(angle), 4)
