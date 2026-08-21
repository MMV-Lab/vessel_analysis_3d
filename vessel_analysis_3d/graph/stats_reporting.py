import csv
from typing import List, Dict
import numpy as np
import pandas as pd

from .core import GraphObj


def save_everything(gh: GraphObj, out_path, base_name):
    # create files containing all statisics in one csv per category (segment, filament,
    # branches and endPtsRatio)
    saveAllStatsAsCSV(
        gh.segStatsDict, out_path / f"{base_name}_Segment_Statistics.csv", base_name
    )
    saveAllFilStatsAsCSV(
        gh.filStatsDict, out_path / f"{base_name}_Filament_Statistics.csv", base_name
    )
    saveBranchesBrPtAsCSV(
        gh.branchesBrPtDict,
        out_path / f"{base_name}_BranchesPerBranchPt.csv",
        base_name,
    )
    saveSummaryStatsAsCSV(
        gh, out_path / f"{base_name}_Summary_Statistics.csv", base_name
    )
    saveSegmentProfilesAsCSV(
        gh.segmentProfilesDict,
        out_path / f"{base_name}_Segment_Diameter_Profiles.csv",
        base_name,
    )


def report_everything(gh: GraphObj, basename):
    all_stat = reportAllStats(gh.segStatsDict, basename)
    all_filstat = reportAllFilStats(gh.filStatsDict, basename)
    all_brstat = reportBranchesBrPt(gh.branchesBrPtDict, basename)
    summary_stat = reportSummaryStats(gh, basename)
    profile_stat = reportSegmentProfiles(gh.segmentProfilesDict, basename)

    return all_stat, all_filstat, all_brstat, summary_stat, profile_stat


def getAllStats(dictionary: Dict, imgName: str) -> List:
    """
    Parameters:
    -----
    imgName: str
        a unique identifier for this file
    dictionary: Dict
        dictionary of all stats. keys: filament IDs, e.g., 100
        101, 104, ..... For each filament ID, it is associated
        with a segment, like ((108, 294, 62), (104, 294, 62)),
        and for each segment, it maps to several features, like
        "diameter", "straightness", etc..

    Return:
    --------
    a single list that can be either converted to a DataFrame
    or saved as a CSV
    """

    # get all segment measurements as list from dictionary
    fil_id = None
    key = None
    for idx in dictionary:
        if bool(dictionary[idx]):
            key = next(iter(dictionary[idx]))
            fil_id = idx
            break
    ms_list = []
    if fil_id is not None:
        for i in dictionary[fil_id][key].keys():
            ms_list.append(i)
    list = [["image", "filamentID", "segmentID"]]  # header list
    for i in ms_list:
        list[0].append(i)
    for filament in dictionary.keys():
        for segment in dictionary[filament]:
            final = []
            for element in segment:
                element = tuple(int(x) for x in element)
                final.append(element)
            segment = tuple(final)
            list_item = [imgName, filament, segment]
            for stat in dictionary[filament][segment]:
                list_item.append(dictionary[filament][segment][stat])
            list.append(list_item)

    return list


def getAllFilStats(dictionary, imgName):
    list = [
        [
            "Image",
            "FilamentID",
            "No. Segments",
            "No. Terminal Points",
            "No. Branching Points",
        ]
    ]
    for filament in dictionary.keys():
        segs = dictionary[filament]["Segments"]
        endPts = dictionary[filament]["TerminalPoints"]
        brPts = dictionary[filament]["BranchPoints"]
        list_item = [imgName, filament, segs, endPts, brPts]
        list.append(list_item)

    return list


def getBranchesBrPt(dictionary, imgName):
    list = [["Image", "FilamentID", "BranchID", "No. Branches per BranchPoint"]]
    for filament in dictionary.keys():
        for segment in dictionary[filament]:
            segment = tuple(int(x) for x in segment)
            branches = dictionary[filament][segment]
            list_item = [imgName, filament, segment, branches]
            list.append(list_item)

    return list


def saveAllStatsAsCSV(dictionary, path, imgName):
    list = getAllStats(dictionary, imgName)
    with open(path, "w", newline="") as file:
        writer = csv.writer(file, delimiter=";")
        writer.writerows(list)


def saveAllFilStatsAsCSV(dictionary, path, imgName):
    list = getAllFilStats(dictionary, imgName)
    with open(path, "w", newline="") as file:
        writer = csv.writer(file, delimiter=";")
        writer.writerows(list)


def saveBranchesBrPtAsCSV(dictionary, path, imgName):
    list = getBranchesBrPt(dictionary, imgName)
    with open(path, "w", newline="") as file:
        writer = csv.writer(file, delimiter=";")
        writer.writerows(list)


def reportAllStats(dictionary, imgName):
    list = getAllStats(dictionary, imgName)

    return pd.DataFrame(list[1:], columns=list[0])


def reportAllFilStats(dictionary, imgName):
    list = getAllFilStats(dictionary, imgName)

    return pd.DataFrame(list[1:], columns=list[0])


def reportBranchesBrPt(dictionary, imgName):
    list = getBranchesBrPt(dictionary, imgName)

    return pd.DataFrame(list[1:], columns=list[0])


def computeSummaryStats(gh: GraphObj, imgName: str) -> Dict:
    """
    Aggregate whole-image morphometrics that are cheap to derive from the
    already-computed per-segment/per-filament statistics: diameter summary,
    totals, and density metrics normalized by tissue volume. Tissue volume is
    the full imaged volume (array shape x pixel size) -- there is no separate
    tissue mask, so this is the best available denominator.
    """
    diameters, lengths, surface_areas = [], [], []
    boundary_clipped_segments = 0
    for filament_segs in gh.segStatsDict.values():
        for stats in filament_segs.values():
            diameters.append(stats["diameter"])
            lengths.append(stats["length"])
            surface_areas.append(stats.get("surfaceArea", 0.0))
            if stats.get("boundaryClipped", False):
                boundary_clipped_segments += 1

    total_branch_points = sum(gh.countBranchPointsDict.values())
    total_terminal_points = sum(gh.countEndPointsDict.values())
    total_segments = sum(gh.countSegmentsDict.values())

    pixel_dims_um = np.asarray(gh.pixelDims, dtype=float)
    tissue_volume_mm3 = float(
        np.prod(gh.skeleton.shape) * np.prod(pixel_dims_um) / 1e9
    )

    raw_seg = gh.rawSegmentation
    vessel_volume_fraction = (
        float((raw_seg > 0).sum() / raw_seg.size)
        if raw_seg is not None and raw_seg.size
        else float("nan")
    )

    total_length_um = float(sum(lengths))
    total_length_mm = total_length_um / 1000.0
    total_surface_area_mm2 = float(sum(surface_areas)) / 1e6

    def density(value):
        return value / tissue_volume_mm3 if tissue_volume_mm3 else float("nan")

    return {
        "image": imgName,
        "meanDiameter_um": float(np.mean(diameters)) if diameters else float("nan"),
        "minDiameter_um": float(np.min(diameters)) if diameters else float("nan"),
        "maxDiameter_um": float(np.max(diameters)) if diameters else float("nan"),
        "totalBranchPoints": total_branch_points,
        "totalTerminalPoints": total_terminal_points,
        "totalSegments": total_segments,
        # segments flagged by GraphObj._flag_boundary_clipped_segments (raw
        # mask within boundaryClipMarginUm of both a volume face and one of
        # the segment's own skeleton nodes) -- included in every number
        # above ("out-of-competition" is not "excluded"), just called out
        # here so a reader knows what fraction of the pooled stats may
        # include a partially-clipped cross-section without
        # cross-referencing Segment_Statistics.csv.
        "boundaryClippedSegments": boundary_clipped_segments,
        "boundaryClippedFraction": (
            boundary_clipped_segments / total_segments if total_segments else float("nan")
        ),
        "totalVesselLength_mm": total_length_mm,
        "tissueVolume_mm3": tissue_volume_mm3,
        "vesselVolumeFraction": vessel_volume_fraction,
        "vesselLengthDensity_mm_per_mm3": density(total_length_mm),
        "branchPointDensity_per_mm3": density(total_branch_points),
        "endPointDensity_per_mm3": density(total_terminal_points),
        "totalSurfaceArea_mm2": total_surface_area_mm2,
        "surfaceAreaDensity_mm2_per_mm3": density(total_surface_area_mm2),
    }


def reportSummaryStats(gh: GraphObj, imgName: str) -> pd.DataFrame:
    return pd.DataFrame([computeSummaryStats(gh, imgName)])


def saveSummaryStatsAsCSV(gh: GraphObj, path, imgName: str):
    df = reportSummaryStats(gh, imgName)
    df.to_csv(path, sep=";", index=False)


def getSegmentProfiles(dictionary: Dict, imgName: str) -> List:
    """
    Long-format table: one row per (segment, point along its skeleton path).

    Parameters
    -----
    dictionary: Dict
        gh.segmentProfilesDict -- filament ID -> segment key -> list of
        (position_um, diameter_um, z, y, x) tuples, in path order.
    """
    list_ = [
        [
            "image",
            "filamentID",
            "segmentID",
            "pointIndex",
            "position_um",
            "diameter_um",
            "z",
            "y",
            "x",
        ]
    ]
    for filament in dictionary.keys():
        for segment, profile in dictionary[filament].items():
            segmentID = tuple(tuple(int(x) for x in node) for node in segment)
            for point_index, (position_um, diameter_um, z, y, x) in enumerate(profile):
                list_.append(
                    [imgName, filament, segmentID, point_index, position_um, diameter_um, z, y, x]
                )
    return list_


def saveSegmentProfilesAsCSV(dictionary, path, imgName):
    list_ = getSegmentProfiles(dictionary, imgName)
    with open(path, "w", newline="") as file:
        writer = csv.writer(file, delimiter=";")
        writer.writerows(list_)


def reportSegmentProfiles(dictionary, imgName) -> pd.DataFrame:
    list_ = getSegmentProfiles(dictionary, imgName)
    return pd.DataFrame(list_[1:], columns=list_[0])
