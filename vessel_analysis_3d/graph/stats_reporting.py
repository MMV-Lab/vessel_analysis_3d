import csv
from typing import List, Dict
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


def report_everything(gh: GraphObj, basename):
    all_stat = reportAllStats(gh.segStatsDict, basename)
    all_filstat = reportAllFilStats(gh.filStatsDict, basename)
    all_brstat = reportBranchesBrPt(gh.branchesBrPtDict, basename)

    return all_stat, all_filstat, all_brstat


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
    fil_id = 0
    key = 0
    for idx in dictionary:
        if bool(dictionary[idx]):
            key = next(iter(dictionary[idx]))
            fil_id = idx
            break
    ms_list = []
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
