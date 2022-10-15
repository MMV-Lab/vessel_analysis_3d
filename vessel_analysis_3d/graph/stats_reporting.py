import csv
import pandas as pd

from .core import GraphObj


def save_everything(gh: GraphObj, out_path, base_name):

    # create files containing all statisics in one csv per category (segment, filament, branches and endPtsRatio)
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


def saveAllStatsAsCSV(dictionary, path, imgName):
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
            list_item = [imgName, filament, segment]
            for stat in dictionary[filament][segment]:
                list_item.append(dictionary[filament][segment][stat])
            list.append(list_item)
    with open(path, "w", newline="") as file:
        writer = csv.writer(file, delimiter=";")
        writer.writerows(list)


def saveAllFilStatsAsCSV(dictionary, path, imgName):
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
    with open(path, "w", newline="") as file:
        writer = csv.writer(file, delimiter=";")
        writer.writerows(list)


def saveBranchesBrPtAsCSV(dictionary, path, imgName):
    list = [["Image", "FilamentID", "BranchID", "No. Branches per BranchPoint"]]
    for filament in dictionary.keys():
        for segment in dictionary[filament]:
            branches = dictionary[filament][segment]
            list_item = [imgName, filament, segment, branches]
            list.append(list_item)
    with open(path, "w", newline="") as file:
        writer = csv.writer(file, delimiter=";")
        writer.writerows(list)


def reportAllStats(dictionary, imgName):
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

    list = []
    for i in ms_list:
        list[0].append(i)
    for filament in dictionary.keys():
        for segment in dictionary[filament]:
            list_item = [imgName, filament, segment]
            for stat in dictionary[filament][segment]:
                list_item.append(dictionary[filament][segment][stat])
            list.append(list_item)

    return pd.DataFrame(list, columns=["image", "filamentID", "segmentID"])


def reportAllFilStats(dictionary, imgName):
    list = []
    for filament in dictionary.keys():
        segs = dictionary[filament]["Segments"]
        endPts = dictionary[filament]["TerminalPoints"]
        brPts = dictionary[filament]["BranchPoints"]
        list_item = [imgName, filament, segs, endPts, brPts]
        list.append(list_item)

    return pd.DataFrame(list, columns=["Image", "FilamentID", "No. Segments", "No. Terminal Points", "No. Branching Points"])


def reportBranchesBrPt(dictionary, imgName):
    list = []
    for filament in dictionary.keys():
        for segment in dictionary[filament]:
            branches = dictionary[filament][segment]
            list_item = [imgName, filament, segment, branches]
            list.append(list_item)

    return pd.DataFrame(list, columns=["Image", "FilamentID", "BranchID", "No. Branches per BranchPoint"])
