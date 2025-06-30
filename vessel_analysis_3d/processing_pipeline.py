#!/usr/bin/env python
# -*- coding: utf-8 -*-

# code adapted from https://github.com/RUB-Bioinf/VesselExpress/blob/master/VesselExpress/workflow/scripts/graphAnalysis.py  # noqa E501

# Imports should be grouped into:
# Standard library imports
# Related third party imports
# Local application / relative imports
# in that order

# Standard library
import logging
from typing import Dict
from pathlib import Path
import tempfile
from datetime import datetime

# Third party
import numpy as np
from bioio import BioImage
from bioio.writers import OmeTiffWriter
from skimage.morphology import skeletonize

# Relative
from .graph.networkx_from_array import get_networkx_graph_from_array
from .graph.core import GraphObj
from .graph.stats_reporting import save_everything, report_everything


###############################################################################

log = logging.getLogger(__name__)

###############################################################################


class Pipeline3D(object):
    """
    The parent object class.
    """

    def __init__(self):
        self.filenames = None
        self.analysis_parameters = None

    def config_data(self, param_dict):
        """
        locate segmentation files and update parameters for output paths
        """
        seg_p = param_dict["input_dir"]
        self.filenames = sorted(Path(seg_p).glob(param_dict["input_type"]))

        if self.filenames is None or len(self.filenames) == 0:
            raise FileNotFoundError(f"no files at {seg_p}")

        ##########################################
        # setup path for intermediate results
        ##########################################
        if "output_dir" in param_dict:
            tmp_path = Path(param_dict["output_dir"])
        else:
            tmp_path = Path("./_tmp")

        tmp_path.mkdir(exist_ok=True)

        # create a random folder for each run
        current_time = datetime.now()
        time_pref = str(current_time)[:-7]
        time_pref = time_pref.replace(" ", "_").replace("-", "_").replace(":", "_")
        self.tmp_path = Path(
            tempfile.TemporaryDirectory(prefix=time_pref + "_", dir=tmp_path).name
        )
        self.tmp_path.mkdir(exist_ok=True)

    def config_analysis(self, param_dict):
        self.analysis_parameters = param_dict

        # validate with default parameters
        if "pixelDimensions" not in param_dict:
            self.analysis_parameters["pixelDimensions"] = [2.0, 1.015625, 1.015625]

        if "pruningScale" not in param_dict:
            self.analysis_parameters["pruningScale"] = 1.5

        if "lengthLimit" not in param_dict:
            self.analysis_parameters["lengthLimit"] = 3

        if "diaScale" not in param_dict:
            self.analysis_parameters["diaScale"] = 2

        if "branchingThreshold" not in param_dict:
            self.analysis_parameters["branchingThreshold"] = 0.25

    @staticmethod
    def process_one_file(
        seg: np.ndarray,
        skl: np.ndarray,
        param_dict: Dict,
        save_path: Path = None,
        basename: str = "default",
    ):
        """process one image, results are either returned as variables or saved to files

        Parameters:
        -------------
        seg: numpy.ndarray
        skl: numpy.ndarray
        param_dict: Dict
            all paramters for stats calculation
        save_path: Path
            if provided, the stats will be saved to CSV, otherwise returned as DataFrame
        basename: str
            a unique identifier for this file
        """
        # skeleton to graph
        networkxGraph = get_networkx_graph_from_array(skl)

        # Statistical Analysis
        gh = GraphObj(seg, skl, networkxGraph, **param_dict)
        skl_final = gh.prune_and_analyze(return_final_skel=True)

        # get all branch points
        brPts = []
        for i in gh.branchPointsDict.values():
            if i.keys():
                for k in i.keys():
                    brPts.append(k)

        # get all endpoints
        endPts = []
        for i in gh.endPointsDict.values():
            for li in i:
                endPts.append(li)

        if save_path is None:
            # return final skeleton as an array and branch points as a list
            reports = report_everything(gh, basename)
            return skl_final, brPts, endPts, reports
        else:
            # save the final skeleton and visualization of branchpoints and endpoints
            # on image as TIFFs
            brPt_img = np.zeros(seg.shape)
            for ind in brPts:
                brPt_img[ind] = 255

            endPt_img = np.zeros(seg.shape)
            for ind in endPts:
                endPt_img[ind] = 255

            skl_final = skl_final.astype(np.uint8)
            skl_final[skl_final > 0] = 255

            OmeTiffWriter.save(
                skl_final,
                save_path / f"{basename}_skeleton_final.tiff",
                dim_order="ZYX",
            )
            OmeTiffWriter.save(
                brPt_img, save_path / f"{basename}_branch_points.tiff", dim_order="ZYX"
            )
            OmeTiffWriter.save(
                endPt_img, save_path / f"{basename}_end_points.tiff", dim_order="ZYX"
            )

            save_everything(gh, save_path, basename)

    def run(self):
        for fn in self.filenames:
            # load the segmentation
            seg = BioImage(fn).get_image_data("ZYX", C=0, T=0)
            seg = seg.astype(np.uint8)
            seg[seg > 0] = 1

            # segmentation -> skeleton
            skl = skeletonize(seg > 0, method="lee")
            skl = skl.astype(np.uint8)
            skl[skl > 0] = 1

            # run analysis
            self.process_one_file(
                seg,
                skl,
                self.analysis_parameters,
                save_path=Path(self.tmp_path),
                basename=fn.stem,
            )

    # Representation's (reprs) are useful when using interactive Python sessions or
    # when you log an object. They are the shorthand of the object state. In this case,
    # our string method provides a good representation.
    def __repr__(self):
        return str(self)
