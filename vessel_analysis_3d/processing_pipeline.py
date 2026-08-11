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
import sys
from typing import Dict, Optional
from pathlib import Path
import tempfile
from datetime import datetime

# Third party
import logging
import numpy as np
import zarr
from scipy.ndimage import binary_closing, binary_dilation, binary_erosion
from tifffile import imread, imwrite
try:
    from bioio import BioImage
except ImportError:
    BioImage = None
try:
    from bioio.writers import OmeTiffWriter
except ImportError:
    OmeTiffWriter = None
from skimage.morphology import skeletonize

# Relative
from .graph.networkx_from_array import get_networkx_graph_from_array
from .graph.core import GraphObj
from .graph.stats_reporting import save_everything, report_everything


###############################################################################

log = logging.getLogger(__name__)

###############################################################################


def _anisotropic_ellipsoid_footprint(radius_um: float, pixel_dims) -> np.ndarray:
    """Boolean ellipsoid structuring element with the given physical radius.

    Voxel radii are derived per-axis from ``pixel_dims`` so the footprint is
    round in physical (um) space even though the voxel grid is anisotropic
    (e.g. a coarser Z step than XY).
    """
    radii_vox = [max(1, int(round(radius_um / d))) for d in pixel_dims]
    grids = np.ogrid[tuple(slice(-r, r + 1) for r in radii_vox)]
    dist2 = sum((g / r) ** 2 for g, r in zip(grids, radii_vox))
    return dist2 <= 1


class Pipeline3D(object):
    """
    The parent object class.
    """

    def __init__(self):
        self.filenames = None
        self.analysis_parameters = None
        self.input_metadata = {}

    def _is_zarr_root(self, path: Path) -> bool:
        return path.is_dir() and (
            path.name.endswith(".ome.zarr")
            or path.name.endswith(".zarr")
            or (path / ".zgroup").exists()
            or (path / ".zarray").exists()
        )

    def config_data(self, param_dict):
        """
        locate segmentation files and update parameters for output paths
        """
        seg_p = Path(param_dict["input_dir"])
        if self._is_zarr_root(seg_p):
            self.filenames = [seg_p]
        else:
            candidates = sorted(seg_p.glob(param_dict["input_type"]))
            self.filenames = [
                p
                for p in candidates
                if self._is_zarr_root(p)
                or p.suffix.lower() in {".tif", ".tiff"}
            ]

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

        physical_dims = [
            self.input_metadata.get("PhysicalSizeZ"),
            self.input_metadata.get("PhysicalSizeY"),
            self.input_metadata.get("PhysicalSizeX"),
        ]
        if all(isinstance(v, (int, float)) and v > 0 for v in physical_dims):
            self.analysis_parameters["pixelDimensions"] = physical_dims

        if "pruningScale" not in param_dict:
            self.analysis_parameters["pruningScale"] = 1.5

        if "lengthLimit" not in param_dict:
            self.analysis_parameters["lengthLimit"] = 3

        if "diaScale" not in param_dict:
            self.analysis_parameters["diaScale"] = 2

        if "branchingThreshold" not in param_dict:
            self.analysis_parameters["branchingThreshold"] = 0.25

        if "minBranchLengthUm" not in param_dict:
            self.analysis_parameters["minBranchLengthUm"] = 20.0

        if "minBranchRadiusUm" not in param_dict:
            self.analysis_parameters["minBranchRadiusUm"] = 25.0

        if "minJunctionRadiusUm" not in param_dict:
            self.analysis_parameters["minJunctionRadiusUm"] = 25.0

        if "maskOpeningRadiusUm" not in param_dict:
            self.analysis_parameters["maskOpeningRadiusUm"] = 3.0

        if "maskTipRecoverySteps" not in param_dict:
            # surface noise sits at nearly the same scale as a real tip taper here:
            # 1 step stayed clean in validation (26 branch points), 2 steps already
            # reconnected most surface roughness (614 branch points). Don't raise
            # without re-validating on a crop.
            self.analysis_parameters["maskTipRecoverySteps"] = 1

        if "clusterCollapseScale" not in param_dict:
            self.analysis_parameters["clusterCollapseScale"] = 3.0

    def process_one_file(
        self,
        seg: np.ndarray,
        skl: np.ndarray,
        param_dict: Dict,
        save_path: Path = None,
        basename: str = "default",
        raw_segmentation: Optional[np.ndarray] = None,
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
        raw_segmentation: numpy.ndarray, optional
            segmentation before mask-opening preprocessing; if given, true skeleton
            endpoints are extended back out to this mask's boundary along their
            local tangent, undoing the tip retraction caused by mask opening.
            Defaults to `seg` (no extension) when not provided.
        """
        # skeleton to graph
        networkxGraph = get_networkx_graph_from_array(skl)

        # Statistical Analysis
        gh = GraphObj(
            seg,
            skl,
            networkxGraph,
            rawSegmentation=raw_segmentation,
            **param_dict,
        )
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
            brPt_img = np.zeros(seg.shape, dtype=np.uint8)
            for ind in brPts:
                brPt_img[ind] = 255

            endPt_img = np.zeros(seg.shape, dtype=np.uint8)
            for ind in endPts:
                endPt_img[ind] = 255

            skl_final = skl_final.astype(np.uint8)
            skl_final[skl_final > 0] = 255

            out_meta = dict(self.input_metadata or {})
            out_meta.setdefault("axes", "ZYX")
            self._write_ome_tiff(
                save_path / f"{basename}_raw_segmentation.ome.tiff",
                seg.astype(np.uint8),
                out_meta,
            )
            self._write_ome_tiff(
                save_path / f"{basename}_skeleton_final.ome.tiff",
                skl_final,
                out_meta,
            )
            self._write_ome_tiff(
                save_path / f"{basename}_branch_points.ome.tiff",
                brPt_img,
                out_meta,
            )
            self._write_ome_tiff(
                save_path / f"{basename}_end_points.ome.tiff",
                endPt_img,
                out_meta,
            )

            save_everything(gh, save_path, basename)

    def _write_ome_tiff(self, path: Path, arr: np.ndarray, metadata: Optional[Dict] = None) -> None:
        if metadata is None:
            metadata = {}
        metadata.setdefault("axes", "ZYX")
        arr = np.ascontiguousarray(arr)

        # Ensure predictor is only used for integer dtypes; cast simple float images to uint8
        predictor = 2
        if arr.dtype.kind == "f":
            # if values are integer-valued within 0-255, cast to uint8 for compact storage
            try:
                vmin = float(arr.min())
                vmax = float(arr.max())
            except Exception:
                vmin, vmax = 0.0, 1.0
            if np.allclose(arr, np.round(arr)) and vmin >= 0 and vmax <= 255:
                arr = arr.astype(np.uint8)
                predictor = 2
            else:
                # use float32 and do not use horizontal predictor
                arr = arr.astype(np.float32)
                predictor = None

        if predictor is None:
            imwrite(
                str(path),
                arr,
                ome=True,
                metadata=metadata,
                compression="zlib",
            )
        else:
            imwrite(
                str(path),
                arr,
                ome=True,
                metadata=metadata,
                compression="zlib",
                predictor=predictor,
            )

    def _extract_zarr_metadata(self, root, source_path: Optional[Path] = None):
        metadata: Dict = {"axes": "ZYX"}
        multiscales = getattr(root, "attrs", {}).get("multiscales")
        if isinstance(multiscales, list) and len(multiscales) > 0:
            dataset = multiscales[0].get("datasets", [])[0]
            ct = dataset.get("coordinateTransformations", [])
            if ct and isinstance(ct, list):
                scale = ct[0].get("scale")
                if isinstance(scale, (list, tuple)) and len(scale) >= 3:
                    metadata["PhysicalSizeZ"] = float(scale[0])
                    metadata["PhysicalSizeY"] = float(scale[1])
                    metadata["PhysicalSizeX"] = float(scale[2])
                    metadata["PhysicalSizeZUnit"] = "µm"
                    metadata["PhysicalSizeYUnit"] = "µm"
                    metadata["PhysicalSizeXUnit"] = "µm"

        if "PhysicalSizeX" not in metadata:
            source = f" for {source_path}" if source_path is not None else ""
            print(
                f"Warning: input metadata missing physical pixel sizes{source}; output OME-TIFF will use 1,1,1 scale",
                file=sys.stderr,
            )
        return metadata

    def _is_zarr_root(self, fn):
        path = Path(fn)
        return path.is_dir() and (
            str(path).endswith(".ome.zarr")
            or str(path).endswith(".zarr")
            or (path / ".zgroup").exists()
            or (path / ".zarray").exists()
        )

    def _load_segmentation(self, fn):
        if self._is_zarr_root(fn):
            z = zarr.open_group(fn, mode="r")
            self.input_metadata = self._extract_zarr_metadata(z, fn)
            if "0" in z:
                seg = z["0"]
            else:
                seg = next(iter(z.values()))
            return np.asarray(seg)

        if BioImage is not None:
            try:
                seg = BioImage(fn).get_image_data("ZYX", C=0, T=0)
                if self._is_zarr_root(Path(fn)):
                    try:
                        z = zarr.open_group(fn, mode="r")
                        self.input_metadata = self._extract_zarr_metadata(z, fn)
                    except Exception:
                        self.input_metadata = {"axes": "ZYX"}
                else:
                    self.input_metadata = {"axes": "ZYX"}
                return seg
            except Exception:
                pass

        if fn.endswith(".tiff") or fn.endswith(".tif"):
            self.input_metadata = {"axes": "ZYX"}
            return imread(fn)

        raise ValueError(f"Unsupported segmentation format: {fn}")

    def run(self):
        for fn in self.filenames:
            # load the segmentation
            seg = self._load_segmentation(str(fn))
            seg = seg.astype(np.uint8)
            seg[seg > 0] = 1

            physical_dims = [
                self.input_metadata.get("PhysicalSizeZ"),
                self.input_metadata.get("PhysicalSizeY"),
                self.input_metadata.get("PhysicalSizeX"),
            ]
            if all(isinstance(v, (int, float)) and v > 0 for v in physical_dims):
                self.analysis_parameters["pixelDimensions"] = physical_dims

            # keep the untouched mask around: mask opening below trims a few
            # voxels off every tapering vessel tip along with the noise it is
            # meant to suppress. Rather than loosen the opening to recover that
            # (tried: reintroduces almost all of the surface noise, see
            # maskTipRecoverySteps below), true endpoints are extended back out
            # to this raw boundary afterwards in GraphObj.
            seg_raw = seg.copy()

            # Denoise thin, few-voxel bridges caused by segmentation-boundary
            # uncertainty ("gray areas") before skeletonization. A morphological
            # opening only cuts structures narrower than the footprint and
            # restores the rest via dilation, so real vessels (wider than the
            # footprint, even if thin overall) are preserved.
            mask_opening_radius_um = float(
                self.analysis_parameters.get("maskOpeningRadiusUm", 3.0)
            )
            # how many dilation steps to grow back after erosion; kept at 1
            # (the same footprint used once, i.e. a plain opening) because this
            # segmentation's surface noise sits at nearly the same scale as a
            # real tip taper -- 2+ steps was shown to reconnect most of the
            # noise (26 -> 614 branch points on a validation crop).
            mask_recovery_steps = int(
                self.analysis_parameters.get("maskTipRecoverySteps", 1)
            )
            if mask_opening_radius_um > 0:
                pixel_dims = self.analysis_parameters.get("pixelDimensions", [1.0, 1.0, 1.0])
                footprint = _anisotropic_ellipsoid_footprint(mask_opening_radius_um, pixel_dims)
                seg_mask = seg > 0
                eroded = binary_erosion(seg_mask, structure=footprint)
                opened = binary_dilation(
                    eroded,
                    structure=footprint,
                    iterations=mask_recovery_steps,
                    mask=seg_mask,
                )
                closed = binary_closing(opened, structure=footprint)
                seg = closed.astype(np.uint8)

            # segmentation -> skeleton
            skl = skeletonize(seg > 0, method="lee")
            skl = skl.astype(np.uint8)
            skl[skl > 0] = 1

            self.process_one_file(
                seg,
                skl,
                self.analysis_parameters,
                save_path=Path(self.tmp_path),
                basename=fn.stem,
                raw_segmentation=seg_raw,
            )

    # Representation's (reprs) are useful when using interactive Python sessions or
    # when you log an object. They are the shorthand of the object state. In this case,
    # our string method provides a good representation.
    def __repr__(self):
        return str(self)
