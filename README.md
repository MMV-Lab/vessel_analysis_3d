# 3D Vascular Structure Analysis

[![Build Status](https://github.com/MMV-Lab/vessel_analysis_3d/workflows/Build%20Main/badge.svg)](https://github.com/MMV-Lab/vessel_analysis_3d/actions)
[![Documentation](https://github.com/MMV-Lab/vessel_analysis_3d/workflows/Documentation/badge.svg)](https://MMV-Lab.github.io/vessel_analysis_3d/)
[![Code Coverage](https://codecov.io/gh/MMV-Lab/vessel_analysis_3d/branch/main/graph/badge.svg)](https://codecov.io/gh/MMV-Lab/vessel_analysis_3d)

A Python package for analyzing 3D vascular structures from segmentations

---


## Quick Start

In general, the analysis pipeline aims to convert segmented 3d vascular structures into statistics, with the following 3 steps:

1. Binary (segmentation) to preliminary skeleton

2. Preliminary skeleton to network representation (including network pruning)

3. Extract statistics

You can run analysis like `run_vessel_analysis --config ./example_configs/example.yaml`.

Or, you can do this analysis in other functions by calling the processing pipeline

```python
from vessel_analysis_3d.processing_pipeline import Pipeline3D
from skimage.morphology import skeletonize
import numpy as np

# Suppose SEG is the segmentation

# get preliminary skeleton
SKL = skeletonize(SEG > 0, method="lee")
SKL = SKL.astype(np.uint8)
SKL[SKL > 0] = 1

# Support PARAMS contains all parameters
skl_final, brPts, endPts, reports = Pipeline3D.process_one_file(SEG, SKL, PARAMS)

# skl_final is your final skeleton
# brPts and endPts are two lists for the positions of branch points and end points
# reports contain three pandas DataFrames ready for plotting
```

## Installation


Before starting, we recommend to [create a new conda environment](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-with-commands) or [a virtual environment](https://docs.python.org/3/library/venv.html) with Python 3.10+.

```bash
conda create -y -n 3danalysis -c conda-forge python=3.11
conda activate 3danalysis
```

**Stable Release:** `pip install vessel_analysis_3d`<br> (not released yet)
**Development Head:** `pip install git+https://github.com/MMV-Lab/vessel_analysis_3d.git`

## Documentation

For full package documentation please visit [MMV-Lab.github.io/vessel_analysis_3d](https://MMV-Lab.github.io/vessel_analysis_3d).

## Development

See [CONTRIBUTING.md](CONTRIBUTING.md) for information related to developing the code.

**MIT license**

