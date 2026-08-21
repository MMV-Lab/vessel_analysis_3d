# vessel_analysis_3d (BioPrint-Segmentation fork)

Fork of [MMV-Lab/vessel_analysis_3d](https://github.com/MMV-Lab/vessel_analysis_3d), branch
`feature/skeletonization-and-morphometrics`. Converts a binary 3D vessel segmentation into a
skeleton graph and per-segment/per-filament morphometrics (diameter, length, straightness,
branching angle, volume, surface area, ...), tuned here for large, densely-vascularized
bioprinted-tissue segmentations rather than the original small-scale use case.

## What's changed from upstream

- **Mask smoothing is skeleton-only.** A Gaussian-smoothed, morphologically-opened copy of the
  mask decides skeleton topology; every measurement and export still uses the real, unsmoothed
  segmentation.
- **Redundancy-gated spur pruning** for rough vessel walls, instead of a single absolute
  threshold that could delete real anatomy.
- **Isolated components are no longer cascade-pruned to nothing.** A standalone, genuinely
  branched structure with no larger trunk to fall back on now has its own longest throughline
  restored if pruning would otherwise remove it entirely.
- **Endpoint extension is bounded by cumulative direction change**, fixing "hook" artifacts
  where the skeleton could double back on itself near a blunt or ambiguous tip.
- **Segments trace the real skeleton path**, not a straight line between their endpoints;
  length/straightness/volume/surface area are measured along that real path.
- **Per-segment `boundaryClipped` flag**: a segment is flagged if raw-mask foreground sits
  within `boundaryClipMarginUm` (default 25um) of both a volume face and one of its own
  skeleton nodes -- catches a vessel truncated at the crop edge whether it terminates there or
  just runs alongside it. Flagged segments are marked, never excluded, from any stat.
- **Border-endpoint exclusion** uses the same physical-margin test (was an exact-boundary-voxel
  check, which mask-opening erosion routinely defeated).

## Known limitations

- Lee's thinning can produce **no skeleton at all** through some wide/short mask regions, even
  when the mask there is fully solid -- there is no segment to flag or measure in that case.
- A **thick vessel truncated exactly at the volume edge** can be skipped for the same reason,
  most often for vessels cut off by the crop boundary rather than a real anatomical end.

Both are upstream of anything this package's graph/pruning logic can fix from the skeleton
alone; see the docstring of `Filament._removeBorderPtsFromEndPts` for the full detail.

## Quick start

```python
from vessel_analysis_3d.processing_pipeline import Pipeline3D
from skimage.morphology import skeletonize
import numpy as np

# SEG: your binary segmentation as a numpy array
SKL = skeletonize(SEG > 0, method="lee").astype(np.uint8)

# PARAMS: dict of analysis parameters, see example_configs/example.yaml
skl_final, brPts, endPts, reports = Pipeline3D.process_one_file(SEG, SKL, PARAMS)
```

Or from the command line:

```bash
run_vessel_analysis --config /path/to/your_config.yaml
```

See `example_configs/example.yaml` for a starting point -- copy it and fill in your own
`input_dir`/`output_dir` and pixel dimensions.

## Installation

```bash
conda create -y -n 3danalysis -c conda-forge python=3.11
conda activate 3danalysis
pip install git+https://github.com/MMV-Lab/vessel_analysis_3d.git@feature/skeletonization-and-morphometrics
```

## Tests

```bash
pytest vessel_analysis_3d/tests/
```

## Development

See [CONTRIBUTING.md](CONTRIBUTING.md). **MIT license**, see [LICENSE](LICENSE).
