# vessel_analysis_3d

Fork of [MMV-Lab/vessel_analysis_3d](https://github.com/MMV-Lab/vessel_analysis_3d), branch
`feature/skeletonization-and-morphometrics`. Converts a binary 3D vessel segmentation into a
skeleton graph and per-segment/per-filament morphometrics (diameter, length, straightness,
branching angle, volume, surface area, ...), tuned here for large, densely-vascularized
bioprinted-tissue segmentations rather than the original small-scale use case.

## Installation

```bash
conda create -y -n 3danalysis -c conda-forge python=3.11
conda activate 3danalysis
pip install git+https://github.com/MMV-Lab/vessel_analysis_3d.git@feature/skeletonization-and-morphometrics
```

## Running from the command line

```bash
run_vessel_analysis --config /path/to/your_config.yaml
```

Copy `example_configs/example.yaml` as a starting point and fill in your own `input_dir`/
`output_dir` and pixel dimensions.

Output lands in a timestamped subfolder of `output_dir`: `*_Segment_Statistics.csv`,
`*_Filament_Statistics.csv`, `*_Summary_Statistics.csv`, `*_Segment_Diameter_Profiles.csv`,
`*_BranchesPerBranchPt.csv`, and OME-TIFFs for the raw segmentation, final skeleton, branch
points, end points, and border-excluded end points.

## Features

- **Skeleton-only mask smoothing.** A Gaussian-smoothed, morphologically-opened copy of the
  mask decides skeleton topology; every measurement and export still uses the real, unsmoothed
  segmentation.
- **Redundancy-gated spur pruning.** Rough-vessel-wall branch noise is only removed when a
  trusted, already-confirmed path already covers the same territory nearby, instead of a
  single absolute threshold that can delete real anatomy.
- **Isolated component restoration.** A standalone, genuinely branched structure with no
  larger trunk for pruning to fall back on keeps its own longest throughline instead of being
  cascade-pruned to nothing.
- **Bounded endpoint extension.** Endpoint extension is capped by cumulative direction change,
  preventing "hook" artifacts where the skeleton doubles back on itself near a blunt or
  ambiguous tip.
- **Real skeleton-path segments.** Segments trace the actual skeleton path, not a straight line
  between their endpoints; length/straightness/volume/surface area are measured along that
  real path.
- **Branching angles** are measured between the straight branch-point-to-far-endpoint
  connections rather than the traced curve -- with real tortuosity in the way there's no single
  well-defined point along a curve to anchor an angle to.
- **Per-segment `boundaryClipped` flag.** Flags a segment if raw-mask tissue sits within
  `boundaryClipMarginUm` (default 25um) of both a volume face and one of its own skeleton
  nodes -- catches a vessel cut off at the edge whether it terminates there or just runs
  alongside it. Flagged, never excluded, from any stat.
- **Physical-margin border-endpoint exclusion.** Uses the same margin test as above instead of
  an exact-boundary-voxel match, which mask-opening erosion routinely defeats.

## Known limitations

- Lee's thinning can produce **no skeleton at all** through some wide/short mask regions, even
  when the mask there is fully solid -- there is no segment to flag or measure in that case.
- A **thick vessel truncated exactly at the volume edge** can be skipped for the same reason,
  most often for vessels cut off by the crop boundary rather than a real anatomical end.

Both are upstream of anything this package's graph/pruning logic can fix from the skeleton
alone; see the docstring of `Filament._removeBorderPtsFromEndPts` for the full detail.

## Development

See [CONTRIBUTING.md](CONTRIBUTING.md). **MIT license**, see [LICENSE](LICENSE).
