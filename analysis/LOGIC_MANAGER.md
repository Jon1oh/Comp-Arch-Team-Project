# Logic Manager

`analysis/logic_manager.py` connects the DLC processing section to a future AI
section. It accepts the dictionary returned by `process_video()` or a saved
keypoint JSON file, then returns a JSON-compatible summary. It requires Python
3.10 or newer and uses only the standard library. It was tested in this
project's `dlc-cat` environment with Python 3.12.16.

## Use with the current project

Run this from the project root to analyze an existing export:

```powershell
python analysis/logic_manager.py `
  analysis/DeepLabCut-Processing/outputs/test1_cat_play_keypoints.json `
  --output analysis/DeepLabCut-Processing/outputs/test1_cat_play_logic_summary.json
```

The command writes JSON to standard output. `--output` additionally saves that
same summary to a file. No CSV is created. It refuses to overwrite the input
keypoint file. Invalid inputs or configuration produce a nonzero exit code and
a short error on standard error.

Use either of these interfaces from Python, starting in the project root:

```python
from analysis.logic_manager import analyze_dataset, analyze_file, LogicConfig

# Immediately after the DLC module returns its dictionary:
# dataset = process_video(video_path)
# summary = analyze_dataset(dataset)

# Or read a previously exported dataset:
summary = analyze_file(
    "analysis/DeepLabCut-Processing/outputs/test2_cat_stare_keypoints.json",
    "analysis/DeepLabCut-Processing/outputs/test2_cat_stare_logic_summary.json",
)

# Send this smaller JSON to your AI section when that section is implemented:
import json
ai_input = json.dumps(summary, allow_nan=False)
```

`main.py` is currently empty, while `animal_logic.py` and `pose_detector.py`
contain comments only. The new module can be imported without changing those
files. It does not run DLC or call an AI service. When coordinating the full
pipeline, call `analyze_dataset(dataset)` after DLC and pass its returned
dictionary to your AI module. The existing DLC folder name contains a hyphen,
so it cannot be used directly in a dotted Python import; reuse whichever import
mechanism your Input Manager uses for that module.

You can also reuse an instance:

```python
from analysis.logic_manager import LogicManager

manager = LogicManager()
summary = manager.analyze(dataset)
```

## Processing stages

1. Validate single-animal metadata, frame order and timestamps. Both `fps` and
   `video_fps` metadata conventions are accepted, with corresponding aliases for
   dimensions, duration and frame count. Missing timestamps use `frame / fps`.
2. Map SuperAnimal labels to canonical landmarks. Examples:
   `left_earbase -> left_ear_base`, `left_earend -> left_ear_tip`,
   `tail_end -> tail_tip`, and `back_middle -> spine_mid`. `back_base` and
   `back_end` are explicitly treated as front/rear trunk proxies for shoulder
   and hip. Antler, paw and other unused quadruped labels do not affect quality.
3. Reject non-finite, missing, out-of-bounds, low-confidence or explicitly
   invalid observations. Keep the frame and its other reliable landmarks.
   If a point lacks `valid`, recompute validity from its numeric measurements.
   An existing `valid=False` remains excluded even if the threshold is lowered.
4. Apply a centered moving average, independently within each uninterrupted
   valid landmark run. Never fill missing observations or smooth across gaps.
5. Calculate displacement, timestamp-based speed, image-plane angles, angular
   change rates, movement direction changes and movement bouts.
6. Aggregate a whole-video summary and three-second intervals. Activity changes
   are compared between adjacent intervals only when both labels are known.
   Long videos use at most 60 intervals, widening them automatically.

The input dictionary is not modified. Each landmark uses one source label for
the whole video; the implementation does not switch aliases between frames.
Distances are in pixels. Normalized speed is pixel speed divided by the image
diagonal, preserving aspect ratio and allowing resolution comparisons. No
physical distance or calibrated speed is inferred.

Head movement uses the nose in image coordinates. Head orientation is the axis
from the midpoint of the ear bases to the nose. Body movement uses the midpoint
of shoulder and hip, requiring both to be reliable. Tail and ear activity use
tip-minus-base coordinates, reducing shared translation. These relative
coordinates still respond to whole-body rotation and perspective changes.

Angles use image coordinates: right is 0 degrees and down is +90 degrees.
Circular angle averaging and differences handle the -180/+180 degree boundary.
The spine interior angle is 180 degrees for a straight three-point spine; it
does not establish whether a cat is standing, sitting or crouching.

## Output structure

| Field | Contents |
| --- | --- |
| `video_summary` | Name, duration, analyzed range, FPS, dimensions and frame count |
| `data_quality` | Detection reliability, unavailable landmarks and per-landmark coverage |
| `observed_features` | Head, ear, eye, tail and body geometry/activity |
| `movement_metrics` | Displacement, speed, coverage and counts for each canonical landmark |
| `temporal_patterns` | Interval activity, quality and changes from the preceding interval |
| `analysis_settings` | Configuration, exact label mapping, units and references |
| `warnings` | Frame-count discrepancies and excluded temporal gaps |
| `limitations` | Missing evidence and constraints the AI should retain |

Unknown measurements are `null`; insufficient activity evidence is `"unknown"`.
Zero means zero measured motion. All numbers are finite and rounded to six
decimal places in the final summary. Full frame coordinates are not included.

Quality is the fraction of reliable detections among mapped canonical
landmarks over all frames. A mapped point missing in a frame counts against
quality. A landmark that is entirely unavailable is listed separately rather
than included in the denominator. Therefore a `good` quality rating does not
mean every required feature is measurable; always check the per-feature
coverage and limitations. `good` is at least 80%, `limited` is at least 50%,
and lower percentages are `poor`.

## Configuration

```python
config = LogicConfig(
    confidence_threshold=0.70,
    smoothing_window=5,       # positive odd integer; use 1 to disable
    segment_seconds=3.0,
    low_speed_threshold=0.02,
    high_speed_threshold=0.10,
    minimum_coverage=0.50,
)
summary = analyze_dataset(dataset, config=config)
```

The default activity bands are `low` below 0.02 image diagonals/second,
`moderate` below 0.10 and `high` otherwise. A label requires at least two valid
adjacent-frame comparisons and reliable observations in at least 50% of the
frames in that scope. A movement bout begins when a valid pair crosses into
activity at or above the low-speed threshold; gaps start a new observed bout.

Measurements require consecutive frame indices and a timestamp difference no
greater than `maximum_gap_factor / fps`, default `1.5 / fps`. Gaps break
smoothing, displacement comparisons and direction-change continuity. Smoothing
is performed on the full video before interval aggregation, so samples near
an interval boundary can use neighboring observations from the same valid run.

`direction_change_degrees` defaults to 60 for image-plane movement direction
changes. `angular_deadband_degrees` defaults to 2 for angular direction
reversals. The latter is an exploratory activity statistic; it is not a
validated tail-swish count or oscillation frequency. The thresholds, activity
bands and quality scores need validation using annotated cat videos.

For a differently named model, override canonical-to-source labels explicitly:

```python
summary = analyze_dataset(
    dataset,
    keypoint_map={"nose": "snout", "tail_tip": "tail_end", "tail_mid": None},
)
```

CLI options also include `--confidence`, `--smoothing-window` and
`--segment-seconds`.

## Validation

```powershell
python -m unittest discover -s tests -p test_logic_manager.py -v
```

All 25 tests passed. They cover known displacement and speed, resolution
normalization, relative motion, both label/metadata formats, invalid and
missing detections, smoothing boundaries, temporal gaps, angle wrapping,
activity changes, malformed inputs, file round trips, input overwrite
protection and the command-line interface.

The three existing exports were also processed successfully:

| Existing export | Frames | Reliable detections | Quality | Summary size |
| --- | ---: | ---: | --- | ---: |
| `test1_cat_play_keypoints.json` | 277 | 59.90% | limited | 16,118 bytes |
| `test2_cat_stare_keypoints.json` | 186 | 89.11% | good | 15,521 bytes |
| `test3_cat_grooming_keypoints.json` | 176 | 54.55% | limited | 15,157 bytes |

These are observed detection-quality results, not behavioral ground truth.
Tail-tip reliability was only 30.69% in the play export and 9.09% in grooming.
Even in the stare export, joint reliable tail-base/tip coverage was below the
default 50% requirement. Whole-video tail activity is consequently `unknown`
for these samples. Do not convert that into a claim that the tail was still.

The AI should distinguish measurements from possible interpretations, retain
these limitations, and avoid inferring gaze, ear rotation, happiness, fear or
aggression from a single coordinate pattern. Camera motion, occlusion and
perspective remain potential sources of apparent movement.
