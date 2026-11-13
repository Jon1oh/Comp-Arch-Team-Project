# Movement evidence for AI interpretation

`movement_summary.py` analyzes raw single-animal DeepLabCut keypoint JSON. It is
activity-independent: the same measurements run for a stationary cat, walking,
play, grooming, or other movement. Missing body parts disable only assessments
that depend on them. It is not a universal behavior recognizer.

The pipeline is:

1. Validate metadata, frame order, timestamps, and single-animal input.
2. Select a fixed source label for each landmark across the entire clip.
3. Filter exporter-invalid, low-confidence, non-finite and out-of-image points.
4. Remove isolated extreme-speed spikes, then apply a centred coordinate median.
5. Compute fixed-reference geometry and consecutive-pair movement statistics.
6. Assess each feature's observation coverage and observed duration.
7. Apply versioned rules to the whole clip and bounded time windows.
8. Return JSON evidence for the AI, with no network calls.

## Run or import

From the repository root:

```powershell
python -m logic.movement_summary analysis/DeepLabCut-Processing/outputs/test2_cat_stare_keypoints.json --output summary.json
```

Enable the optional, unvalidated behavior hypotheses:

```powershell
python -m logic.movement_summary input_keypoints.json --output summary.json --experimental-behaviors
```

The CLI also accepts `--confidence`, `--segment-seconds`, and `--smoothing-window`.
All remaining rule settings are available through `MovementConfig`:

```python
from logic.movement_summary import MovementConfig, analyze_file, analyze_dataset

settings = MovementConfig(
    confidence_threshold=0.7,
    segment_seconds=3.0,
    experimental_behavior_candidates=False,
)
summary = analyze_file("input_keypoints.json", "summary.json", config=settings)
# Or: summary = analyze_dataset(raw_dataset, config=settings)

# Use only when you intentionally want to send the evidence to Gemini:
from ai.ai import analyze_data
interpretation = analyze_data(summary)
```

`analyze_file` creates the output's parent directories and refuses to overwrite
the input keypoint file. `analyze_dataset` does not mutate its input. The logic
requires no packages beyond Python's standard library. AI calls require the
Gemini/dotenv dependencies and environment settings in the project's requirements.

## Geometry and available measurements

| Feature | Reference and interpretation |
| --- | --- |
| `body` | Midpoint of front/rear trunk landmarks; both and a nondegenerate axis are required. |
| `head_relative` | Nose position relative to torso centre, rotated into torso coordinates. |
| `head_orientation` | Image-plane angle from ear-base midpoint toward nose. |
| `head_orientation_relative_to_body` | Head-axis angle minus torso-axis angle. |
| `left_ear_orientation_relative_to_head`, `right_ear_orientation_relative_to_head` | Ear base-to-tip angle minus head-axis angle. |
| `tail_relative` | Tail tip minus tail base in image axes; removes translation, but not body rotation. |
| `tail_orientation_relative_to_body` | Tail base-to-tip angle minus torso-axis angle. |
| Four `*_paw` features | Each paw relative to torso centre, rotated into torso coordinates. |

Front/rear trunk labels are geometric proxies for shoulder/hip. The model's
antler labels are excluded. A reliable pose in a different camera perspective
can still change projected angles and apparent distances.

Linear features include:

- `average_displacement_pixels`, `maximum_displacement_pixels`: consecutive-pair
  travel, not beginning-to-end displacement.
- `total_distance_pixels`: total travel across usable pairs only.
- `maximum_excursion_pixels`: largest distance from the first point of a
  continuous valid run. Different runs are never joined.
- `average_speed_pixels_per_second`: total travel / observed pair time.
- `maximum_speed_pixels_per_second`, `median_speed_pixels_per_second`.
- `average_speed_diagonals_per_second`: average pixel speed / image diagonal.

Angular features include average/maximum angular speed, maximum excursion from
a continuous run's initial angle, and angular direction reversals. Wrapped
angles are handled across the -180/180 boundary. Reversals use cumulative
departure from an extremum to ignore sub-deadband jitter near a turnaround.

Every feature reports `valid_frame_percentage`, `valid_pair_percentage`,
`valid_measurement_count`, `observed_pair_seconds`, `observed_time_percentage`,
`data_sufficient`, and explicit insufficiency reasons. Numeric estimates can
still appear with insufficient data; the AI must consult that quality flag.
An unobservable numeric measurement is `null`, not zero.

Motion statistics also include meaningful direction changes, sustained movement
bouts, active-time percentage, and longest continuous stationary duration. Bouts
require a minimum observed duration and use a lower exit threshold to reduce
fragmentation. Stationary runs must stay below the speed threshold and within a
bounded radius; gaps break both movement and stationary runs.

## Rules and statuses

Default candidates cover body motion/stationarity, relative head motion,
head-angle changes, independent ear-angle movement, tail movement/oscillation,
relative movement of each paw, and repetitive front-paw motion. Oscillation and
repetitive-paw candidates do not establish tail signaling or kneading.

Each candidate has a label, rule ID/version, interval, status, readable rule,
numeric evidence, `score: null`, and persistent interpretation limitations:

- `detected`: sufficient data exists and this engineering rule was met.
- `not_detected`: sufficient data exists but this rule was not met.
- `insufficient_data`: the observations cannot support an assessment.

`body_stationary` means at least one qualifying stationary run occurred within
the stated interval. It can coexist with `body_moving` when a clip contains
both movement and stillness. A global result does not imply that every window
had the same result.

Optional `grooming_like_motion_candidate` requires simultaneous nose/paw
proximity, torso-relative head movement and low body speed on the same observed
pairs, plus a stationary body run in the window. It does not confirm contact,
licking or grooming. Optional `play_like_activity_candidate` requires body
movement/direction changes and a moving reliable front paw; other activities
can produce the same pattern. Both are disabled by default and need validation.

## Starting settings

| Setting | Default |
| --- | --- |
| Confidence threshold | 0.60, while always respecting exporter `valid: false` |
| Window length / maximum windows | 3 seconds / 60; long clips use larger windows |
| Median smoothing | 3 frames; 1 disables it |
| Joint frame / pair coverage | 60% / 50% |
| Minimum consecutive measurements / observed time | 3 / 0.5 seconds |
| Minimum observed-time fraction | 50% of elapsed time |
| Maximum pair interval | 1.5 / FPS, with consecutive frame numbers required |
| Minimum axis length | 0.005 image diagonals |
| Isolated-spike speed | Both legs exceed 4 diagonals/second while neighbours agree |
| Body / relative movement speed | 0.02 / 0.01 diagonals/second |
| Angular motion / reversal deadband | 15 degrees/second / 2 degrees |
| Linear direction change | 60 degrees |
| Minimum movement bout / exit ratio | 0.2 seconds / 70% of the entry threshold |
| Stationary run / radius | 1 second / 0.01 image diagonals |
| Tail oscillation reversals / excursion | 3 / 20 degrees |
| Repetitive-paw direction changes | 3 per paw |
| Optional nose/paw distance / co-occurrence | 0.25 trunk lengths / 30% of joint observed pair time |

These values have **not** been established as biological standards. Synthetic
tests check the arithmetic and missing-data behavior; they do not establish
accuracy on cats. Validate thresholds against manually annotated intervals in
multiple videos and evaluate on different clips/cats from those used for tuning.
Do not tune behavior detection to filenames such as "play" or "stare".

## Saved summary and AI integration

The full output schema is `movement-summary/1`. It contains `video`,
`data_quality`, `measurements`, `candidates`, `intervals`, `analysis_settings`,
`warnings`, and `limitations`. Per-interval measurement tables support review.

`ai.ai.build_prompt` accepts raw keypoints or validates this saved summary.
For the AI request it constructs a smaller `movement-evidence/1` transport view:
whole-clip measurements, interval decision evidence, shared rules, and shared
limitations referenced by IDs. It omits duplicate per-window measurement tables.
This prompt-only view is not a substitute for the saved summary input contract.
Previously generated `behavior-summary/1` summaries remain readable; unsupported
summary versions fail explicitly rather than being misread as raw keypoints.

Overall quality is `good` only when all tracked features pass their own gates,
`limited` when some pass, and `poor` when none pass. Missing tail observations
must not erase reliable head evidence. Inspect per-feature quality and the
interval's evidence instead of relying on the overall rating alone.

Camera motion, zoom and 3D effects are not corrected. Pixel speed is not physical
speed. Image-diagonal normalization does not remove camera-distance differences.
Smoothing and spike filtering can suppress genuine fast movements. No gaze,
attention, emotion, diagnosis, or confirmed semantic behavior is established.

## Verification

```powershell
python -m unittest discover -s tests -v
```

Tests cover stationary and translating bodies, body rotation, independent head
turns, tail oscillation, shared-motion cancellation, sustained-bout requirements,
missing/out-of-bounds/invalid landmarks, time gaps, isolated spikes, resizing,
same-time grooming cues, single-visible-paw activity, export protection and AI
prompt routing. All three supplied JSON clips are also checked for finite,
contract-valid output without making API calls.
