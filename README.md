# Comp-Arch-Team-Project

## Movement logic for single-cat videos

`logic/movement_summary.py` is the primary logic entry point. It accepts exported
DeepLabCut keypoints from different single-cat activities and produces measured
movement evidence, joint observation quality, and auditable rule results. It
uses the Python standard library; it does not run DeepLabCut or call the AI API.

```powershell
python -m logic.movement_summary analysis/DeepLabCut-Processing/outputs/test2_cat_stare_keypoints.json --output analysis/DeepLabCut-Processing/outputs/summaries/test2_cat_stare_summary.json
```

If Python is not on PATH in this machine's shell, use the installed executable:

```powershell
& "$env:USERPROFILE/miniconda3/python.exe" -m logic.movement_summary analysis/DeepLabCut-Processing/outputs/test2_cat_stare_keypoints.json --output summary.json
```

The same command works with play, grooming, walking, resting, or other supported
single-cat clips. The module measures observable motion; it does not guarantee
recognition of every behavior. Grooming-like and play-like hypotheses are
experimental and require `--experimental-behaviors`. Their thresholds require
validation against manually reviewed videos.

`ai.ai.analyze_data()` accepts either raw keypoints or the saved
`movement-summary/1` summary. Raw input uses the new logic automatically. The
prompt includes compact evidence and rule limitations. Gemini is called only
when `analyze_data()` is invoked; generating summaries or calling `build_prompt()`
works offline. Existing `behavior-summary/1` summaries remain supported.

See [logic/README.md](logic/README.md) for definitions, configuration, quality
checks, API use, and limitations. Generated examples for the three supplied clips
are in `analysis/DeepLabCut-Processing/outputs/summaries/`.

Run the logic and existing preprocessor tests without API credentials:

```powershell
python -m unittest discover -s tests -v
```

## DLC behavior evidence preprocessing

The `ai/behavior_preprocessor.py` module converts a DeepLabCut keypoint JSON file
into a compact, time-windowed summary before it is sent to the AI. It validates
timestamps and video metadata, filters invalid or low-confidence landmarks,
reports per-landmark coverage, and measures body, head-relative, tail-relative,
and paw-relative motion. Candidate statuses are `detected`, `not_detected`, or
`insufficient_data`; candidates describe measured motion and do not claim a
specific behavior or emotional state. Speeds are normalized by the image
diagonal, and thresholds are starting values that require validation.

Run it directly to inspect or save the summary:

```powershell
python -m ai.behavior_preprocessor analysis\DeepLabCut-Processing\outputs\test1_cat_play_keypoints.json --output summary.json
```

The confidence, segment length, and motion thresholds can be changed through the
CLI options or `PreprocessConfig`. Motion thresholds are in image diagonals per
second.

This earlier preprocessor remains available for existing callers and its saved
summaries. New raw-data AI calls use `logic/movement_summary.py` as described
above. Motion thresholds should be validated against labeled videos before
being used as behavior standards.
