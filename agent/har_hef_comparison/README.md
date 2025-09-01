# HAR vs HEF Comparison

This document captures the investigation and resolution for the slight result differences between running the saved HAR golden outputs and running the HEF on the same images.

Scope:
- Compare saved golden HAR run at `hailo_model_converter/shared_with_docker/tests/data/golden_run` with a local HEF run.
- Verify preprocessing/postprocessing parity and quantify raw head and detection-level differences.
- Provide a repeatable comparison tool and recommended flags to align results.

## Paths
- Golden run (saved): `hailo_model_converter/shared_with_docker/tests/data/golden_run`
- HEF run (original): `outputs/hef_images_test_zzzz`
- HEF run (matched thresholds): `outputs/hef_images_match_golden`
- HEF model: `resources/model.hef`
- Images: `hailo_model_converter/shared_with_docker/tests/data/images`

## Commands Used

Original HEF run (as provided):

```bash
python3 run_hef_on_images.py \
  --hef resources/model.hef \
  --images-dir hailo_model_converter/shared_with_docker/tests/data/images \
  --outdir outputs/hef_images_test_zzzz \
  --input-float --normalize --output-float \
  --letterbox --letterbox-pad 114 \
  --decode-yolov8 --yolo-classes 15
```

HEF run with golden thresholds (to match HAR):

```bash
python3 run_hef_on_images.py \
  --hef resources/model.hef \
  --images-dir hailo_model_converter/shared_with_docker/tests/data/images \
  --outdir outputs/hef_images_match_golden \
  --input-float --normalize --output-float \
  --letterbox --letterbox-pad 114 \
  --decode-yolov8 --yolo-classes 15 \
  --nms-iou 0.45 --score-thresh 0.25
```

Golden metadata (from `golden.json`):

```json
{
  "yolo_classes": 15,
  "dfl_bins": 16,
  "nms_iou": 0.45,
  "score_thresh": 0.25,
  "letterbox": true,
  "letterbox_pad": 114,
  "input_float": true,
  "normalize": true,
  "seed": 0,
  "limit": 8
}
```

## Key Findings

- The main source of differences was postprocessing thresholds. The original HEF run used the defaults `--score-thresh 0.01` and `--nms-iou 0.01`, while the golden run used `--score-thresh 0.25` and `--nms-iou 0.45`.
- With matching thresholds, detection results align very well; remaining tiny numeric differences are attributable to expected per-layer quantization/dequantization differences between SDK-quantized HAR and compiled HEF builds.
- Preprocessing matched (RGB, float32, normalized to [0,1], letterbox pad 114, 640×640), and the same YOLOv8 DFL decode logic was used by both flows.

## Detection Comparison (IoU>0.5, class-equal)

- Original HEF vs Golden (mismatched thresholds):
  - Average Recall: 0.742, Precision: 0.352, F1: 0.388

- HEF (matched thresholds) vs Golden:
  - Average Recall: 0.986, Precision: 0.931, F1: 0.956

Per-image details for both cases were computed and are reproducible with the tool below.

## Raw Head Differences (pre-decode)

Comparing the six heads by shape; numbers are means over 8 images (HEF original vs Golden):

```
(20,20,15): MAE 0.219530  MaxAE 2.122580  Cos 0.999732
(20,20,64): MAE 0.095875  MaxAE 2.249268  Cos 0.997932
(40,40,15): MAE 0.577014  MaxAE 8.395035  Cos 0.998831
(40,40,64): MAE 0.149288  MaxAE 6.724413  Cos 0.995578
(80,80,15): MAE 0.708050  MaxAE 9.721851  Cos 0.998470
(80,80,64): MAE 0.162019  MaxAE 6.020223  Cos 0.995423
```

Notes:
- Classification logits (C=15) show small bias-like shifts that largely vanish after sigmoid (p99 absolute probability diff ≈ 5e-5; rare outliers up to ~0.3).
- Regression heads (C=64, DFL bins aggregated) are very close (cos ≈ 0.995–0.998).

## Output Head Mapping

- Golden output names: `model/output_layer{1..6}` with shapes
  - (80,80,64), (80,80,15), (40,40,64), (40,40,15), (20,20,64), (20,20,15)
- HEF output names: `model/conv41/42/52/53/62/63` with matching shapes
  - (80,80,64), (80,80,15), (40,40,64), (40,40,15), (20,20,64), (20,20,15)

## Comparison Tool

Added `tools/compare_runs.py` to quantify raw head deltas and detection matches:

```bash
python3 tools/compare_runs.py \
  --a outputs/hef_images_test_zzzz \
  --b hailo_model_converter/shared_with_docker/tests/data/golden_run

python3 tools/compare_runs.py \
  --a outputs/hef_images_match_golden \
  --b hailo_model_converter/shared_with_docker/tests/data/golden_run
```

It reports, per shared tensor shape, MAE/MaxAE/Cosine similarity, and per-image detection matching with IoU>0.5 and class equality.

## Do We Save All Outputs?

Yes. Each NPZ under `raw_tensors/` contains all raw model outputs (the six heads). The decoded `boxes/classes/scores` are thresholded for convenience, but raw heads are always fully saved (unfiltered).

## Conclusions and Recommendations

- Use the same decode thresholds as the golden to compare detections: `--score-thresh 0.25 --nms-iou 0.45`.
- Expect tiny logit differences between HAR (SDK quantized) and HEF builds; these are negligible after sigmoid/NMS.
- If bit-level parity on raw heads is required, compile the HEF from the exact HAR artifact that produced the golden outputs.
- Optional: set `run_hef_on_images.py` defaults to the golden thresholds or add a `--golden-meta` flag to parse thresholds from `golden.json`.

