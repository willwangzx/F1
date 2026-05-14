# F1 Recognition

This repository is an image-based F1 car recognition pipeline. It trains a single-class
YOLO detector to find F1 cars, crops each detected car, then trains a multi-head image
classifier to predict labels such as `team`, `driver`, and `car_model`.

The project is meant to support a practical annotation-to-training loop:

1. Convert or write annotations in a simple CSV format.
2. Build detector and classifier datasets from those annotations.
3. Train a YOLO detector.
4. Train a crop classifier.
5. Run end-to-end inference and review mistakes.

## Repository Layout

```text
F1/
|-- f1_recognition/              # Shared annotation, IO, metric, and classifier code
|-- configs/
|   `-- project.example.yaml      # Classifier training defaults
|-- templates/
|   |-- annotations_template.csv  # Positive bounding-box annotation template
|   `-- negative_images_template.csv
|-- models/                       # Preserved trained checkpoints
|-- annotation/                    # Example CVAT XML exports
|-- prepare_dataset.py            # Build detector/classifier datasets
|-- train_detector.py             # Train YOLO detector
|-- train_classifier.py           # Train multi-head classifier
|-- predict.py                    # Single-image detection + classification
|-- visualize_cars.py             # Batch visualization and prediction CSV export
|-- analyze_errors.py             # Classifier error analysis
|-- mine_detector_errors.py       # Detector false-positive / missed-GT mining
|-- convert_cvat_to_annotations.py
|-- requirements.txt
`-- README.md
```

## Environment

The pinned requirements use PyTorch CUDA 13.0 wheels and Ultralytics YOLO.

```powershell
python -m pip install -r requirements.txt
```

If you use conda, activate your environment first:

```powershell
conda activate D:\Anaconda\envs\torch313
python -m pip install -r requirements.txt
```

## Quick Inference With Preserved Models

Pretrained checkpoints are stored in `models/`:

- `models/best_detector.pt`
- `models/best_classifier.pt`
- `models/best_detector_legacy_high_map.pt`

Run one image end to end:

```powershell
python predict.py `
  --image D:\Arc\F1\_W721720.jpg `
  --detector .\models\best_detector.pt `
  --classifier .\models\best_classifier.pt `
  --output-dir .\outputs\predictions
```

This writes:

- `outputs/predictions/*_predictions.json`
- `outputs/predictions/*_predictions.jpg`

Run a folder of images and generate a review CSV:

```powershell
python visualize_cars.py `
  --input D:\Arc\F1 `
  --detector .\models\best_detector.pt `
  --classifier .\models\best_classifier.pt `
  --output-dir .\outputs\visualize `
  --recursive
```

This writes visualized images plus `outputs/visualize/predictions_summary.csv`.

## Annotation Format

The main input is a CSV. Start from `templates/annotations_template.csv`.

Required columns:

- `image_path`
- `x1`, `y1`, `x2`, `y2`
- `team`
- `driver`
- `car_model`

Optional columns:

- `split`: `train`, `val`, or blank
- `source`: annotation source, defaults to `private`
- `batch_id`: keeps related burst/sequence images in the same split
- `tcam_visible`: values like `yes/no`, `true/false`, or `1/0`
- `tcam_color`
- `tcam_x1`, `tcam_y1`, `tcam_x2`, `tcam_y2`

Use `unknown_driver` when the driver cannot be identified confidently. A convenient
first-pass `car_model` convention is `2026_<Team>`, for example `2026_Ferrari`.

If your CSV uses relative image paths, pass `--image-root` to the dataset preparation
and conversion commands.

## Convert CVAT XML

CVAT image XML exports can be converted into the project CSV format:

```powershell
python convert_cvat_to_annotations.py `
  --xml .\annotation\annotations.xml `
  --image-root D:\Arc\F1 `
  --output .\outputs\annotations.csv `
  --label f1_car
```

The converter expects each `f1_car` box to carry optional CVAT attributes named
`team`, `driver`, and `car_model`.

## Prepare Datasets

Build a YOLO detector dataset and classifier crop manifest:

```powershell
python prepare_dataset.py `
  --annotations .\outputs\annotations.csv `
  --image-root D:\Arc\F1 `
  --output-dir .\outputs\dataset `
  --max-long-edge 2560 `
  --val-ratio 0.2
```

Useful options:

- `--max-long-edge 0`: keep original image pixels.
- `--negative-images .\templates\negative_images_template.csv`: add detector-only hard negatives.
- `--export-classifier-shards`: export preprocessed classifier tensor shards for faster training.

Key outputs:

- `outputs/dataset/detector/data.yaml`
- `outputs/dataset/classifier/manifest.csv`
- `outputs/dataset/metadata/label_maps.json`
- `outputs/dataset/metadata/dataset_summary.json`
- `outputs/dataset/metadata/images.csv`

For full-resolution detector-only experiments with offline train augmentations:

```powershell
python prepare_detector_augmented.py `
  --annotations .\outputs\annotations.csv `
  --image-root D:\Arc\F1 `
  --output-dir .\outputs\dataset_f1_augmented_original
```

## Train The Detector

Train a YOLO detector from the prepared `data.yaml`:

```powershell
python train_detector.py `
  --data .\outputs\dataset\detector\data.yaml `
  --model yolov8n.pt `
  --epochs 80 `
  --imgsz 1280 `
  --batch 4 `
  --workers 0
```

Useful options:

- `--model yolov8s.pt` or another Ultralytics checkpoint for larger models.
- `--auto-batch`: probe CUDA memory and use the largest safe batch size.
- `--oom-fallback-batch 2`: retry once with a smaller batch after CUDA OOM.
- `--cache-mode ram` or `--cache-mode disk`: use Ultralytics image caching.
- `--exist-ok`: allow writing into an existing run directory.

The best checkpoint is written under:

```text
outputs/training/detector/<run_name>/weights/best.pt
```

## Train The Classifier

Train the crop classifier:

```powershell
python train_classifier.py `
  --manifest .\outputs\dataset\classifier\manifest.csv `
  --label-maps .\outputs\dataset\metadata\label_maps.json `
  --config .\configs\project.example.yaml `
  --output-dir .\outputs\training\classifier
```

The classifier code supports the heads `team`, `driver`, and `car_model`. The sample
config currently trains `team` and `car_model`; pass explicit heads if you want all
three:

```powershell
python train_classifier.py `
  --manifest .\outputs\dataset\classifier\manifest.csv `
  --label-maps .\outputs\dataset\metadata\label_maps.json `
  --config .\configs\project.example.yaml `
  --heads team driver car_model `
  --output-dir .\outputs\training\classifier
```

Useful options:

- `--backbone resnet50`, `convnext_tiny`, `efficientnet_b3`, or supported YOLO cls backbones.
- `--cache-mode ram`: cache decoded crops in memory.
- `--shard-dir .\outputs\dataset\classifier\shards`: train from exported tensor shards.
- `--val-cache-mode val_tensors`: cache transformed validation tensors after first use.
- `--auto-batch`: probe CUDA memory before training.
- `--balanced-head team driver`: class-balance sampling by one or more heads.
- `--freeze-backbone --freeze-backbone-epochs 3`: staged fine-tuning.
- `--grad-accum-steps 2`: use a larger effective batch size.
- `--quick-val-ratio 0.25 --val-every 2`: speed up experiments.

Key outputs:

- `outputs/training/classifier/best_classifier.pt`
- `outputs/training/classifier/eval_classifier.csv`
- `outputs/training/classifier/training_history.json`
- `outputs/training/classifier/run_summary.json`

## Analyze Classifier Errors

After classifier training, summarize validation mistakes:

```powershell
python analyze_errors.py `
  --eval-csv .\outputs\training\classifier\eval_classifier.csv `
  --output-dir .\outputs\analysis
```

Outputs:

- `outputs/analysis/metrics.json`
- `outputs/analysis/confusions.json`
- `outputs/analysis/report.md`

## Mine Detector Errors

First run `visualize_cars.py` on labeled images to create `predictions_summary.csv`,
then compare predictions against the annotation CSV:

```powershell
python mine_detector_errors.py `
  --annotations .\outputs\annotations.csv `
  --predictions .\outputs\visualize\predictions_summary.csv `
  --image-root D:\Arc\F1 `
  --output-dir .\outputs\detector_error_mining
```

This writes detector metrics, a hard-sample review CSV, and candidate negative-image
rows that can be reviewed before feeding them back through `prepare_dataset.py`.

To crop background false positives for review:

```powershell
python mine_detector_errors.py `
  --annotations .\outputs\annotations.csv `
  --predictions .\outputs\visualize\predictions_summary.csv `
  --negative-crop-dir .\outputs\hard_negative_crops
```

## Helper Tools

Preview or apply stable image renames:

```powershell
python rename_images.py `
  --image-dir D:\Arc\F1 `
  --prefix f1_ `
  --digits 5 `
  --mapping .\outputs\rename_mapping.csv
```

Add `--apply` to actually rename files. Add `--update-csv` to rewrite `image_path`
values in an annotation CSV.

Build a detector dataset where all labeled images are used for training and a random
subset is also used for validation:

```powershell
python make_detector_trainall_randomval.py `
  --source-detector .\outputs\dataset\detector `
  --output .\outputs\dataset_detector_trainall_randomval `
  --val-ratio 0.2
```

## Current Scope

This repo intentionally keeps the first production loop simple:

- Still-image detection and classification only.
- No HTTP API.
- No video tracking.
- No hard-coded `T-cam color -> driver` rule.
- No separate T-cam branch model yet.

The priority is to keep the data preparation, training, evaluation, and inference loop
easy to inspect and iterate.
