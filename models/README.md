# Preserved Models

- `best_classifier.pt`: ConvNeXt-Tiny classifier trained from normalized CVAT annotations; heads `team` and `car_model`; best validation score 0.978947.
- `best_detector.pt`: YOLO26s detector fine-tuned from `yolo26s.pt` at imgsz 1920, batch 4, with train-only hflip/brightness/contrast augmentation; validation mAP50 0.965000, mAP50-95 0.873613.
- `best_detector_legacy_high_map.pt`: copied from `outputs/training/detector/f1_car_detector-2/weights/best.pt`; older smaller validation split, mAP50 0.995000, mAP50-95 0.900360.
