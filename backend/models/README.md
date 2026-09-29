# Models

| File | Model | Classes | Made by |
|---|---|---|---|
| `finder.onnx` | YOLO11n-seg, 640 px (v2, trained with other-produce hard negatives) | onion, coin | `ml/notebooks/02_train_finder.ipynb` + `05_hard_negatives.ipynb` |
| `classifier.onnx` | YOLO11n-cls, 224 px (v2) | good, black_mould, rotten, sprouted, damaged, not_onion | `ml/notebooks/04_train_classifier.ipynb` + `05_hard_negatives.ipynb` |

Both run on CPU with onnxruntime. Either can be missing: without the finder, use the printed calibration sheet (onions found by colour); without the classifier, defects aren't checked.
