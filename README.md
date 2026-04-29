# F1 赛车识别项目

这个工程实现了首版 `F1 赛车检测 + 多头分类` 流程：

- `train_detector.py` 训练单类检测器，先把图里的 F1 赛车找出来
- `train_classifier.py` 训练多头分类器，同时输出 `team / driver / car_model`
- `predict.py` 做端到端单图推理并保存可视化结果
- `analyze_errors.py` 读取分类验证输出，生成混淆分析

## 目录

```text
F1/
├─ f1_recognition/
├─ configs/
├─ templates/
├─ prepare_dataset.py
├─ train_detector.py
├─ train_classifier.py
├─ predict.py
├─ analyze_errors.py
└─ requirements.txt
```

## 1. 环境准备

推荐直接使用你现有的 `D:\Anaconda\envs\torch313`。

```powershell
conda activate D:\Anaconda\envs\torch313
python -m pip install -r requirements.txt
```

## 2. 你需要准备的标注

首版输入统一为一个 CSV。模板见 [templates/annotations_template.csv](/d:/Projects/F1/templates/annotations_template.csv:1)。

必填列：

- `image_path`
- `x1`, `y1`, `x2`, `y2`
- `team`
- `driver`
- `car_model`

建议填写的列：

- `batch_id`
  用于把连拍样本分在同一个 split，减少数据泄漏
- `tcam_visible`
  `yes/no`
- `tcam_color`
  如果你愿意记录颜色，可以先写；不写也不影响首版训练
- `tcam_x1`, `tcam_y1`, `tcam_x2`, `tcam_y2`
  如果后续你想升级成 T 架局部双分支模型，这组坐标会很有用

`driver` 标注建议：

- 能明确区分时，直接标具体车手
- 看不清 T 架、号码、头盔时，标 `unknown_driver`
- 先用你的人工判断把真值标对，不需要现在就把 T 架颜色规则写成硬编码映射

`car_model` 首版建议统一成 `2026_<team>`，例如：

- `2026_Ferrari`
- `2026_McLaren`
- `2026_RedBull`

## 3. 数据准备

把标注 CSV 转成检测与分类训练集：

```powershell
python prepare_dataset.py `
  --annotations .\templates\annotations_template.csv `
  --output-dir .\outputs\dataset `
  --max-long-edge 2560 `
  --val-ratio 0.2
```

输出结果包括：

- `outputs/dataset/detector/data.yaml`
- `outputs/dataset/classifier/manifest.csv`
- `outputs/dataset/metadata/label_maps.json`
- `outputs/dataset/metadata/dataset_summary.json`

## 4. 训练检测器

```powershell
python train_detector.py `
  --data .\outputs\dataset\detector\data.yaml `
  --model yolov8n.pt `
  --epochs 80 `
  --imgsz 1280 `
  --batch 4
```

如果你显存比较宽松，可以后续尝试：

- `yolov8s.pt`
- 更大的 `imgsz`
- 更长训练轮数

## 5. 训练多头分类器

```powershell
python train_classifier.py `
  --manifest .\outputs\dataset\classifier\manifest.csv `
  --label-maps .\outputs\dataset\metadata\label_maps.json `
  --config .\configs\project.example.yaml `
  --output-dir .\outputs\training\classifier
```

分类器默认使用：

- `resnet50` backbone
- 单 backbone + 三个 head
- 输出 `team / driver / car_model`

训练结束后重点关注：

- `outputs/training/classifier/best_classifier.pt`
- `outputs/training/classifier/eval_classifier.csv`
- `outputs/training/classifier/training_history.json`

## 6. 分析分类错误

```powershell
python analyze_errors.py `
  --eval-csv .\outputs\training\classifier\eval_classifier.csv `
  --output-dir .\outputs\analysis
```

分析输出包括：

- 总体准确率
- `tcam_visible` 子集上的车手准确率
- 同队双车手混淆趋势

## 7. 单图推理

```powershell
python predict.py `
  --image D:\Arc\F1\_W721720.jpg `
  --detector .\outputs\training\detector\f1_car_detector\weights\best.pt `
  --classifier .\outputs\training\classifier\best_classifier.pt `
  --output-dir .\outputs\predictions
```

会输出：

- `*_predictions.json`
- `*_predictions.jpg`

## 8. 建议的试点节奏

- 先标 `80-120` 张，不要一开始就全量 238 张
- 优先覆盖：
  - 同队两位车手
  - T 架清晰和不清晰两种情况
  - 多车同框
  - 模糊、反光、遮挡
- 先验证 `team`、`driver`、`car_model` 三个头是否都能正常学习
- 如果 `driver` 在 `tcam_visible=yes` 子集上明显更强，再升级双分支模型

## 9. 当前实现的边界

当前版本故意保持 v1 简洁：

- 不做 HTTP API
- 不做视频跟踪
- 不写死 `T架颜色 -> 车手` 规则
- 不做双分支 `整车 + T架局部` 网络

它的目标是先让你把数据、训练、评估、推理的完整闭环跑通。
