# NMS Enhanced + DIoU Guide

This guide shows how to integrate **Soft-NMS** and **DIoU (Distance-IoU)** into your YOLOv6 inference pipeline.

## Files Overview

- **`enhance/scripts/nms_enhanced.py`** – Core NMS implementations:
  - `soft_nms_numpy()` – Linear or Gaussian soft suppression
  - `diou_nms_numpy()` – DIoU-based suppression
  - `enhanced_nms()` – Unified API for all methods

- **`enhance/scripts/adaptive_scale.py`** – Lightweight image resizing utility (uses Pillow)

- **`enhance/scripts/train_with_utf8_conf.py`** – Wrapper to read config files with UTF-8 encoding (fixes UnicodeDecodeError on Windows)

- **`yolov6/core/inferer.py`** – Modified to support `--nms` and `--nms-method` options (auto-imports from `enhance/scripts`)

- **`tools/infer.py`** – Already supports enhanced NMS via CLI flags

---

## Available NMS Methods

### 1. **Standard NMS** (original)
```bash
# Default behavior – standard Hard NMS
python tools/infer.py --weights best.pt --source input.jpg --nms original
```

### 2. **Soft-NMS (Gaussian)**
Smoothly decays confidence scores of overlapping boxes instead of removing them outright.
```bash
python tools/infer.py \
  --weights best.pt \
  --source input.jpg \
  --nms enhanced \
  --nms-method soft \
  --nms-soft-method gaussian \
  --nms-sigma 0.5 \
  --conf-thres 0.25 \
  --iou-thres 0.45
```

### 3. **Soft-NMS (Linear)**
Linearly decays confidence scores.
```bash
python tools/infer.py \
  --weights best.pt \
  --source input.jpg \
  --nms enhanced \
  --nms-method soft \
  --nms-soft-method linear \
  --conf-thres 0.25 \
  --iou-thres 0.45
```

### 4. **DIoU-NMS**
Uses Distance-IoU instead of standard IoU for better suppression of nearby boxes.
```bash
python tools/infer.py \
  --weights best.pt \
  --source input.jpg \
  --nms enhanced \
  --nms-method diou \
  --conf-thres 0.25 \
  --iou-thres 0.45
```

---

## Webcam Inference with Enhanced NMS

Your example command with Soft-NMS + Gaussian applied:

```bash
python tools/infer.py \
  --weights runs/train/exp15/weights/best_ckpt.pt \
  --webcam \
  --webcam-addr 0 \
  --yaml data/Own/data.yaml \
  --device cpu \
  --conf-thres 0.25 \
  --iou-thres 0.45 \
  --nms enhanced \
  --nms-method soft \
  --nms-soft-method gaussian \
  --nms-sigma 0.5 \
  --view-img \
  --project runs/inference \
  --name webcam_live
```

**Explanation of new flags:**
- `--nms enhanced` – Activates enhanced NMS (soft-NMS or DIoU)
- `--nms-method soft` – Use soft suppression (alternatives: `diou`, `nms`)
- `--nms-soft-method gaussian` – Use Gaussian decay (alternative: `linear`)
- `--nms-sigma 0.5` – Decay rate for Gaussian (lower = sharper decay)

---

## Training with DIoU Loss

Your config file `configs/adapt_diou_yolov6.py` already specifies `iou_type='diou'` in the model head. Use the training wrapper to read it with UTF-8:

```bash
python enhance/scripts/train_with_utf8_conf.py \
  --conf-file configs/adapt_diou_yolov6.py \
  -- \
  --data-path data/Own/data.yaml \
  --check-images \
  --check-labels \
  --batch-size 8 \
  --img-size 640 \
  --epoch 50
```

**Note:** The `--` separates args for the wrapper from args forwarded to `tools/train.py`.

---

## Parameter Reference

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--nms` | str | `original` | NMS mode: `original` or `enhanced` |
| `--nms-method` | str | `soft` | When enhanced: `soft`, `diou`, or `nms` |
| `--nms-soft-method` | str | `gaussian` | For soft-NMS: `gaussian` or `linear` |
| `--nms-sigma` | float | `0.5` | Decay rate for Gaussian (0.1–1.0 typical) |
| `--conf-thres` | float | `0.4` | Confidence threshold for detection |
| `--iou-thres` | float | `0.45` | IoU threshold for NMS suppression |
| `--max-det` | int | `1000` | Max detections per image |

---

## Troubleshooting

### **Issue:** `UnicodeDecodeError` when loading config files on Windows
**Solution:** Use the UTF-8 wrapper:
```bash
python enhance/scripts/train_with_utf8_conf.py --conf-file configs/your_config.py -- [other args]
```

### **Issue:** Enhanced NMS falls back to standard NMS
Check the logs. If `nms_enhanced.py` is not found, ensure:
- `enhance/scripts/nms_enhanced.py` exists
- `yolov6/core/inferer.py` is using the updated version (it auto-adds the path)

### **Issue:** Slow inference with CPU and enhanced NMS
Soft-NMS and DIoU-NMS are slightly slower than hard NMS due to per-class processing. This is expected. For faster CPU inference, stick with `--nms original`.

---

## Performance Tips

1. **For better accuracy (slower):** Use Soft-NMS with Gaussian decay
   ```bash
   --nms enhanced --nms-method soft --nms-soft-method gaussian --nms-sigma 0.3
   ```

2. **For speed (moderate accuracy):** Use DIoU-NMS
   ```bash
   --nms enhanced --nms-method diou --iou-thres 0.5
   ```

3. **For production (original):** Stick with standard NMS
   ```bash
   --nms original
   ```

---

## Example: Full Inference Pipeline

```bash
# 1. Test config reading (UTF-8)
python enhance/scripts/train_with_utf8_conf.py \
  --conf-file configs/adapt_diou_yolov6.py \
  --print-conf

# 2. Train model with DIoU loss
python enhance/scripts/train_with_utf8_conf.py \
  --conf-file configs/adapt_diou_yolov6.py \
  -- \
  --data-path data/Own/data.yaml \
  --batch-size 8 \
  --epoch 50

# 3. Run inference with Soft-NMS on trained model
python tools/infer.py \
  --weights runs/train/exp*/weights/best_ckpt.pt \
  --source data/images \
  --yaml data/Own/data.yaml \
  --nms enhanced \
  --nms-method soft \
  --nms-sigma 0.5 \
  --conf-thres 0.25 \
  --view-img

# 4. Live webcam inference with Soft-NMS
python tools/infer.py \
  --weights runs/train/exp*/weights/best_ckpt.pt \
  --webcam \
  --webcam-addr 0 \
  --yaml data/Own/data.yaml \
  --device cpu \
  --nms enhanced \
  --nms-method soft \
  --nms-soft-method gaussian \
  --nms-sigma 0.5 \
  --view-img
```

---

## References

- **Soft-NMS**: Bodla et al. "Soft-NMS -- Improving Object Detection with One Line of Code"
- **DIoU**: Zheng et al. "Distance-IoU Loss: Faster and Better Learning for Bounding Box Regression"
- **YOLOv6**: https://github.com/meituan/YOLOv6

