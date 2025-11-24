# Quick Reference: Your Webcam Inference Commands

## Original Command (with bug fix)
Your original command that caused UnicodeDecodeError:
```bash
python tools/infer.py \
  --weights runs/train/exp15/weights/best_ckpt.pt \
  --webcam \
  --webcam-addr 0 \
  --yaml data/Own/data.yaml \
  --device cpu \
  --conf-thres 0.25 \
  --view-img \
  --project runs/inference \
  --name webcam_live
```

---

## Enhanced Commands: Choose One

### Option 1: Soft-NMS (Gaussian) ⭐ Recommended for best accuracy
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
  --name webcam_softnms_gaussian
```

### Option 2: Soft-NMS (Linear) - Faster decay
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
  --nms-soft-method linear \
  --view-img \
  --project runs/inference \
  --name webcam_softnms_linear
```

### Option 3: DIoU-NMS - Geometry-aware suppression
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
  --nms-method diou \
  --view-img \
  --project runs/inference \
  --name webcam_diounms
```

### Option 4: Original NMS (Baseline)
```bash
python tools/infer.py \
  --weights runs/train/exp15/weights/best_ckpt.pt \
  --webcam \
  --webcam-addr 0 \
  --yaml data/Own/data.yaml \
  --device cpu \
  --conf-thres 0.25 \
  --view-img \
  --project runs/inference \
  --name webcam_original
```

---

## Key Flag Explanations

| Flag | Value | Meaning |
|------|-------|---------|
| `--nms` | `enhanced` | Use advanced NMS (soft or DIoU) |
| `--nms-method` | `soft` | Type of suppression (soft-NMS) |
| `--nms-method` | `diou` | Type of suppression (DIoU-NMS) |
| `--nms-soft-method` | `gaussian` | Smooth Gaussian decay for soft-NMS |
| `--nms-soft-method` | `linear` | Linear decay for soft-NMS |
| `--nms-sigma` | `0.5` | Decay parameter (only for Gaussian soft-NMS) |
| `--iou-thres` | `0.45` | Threshold for suppression (higher = less suppression) |

---

## Comparison Table

| Method | Speed | Accuracy | Best For |
|--------|-------|----------|----------|
| Original NMS | ⚡ Fast | Good | Baseline, production |
| Soft-NMS Gaussian | 🟡 Medium | ⭐ Best | Dense scenes, high accuracy |
| Soft-NMS Linear | 🟡 Medium | Good | Quick decay preference |
| DIoU-NMS | 🟡 Medium | Very Good | Close/overlapping objects |

---

## File Structure

```
YOLOv6/
├── enhance/
│   ├── NMS_AND_DIOU_GUIDE.md          ← Full documentation
│   ├── scripts/
│   │   ├── nms_enhanced.py            ← Soft-NMS + DIoU implementation
│   │   ├── adaptive_scale.py          ← Image resize utility
│   │   ├── train_with_utf8_conf.py    ← UTF-8 config reader wrapper
│   │   └── infer_with_softnms.py      ← (existing) inference example
│   └── config/
├── tools/
│   ├── infer.py                       ← Modified to support enhanced NMS
│   ├── train.py                       ← Original training script
│   └── ...
└── yolov6/
    ├── core/
    │   ├── inferer.py                 ← Modified NMS import
    │   └── ...
    └── ...
```

---

## Quick Troubleshooting

**Q: I get `ImportError` about `nms_enhanced`**
- A: Make sure `enhance/scripts/nms_enhanced.py` exists and `yolov6/core/inferer.py` is updated

**Q: Inference is very slow**
- A: Soft-NMS is slower on CPU. Try `--nms original` or use GPU (change `--device 0`)

**Q: Config file won't load (UnicodeDecodeError)**
- A: Use the wrapper: `python enhance/scripts/train_with_utf8_conf.py --conf-file configs/your.py -- [args]`

---

## Next Steps

1. **Test one command** from the options above (start with Option 1 for best results)
2. **Compare results** – watch the detection output and confidence scores
3. **Tune parameters** – adjust `--nms-sigma` (0.1–1.0) or `--iou-thres` (0.3–0.6) for your use case
4. **Deploy** – Once happy, use it in production with the same flags

