# FPS Optimization Guide

**Current Issue:** FPS: 1.6 (napakababa) – likely due to CPU inference + enhanced NMS

---

## Quick Fixes (Ranked by Impact)

### 🥇 **#1: Switch to GPU** (Biggest impact)
If you have NVIDIA GPU:
```bash
python tools/infer.py \
  --weights runs/train/exp15/weights/best_ckpt.pt \
  --webcam \
  --webcam-addr 0 \
  --yaml data/Own/data.yaml \
  --device 0 \
  --conf-thres 0.25 \
  --nms original \
  --view-img \
  --project runs/inference \
  --name webcam_gpu
```
**Expected FPS:** 10-30+ FPS (depending on GPU)

---

### 🥈 **#2: Use Original NMS instead of Enhanced** (Significant improvement)
Enhanced NMS (soft-NMS, DIoU) are slower due to per-class processing.

```bash
python tools/infer.py \
  --weights runs/train/exp15/weights/best_ckpt.pt \
  --webcam \
  --webcam-addr 0 \
  --yaml data/Own/data.yaml \
  --device cpu \
  --conf-thres 0.25 \
  --nms original \
  --view-img \
  --project runs/inference \
  --name webcam_original_nms
```
**Impact on CPU:** 2-3x faster than enhanced NMS  
**Expected FPS:** 3-5 FPS (vs 1.6 now)

---

### 🥉 **#3: Use Smaller Model** (Medium improvement)
If you're using `yolov6s.pt` or larger, try smaller variant:

```bash
# Instead of best_ckpt.pt, use a smaller model
python tools/infer.py \
  --weights weights/yolov6n.pt \
  --webcam \
  --webcam-addr 0 \
  --yaml data/Own/data.yaml \
  --device cpu \
  --conf-thres 0.25 \
  --nms original \
  --view-img \
  --project runs/inference \
  --name webcam_nano
```
**Model sizes (fastest to slowest):**
- `yolov6n.pt` (nano) – fastest
- `yolov6s.pt` (small) – faster
- `yolov6m.pt` (medium) – medium
- `yolov6l.pt` (large) – slower

---

### **#4: Lower Image Resolution** (Minor improvement)
Default is 640x640. Lower it to 416 or 320:

```bash
python tools/infer.py \
  --weights runs/train/exp15/weights/best_ckpt.pt \
  --webcam \
  --webcam-addr 0 \
  --yaml data/Own/data.yaml \
  --device cpu \
  --img-size 416 416 \
  --conf-thres 0.25 \
  --nms original \
  --view-img \
  --project runs/inference \
  --name webcam_416
```
**Expected change:** ~5-10% faster

---

### **#5: Disable View (Minor improvement)**
Rendering the output slows things down:

```bash
python tools/infer.py \
  --weights runs/train/exp15/weights/best_ckpt.pt \
  --webcam \
  --webcam-addr 0 \
  --yaml data/Own/data.yaml \
  --device cpu \
  --conf-thres 0.25 \
  --nms original \
  --not-save-img \
  --project runs/inference \
  --name webcam_noview
```
**Expected change:** ~10-20% faster (no rendering overhead)

---

## 🎯 Recommended Strategy (Best Balance)

### **For CPU (if no GPU available):**
```bash
# Best accuracy + reasonable speed
python tools/infer.py \
  --weights weights/yolov6s.pt \
  --webcam \
  --webcam-addr 0 \
  --yaml data/Own/data.yaml \
  --device cpu \
  --img-size 416 416 \
  --conf-thres 0.25 \
  --nms original \
  --view-img \
  --project runs/inference \
  --name webcam_optimized_cpu
```
**Expected FPS:** 5-8 FPS

---

### **For GPU (if you have NVIDIA):**
```bash
# High FPS + best accuracy
python tools/infer.py \
  --weights runs/train/exp15/weights/best_ckpt.pt \
  --webcam \
  --webcam-addr 0 \
  --yaml data/Own/data.yaml \
  --device 0 \
  --conf-thres 0.25 \
  --nms enhanced \
  --nms-method soft \
  --nms-soft-method gaussian \
  --nms-sigma 0.5 \
  --view-img \
  --project runs/inference \
  --name webcam_optimized_gpu
```
**Expected FPS:** 15-30 FPS

---

## Speed Comparison Table

| Setup | Model | NMS | Resolution | FPS (Est.) |
|-------|-------|-----|------------|-----------|
| CPU | yolov6n | original | 320 | 2-3 |
| CPU | yolov6n | original | 640 | 1-2 |
| CPU | yolov6s | original | 416 | 3-5 |
| CPU | yolov6s | soft-nms | 416 | 1-2 |
| GPU (RTX3060) | yolov6s | original | 640 | 20-25 |
| GPU (RTX3060) | yolov6l | soft-nms | 640 | 10-15 |

---

## Detailed Optimization Tips

### **CPU-Specific:**
1. ✅ Use original NMS (not enhanced)
2. ✅ Lower resolution (416 or 320)
3. ✅ Use nano or small model
4. ✅ Disable image rendering (`--not-save-img`)
5. ✅ Use half precision disabled (already disabled on CPU)

### **GPU-Specific:**
1. ✅ Enable half precision: `--half`
2. ✅ Use enhanced NMS (faster on GPU)
3. ✅ Keep full resolution for accuracy
4. ✅ Use larger models for better accuracy

---

## Testing Script

Create a test to measure your current FPS:

```bash
# Run for 10 seconds and measure FPS
python tools/infer.py \
  --weights runs/train/exp15/weights/best_ckpt.pt \
  --webcam \
  --webcam-addr 0 \
  --yaml data/Own/data.yaml \
  --device cpu \
  --conf-thres 0.25 \
  --nms original \
  --view-img \
  --project runs/inference \
  --name fps_test
```

Look at the FPS indicator in the top-left corner of the output window.

---

## Troubleshooting

**Q: Still getting 1-2 FPS even with original NMS**
- A: CPU inference is inherently slow. Try GPU or smaller model (yolov6n)

**Q: GPU available but still using CPU?**
- A: Make sure you have CUDA installed. Check:
  ```bash
  python -c "import torch; print(torch.cuda.is_available())"
  ```
  Should print `True`

**Q: Enhanced NMS slower than expected**
- A: That's normal on CPU. Use `--nms original` for speed. Use `--nms enhanced` only on GPU.

---

## Recommended Next Steps

1. **If you have GPU:** Use the GPU command (15-30x faster)
2. **If CPU only:** Use yolov6n or yolov6s + original NMS + 416 resolution
3. **For production:** Optimize model further (quantization, pruning)

