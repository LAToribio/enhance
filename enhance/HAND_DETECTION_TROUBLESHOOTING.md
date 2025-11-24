# Hand Detection Not Working - Troubleshooting Guide

Your model is trained on **6 sign language classes** (Hello, I Love You, No, Please, Thank You, Yes) — NOT generic "hand" detection.

---

## 🔍 Step-by-Step Diagnosis

### **Step 1: Verify Your Dataset is Correct**

Check that your training data actually has annotations:

```powershell
# Check if images exist
dir c:\Enhance\YOLOv6\data\Own\images\train
dir c:\Enhance\YOLOv6\data\Own\images\valid

# Check if label files exist (must be .txt files with same names)
# Example: image.jpg should have image.txt with bounding box annotations
dir c:\Enhance\YOLOv6\data\Own\images\train\labels
dir c:\Enhance\YOLOv6\data\Own\images\valid\labels
```

**Expected format of label files:**
```
0 0.5 0.5 0.3 0.4    # class_id center_x center_y width height (normalized 0-1)
```

---

### **Step 2: Check if Model Actually Trained**

Look at training logs:

```bash
# Check training results
cat runs/train/exp15/results.csv
```

**Look for:**
- ✅ Did loss decrease over epochs?
- ✅ Did mAP increase?
- ❌ If loss stayed flat → model didn't learn (bad data or labels)

---

### **Step 3: Check Label Quality**

Common label issues:

```bash
# Verify label files exist for every image
python -c "
import os
train_imgs = set(os.path.splitext(f)[0] for f in os.listdir('data/Own/images/train') if f.endswith(('.jpg','.png')))
train_labels = set(os.path.splitext(f)[0] for f in os.listdir('data/Own/images/train/labels') if f.endswith('.txt'))
missing = train_imgs - train_labels
if missing:
    print(f'⚠️ Missing labels for: {missing}')
else:
    print('✅ All images have labels')
"
```

---

### **Step 4: Verify Class Numbers in Labels**

Classes should be 0-5 (matching your yaml):
- 0 = Hello
- 1 = I Love You
- 2 = No
- 3 = Please
- 4 = Thank You
- 5 = Yes

```bash
# Check what class IDs are in your labels
python -c "
import os
from pathlib import Path

class_ids = set()
for txt_file in Path('data/Own/images/train/labels').glob('*.txt'):
    with open(txt_file) as f:
        for line in f:
            if line.strip():
                class_id = int(line.split()[0])
                class_ids.add(class_id)

print(f'Class IDs found in labels: {sorted(class_ids)}')
print(f'Expected: {list(range(6))}')
if class_ids == set(range(6)):
    print('✅ All class IDs are correct')
else:
    missing = set(range(6)) - class_ids
    extra = class_ids - set(range(6))
    if missing:
        print(f'⚠️ Missing classes: {missing}')
    if extra:
        print(f'⚠️ Invalid class IDs: {extra}')
"
```

---

### **Step 5: Lower Confidence Threshold**

Your model might be detecting signs but with low confidence:

```bash
# Try much lower confidence threshold
python tools/infer.py \
  --weights runs/train/exp15/weights/best_ckpt.pt \
  --source data/Own/images/test \
  --yaml data/Own/data.yaml \
  --device cpu \
  --conf-thres 0.1 \
  --nms original \
  --view-img \
  --project runs/inference \
  --name test_low_conf
```

Try these thresholds: **0.05, 0.1, 0.15, 0.25**

---

### **Step 6: Check if Images are Being Processed**

Run inference with debugging:

```python
import torch
import cv2
from yolov6.core.inferer import Inferer
from yolov6.utils.events import load_yaml

device = 'cpu'
weights = r'runs/train/exp15/weights/best_ckpt.pt'
yaml_file = r'data/Own/data.yaml'

# Load model
model = torch.load(weights, map_location=device)['model']
yaml_data = load_yaml(yaml_file)

print(f"Model loaded. Classes: {yaml_data['names']}")
print(f"Number of classes: {yaml_data['nc']}")

# Test on single image
test_img = cv2.imread('data/Own/images/test/someimage.jpg')
if test_img is not None:
    print(f"Image shape: {test_img.shape}")
    # Try inference (simplified)
else:
    print("❌ Could not load test image")
```

---

## 🛠️ Common Fixes

### **Issue 1: No detections at all**

**Solution:**
```bash
# Re-train with verification steps
python tools/train.py \
  --conf-file configs/adapt_diou_yolov6.py \
  --data-path data/Own/data.yaml \
  --check-images \
  --check-labels \
  --batch-size 8 \
  --img-size 640 \
  --epoch 50
```

The `--check-images` and `--check-labels` flags will:
- ✅ Verify all images are readable
- ✅ Verify all labels exist and are valid
- ✅ Show warnings if there are issues

---

### **Issue 2: Low confidence detections**

**Solutions:**
1. Lower `--conf-thres` (try 0.1 instead of 0.25)
2. Train longer (increase `--epoch` to 100)
3. Use better quality labels
4. Increase training data

---

### **Issue 3: Wrong classes detected**

**Solution:**
- Check that your label files use class indices 0-5
- Don't use text labels ("hand") in label files — only numbers!

**Correct label format:**
```
0 0.45 0.50 0.30 0.40
3 0.70 0.60 0.25 0.35
```

**Wrong label format (don't use this):**
```
Hello 0.45 0.50 0.30 0.40
Please 0.70 0.60 0.25 0.35
```

---

## ⚡ Quick Tests

### **Test 1: Verify training actually happened**

```bash
# Look at loss curves - should show decreasing loss
cat runs/train/exp15/results.csv | head -20
```

### **Test 2: Try inference on training images**

```bash
# If model was trained, it should detect on training images
python tools/infer.py \
  --weights runs/train/exp15/weights/best_ckpt.pt \
  --source data/Own/images/train \
  --yaml data/Own/data.yaml \
  --device cpu \
  --conf-thres 0.25 \
  --view-img \
  --project runs/inference \
  --name train_inference
```

### **Test 3: Check single image**

```bash
python tools/infer.py \
  --weights runs/train/exp15/weights/best_ckpt.pt \
  --source data/Own/images/test/image001.jpg \
  --yaml data/Own/data.yaml \
  --device cpu \
  --conf-thres 0.1 \
  --view-img
```

---

## 📋 Checklist

- [ ] `data/Own/images/train/` has images
- [ ] `data/Own/images/train/labels/` has corresponding .txt files
- [ ] Each label file has format: `class_id x y w h` (5 values per line)
- [ ] Class IDs are 0-5 (matching your 6 sign language classes)
- [ ] `data.yaml` path is correct: `train: data/Own/images/train`
- [ ] Model file exists: `runs/train/exp15/weights/best_ckpt.pt`
- [ ] Training loss decreased over epochs (check results.csv)

---

## 🔧 Next Steps

1. **Run verification:**
   ```bash
   python tools/train.py \
     --conf-file configs/adapt_diou_yolov6.py \
     --data-path data/Own/data.yaml \
     --check-images \
     --check-labels
   ```

2. **If issues found:** Fix labels/images
3. **Re-train:** Use the command above with `--epoch 50`
4. **Test inference:** Lower `--conf-thres` to 0.1 and test on your images

---

## Need Help?

Provide these details:
- Screenshot of training results (loss, mAP)
- Number of images in: `data/Own/images/{train,valid,test}`
- Sample output when running inference with `--conf-thres 0.1`

