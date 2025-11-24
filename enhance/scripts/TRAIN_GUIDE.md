# Training Guide — YOLOv6 (project-local scripts)

This short guide explains how to train your custom dataset in this repository, using the helper scripts in `enhance/scripts` and the original `tools/train.py`.

Paths in this guide assume you run commands from the repository root (`C:\Enhance\YOLOv6`) on Windows PowerShell.

## Quick overview
- `enhance/scripts/train_with_utf8_conf.py` — wrapper that reads a Python config file using UTF-8 (avoids Windows cp1252 errors) and forwards the call to `tools/train.py`. Supports `--print-conf` for quick config checks.
- `configs/adapt_diou_yolov6.py` — example config that uses DIoU in the head (already provided in the repo).
- Dataset YAML: `data/Own/data.yaml` — points to `data/Own/images/{train,valid,test}` (make sure labels are in `.../labels/`).

## Typical training flow (recommended)
1. Validate dataset structure (already provided):

```powershell
cd C:\Enhance\YOLOv6
python .\enhance\validate_dataset.py
```

Expect: each split shows image/label counts and label content checks.

2. Test reading the config file (quick, non-training):

```powershell
python train.py --conf-file configs\adapt_diou_yolov6.py --print-conf
```

This will print the config read using UTF-8. If you see the file contents, reading works.

3. Run training (recommended full command):

```powershell
python train.py --conf-file configs\adapt_diou_yolov6.py -- \
  --data-path data\Own\data.yaml \
  --check-images --check-labels \
  --batch-size 8 --img-size 640 \
  --epoch 100
```

Notes:
- The `--` separates arguments for the wrapper from the arguments forwarded to `tools/train.py`.
- `--check-images` and `--check-labels` will run quick validations and fail early if labels are invalid.
- Adjust `--epoch`, `--batch-size`, and `--img-size` to your hardware.

## Quick, minimal training (faster debug)
If you want a shorter test run:

```powershell
python train.py --conf-file configs\adapt_diou_yolov6.py -- \
  --data-path data\Own\data.yaml --epoch 10 --batch-size 8
```

## GPU vs CPU
- For serious training, use a CUDA-capable GPU. Provide `--device 0` or similar (the training script will parse it). Example (GPU):

```powershell
python train.py --conf-file configs\adapt_diou_yolov6.py -- --data-path data\Own\data.yaml --device 0 --epoch 100
```

- CPU training is slow; only use it for tiny tests.

## Checkpoints & outputs
- Trained weights and training logs are saved under `runs/train/exp*`.
- Weights appear in `runs/train/exp*/weights/` (e.g. `best_ckpt.pt`, `last_ckpt.pt`).

After training, use `tools/infer.py` with the trained `best_ckpt.pt` for inference.

## Resume training
If you need to resume from a checkpoint, pass the `--resume` or the correct `--weights` option according to `tools/train.py` usage (check the script help: `python tools/train.py -h`).

## Common troubleshooting
- Unicode / encoding error when loading a Python config:
  - Use `enhance/scripts/train.py` (the wrapper) to read and forward a UTF-8-safe config.
- Dataset errors (missing labels, invalid label lines):
  - Run `python .\enhance\validate_dataset.py` and fix issues reported.
- Low GPU memory or OOM:
  - Lower `--batch-size` or use a smaller `--img-size`.
- No detections after training:
  - Verify labels correspond to classes defined in `data/Own/data.yaml`.
  - Verify training logs (`runs/train/exp*/results.csv`) show decreasing loss and improved mAP.

## Example: full pipeline (train -> infer)
1. Validate dataset
```powershell
python .\enhance\validate_dataset.py
```
2. Train
```powershell
python train.py --conf-file configs\adapt_diou_yolov6.py -- --data-path data\Own\data.yaml --check-images --check-labels --batch-size 8 --img-size 640 --epoch 100
```
3. Inference (webcam) using trained weights (change `exp*` to the actual exp folder):
```powershell
python tools\infer.py --weights runs\train\exp15\weights\best_ckpt.pt --webcam --webcam-addr 0 --yaml data\Own\data.yaml --device cpu --conf-thres 0.25 --view-img
```

## Helpful tips
- Keep a separate small validation set for quick tests.
- If training on small datasets, more aggressive augmentation and longer epochs usually help.
- Use smaller model architectures (`yolov6n`, `yolov6s`) for faster experimentation.

---

If you want, I can also:
- Add a short PowerShell script `train_example.ps1` inside this folder to run the recommended command(s), or
- Start a short training run (e.g. 1 epoch) here to show the first logs if you want me to.

