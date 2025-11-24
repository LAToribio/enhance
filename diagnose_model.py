#!/usr/bin/env python3
"""
Diagnostic script to check what your trained model knows about.
This will help us figure out why hand detection isn't working.
"""
import torch
import os
import sys

# Add repo to path
repo_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, repo_root)

def diagnose_model(weights_path, yaml_path):
    print("=" * 60)
    print("MODEL DIAGNOSTICS")
    print("=" * 60)
    print()
    
    # Load YAML to see expected classes
    from yolov6.utils.events import load_yaml
    yaml_data = load_yaml(yaml_path)
    print(f"📋 Dataset YAML expects {yaml_data['nc']} classes:")
    for i, name in enumerate(yaml_data['names']):
        print(f"   [{i}] {name}")
    print()
    
    # Load model and check what it was trained on
    device = torch.device('cpu')
    try:
        model = torch.load(weights_path, map_location=device)
        print(f"✅ Model loaded: {weights_path}")
        print()
        
        # Check if it's a checkpoint dict or model directly
        if isinstance(model, dict):
            if 'model' in model:
                model_state = model['model']
                print(f"📦 Checkpoint dict found with keys: {list(model.keys())}")
                print()
                
                # Try to infer num classes from model structure
                try:
                    # Look for output head that has class predictions
                    for key in model_state:
                        if 'cls_convs' in key or 'cls_preds' in key or 'cls_head' in key:
                            print(f"   Found class head: {key}")
                except Exception as e:
                    print(f"   (Could not parse model structure: {e})")
            else:
                print(f"📦 Keys in checkpoint: {list(model.keys())}")
        else:
            print(f"⚠️  Model is not a checkpoint dict, it's: {type(model)}")
        
        print()
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return
    
    print("=" * 60)
    print("POSSIBLE ISSUES:")
    print("=" * 60)
    print()
    print("1. ❌ Model NOT trained on hand/sign language data?")
    print("   → Did you use the right model weights?")
    print("   → Run: python tools/train.py --conf-file configs/adapt_diou_yolov6.py --data-path data/Own/data.yaml")
    print()
    print("2. ❌ Dataset images/labels missing or corrupted?")
    print("   → Check: data/Own/images/{train,valid,test}/")
    print("   → Check: corresponding *.txt labels exist")
    print()
    print("3. ❌ Class mismatch?")
    print("   → Model expects 6 classes (sign language)")
    print("   → Labels must use indices 0-5 (not hand/hand-like)")
    print()
    print("4. ❌ Model confidence too high?")
    print("   → Try lowering: --conf-thres 0.1 or 0.15")
    print()
    print("5. ❌ Wrong model file?")
    print("   → Check if best_ckpt.pt is from the right training run")
    print()
    print("=" * 60)
    print()

if __name__ == '__main__':
    weights = r'c:\Enhance\YOLOv6\runs\train\exp15\weights\best_ckpt.pt'
    yaml = r'c:\Enhance\YOLOv6\data\Own\data.yaml'
    
    if not os.path.exists(weights):
        print(f"❌ Weights not found: {weights}")
        sys.exit(1)
    
    if not os.path.exists(yaml):
        print(f"❌ YAML not found: {yaml}")
        sys.exit(1)
    
    diagnose_model(weights, yaml)
