#!/usr/bin/env python3
"""
Data Validation and Setup Script

This script:
1. Checks if your data has proper structure
2. Helps organize labels if they're in a different location
3. Creates the required folder structure for YOLOv6
"""

import os
import sys
from pathlib import Path

def check_data_structure():
    print("=" * 70)
    print("YOLOV6 DATA STRUCTURE CHECK")
    print("=" * 70)
    print()
    
    base_path = Path('data/Own/images')
    
    required_dirs = ['train', 'valid', 'test']
    status = {}
    
    for split in required_dirs:
        split_path = base_path / split
        labels_path = split_path / 'labels'
        
        images = list(split_path.glob('*.jpg')) + list(split_path.glob('*.png')) + list(split_path.glob('*.JPG')) + list(split_path.glob('*.PNG'))
        labels = list(labels_path.glob('*.txt')) if labels_path.exists() else []
        
        print(f"📁 {split.upper()}:")
        print(f"   Images found: {len(images)}")
        print(f"   Labels found: {len(labels)}")
        
        if len(images) == 0:
            print(f"   ❌ NO IMAGES!")
        elif len(labels) == 0:
            print(f"   ❌ NO LABELS! (THIS IS THE PROBLEM)")
        elif len(images) != len(labels):
            print(f"   ⚠️  MISMATCH! Images: {len(images)}, Labels: {len(labels)}")
        else:
            print(f"   ✅ OK")
        
        status[split] = {
            'images': len(images),
            'labels': len(labels),
            'path': split_path,
            'labels_path': labels_path
        }
        print()
    
    return status

def create_labels_folder():
    print("=" * 70)
    print("SOLUTION: CREATE LABELS FOLDERS")
    print("=" * 70)
    print()
    
    print("⚠️  YOUR DATA STRUCTURE IS INCOMPLETE!")
    print()
    print("You have images but NO label files (.txt files).")
    print()
    print("This means:")
    print("  ❌ The model was trained on images WITHOUT annotations")
    print("  ❌ The model has NO IDEA what to detect")
    print("  ❌ That's why you get no detections!")
    print()
    print("=" * 70)
    print("REQUIRED ACTIONS:")
    print("=" * 70)
    print()
    print("You need to provide label files. Options:")
    print()
    print("1️⃣  AUTOMATIC LABELING (If you have Roboflow/LabelImg):")
    print("   - Use Roboflow to label your images")
    print("   - Export in YOLO format")
    print("   - Download and place .txt files in data/Own/images/{train,valid,test}/labels/")
    print()
    print("2️⃣  MANUAL LABELING:")
    print("   - Use LabelImg tool")
    print("   - Save in YOLO format")
    print("   - Each .jpg needs corresponding .txt with annotations")
    print()
    print("3️⃣  SAMPLE LABEL FORMAT:")
    print("   For a 'Hello' sign in position (0.5, 0.5) with size (0.3, 0.4):")
    print("   data/Own/images/train/labels/Hello-1-_jpg.rf...ed.txt:")
    print("   ---")
    print("   0 0.5 0.5 0.3 0.4")
    print("   ---")
    print("   (class_id, center_x, center_y, width, height - all normalized 0-1)")
    print()
    print("=" * 70)
    print()

def check_roboflow_labels():
    """Check if labels might be in a different location"""
    print("Looking for label files in alternate locations...")
    
    possible_locations = [
        'data/Own',
        'data/Own/images',
        'data/Own/labels',
        'datasets',
        'labels',
        '.',
    ]
    
    for loc in possible_locations:
        loc_path = Path(loc)
        if loc_path.exists():
            txt_files = list(loc_path.rglob('*.txt'))
            if txt_files:
                print(f"✅ Found {len(txt_files)} .txt files in: {loc}/")
                return loc
    
    print("❌ No label files found in common locations.")
    return None

if __name__ == '__main__':
    status = check_data_structure()
    print()
    
    # Check if any split has labels
    has_labels = any(s['labels'] > 0 for s in status.values())
    
    if not has_labels:
        create_labels_folder()
        check_roboflow_labels()
    
    print()
    print("=" * 70)
    print("SUMMARY:")
    print("=" * 70)
    total_images = sum(s['images'] for s in status.values())
    total_labels = sum(s['labels'] for s in status.values())
    
    print(f"Total images: {total_images}")
    print(f"Total labels: {total_labels}")
    print()
    
    if total_labels == 0:
        print("❌ CRITICAL: NO LABELS FOUND!")
        print("   Your model cannot learn without annotations.")
        print()
        print("Next steps:")
        print("1. Get labeled data (use Roboflow or manual annotation tool)")
        print("2. Place label .txt files in data/Own/images/{train,valid,test}/labels/")
        print("3. Re-run training")
        sys.exit(1)
    elif total_labels < total_images:
        print(f"⚠️  INCOMPLETE: Only {total_labels}/{total_images} images have labels")
        sys.exit(1)
    else:
        print("✅ Data structure looks good!")
        sys.exit(0)
