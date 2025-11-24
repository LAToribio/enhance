#!/usr/bin/env python3
"""
Auto-organize labels into correct YOLO structure
"""

import os
import shutil
from pathlib import Path

def organize_labels():
    print("=" * 70)
    print("ORGANIZING LABELS INTO YOLO STRUCTURE")
    print("=" * 70)
    print()
    
    base_path = Path('data/Own')
    
    # Create label directories if they don't exist
    for split in ['train', 'valid', 'test']:
        labels_dir = base_path / 'images' / split / 'labels'
        labels_dir.mkdir(parents=True, exist_ok=True)
        print(f"✅ Created: {labels_dir}")
    
    print()
    
    # Find all txt files in data/Own/
    label_files = list(base_path.glob('*.txt'))
    print(f"Found {len(label_files)} label files to organize")
    print()
    
    if len(label_files) == 0:
        print("❌ No label files found in data/Own/")
        return False
    
    # Try to match labels to images
    matched = 0
    unmatched = []
    
    for label_file in label_files:
        label_name = label_file.stem  # Remove .txt
        
        # Try to find corresponding image in each split
        found = False
        for split in ['train', 'valid', 'test']:
            images_dir = base_path / 'images' / split
            
            # Try different image extensions
            for ext in ['.jpg', '.png', '.JPG', '.PNG']:
                img_path = images_dir / (label_name + ext)
                if img_path.exists():
                    # Move label to correct location
                    target_label = images_dir / 'labels' / label_file.name
                    shutil.move(str(label_file), str(target_label))
                    print(f"✅ {label_file.name} → {split}/labels/")
                    matched += 1
                    found = True
                    break
            
            if found:
                break
        
        if not found:
            unmatched.append(label_file.name)
    
    print()
    print("=" * 70)
    print(f"Matched: {matched}/{len(label_files)}")
    if unmatched:
        print(f"⚠️  Unmatched ({len(unmatched)}):")
        for f in unmatched[:10]:  # Show first 10
            print(f"   - {f}")
        if len(unmatched) > 10:
            print(f"   ... and {len(unmatched) - 10} more")
    
    print("=" * 70)
    return True

if __name__ == '__main__':
    organize_labels()
    print()
    print("Running verification...")
    os.system('python check_data_structure.py')
