#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
Evaluate predictions.json file and display COCO metrics in standard format.
"""

import argparse
import os
import sys
import json
import tempfile
import yaml
from pathlib import Path

ROOT = os.getcwd()
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


def find_annotation_file(data_yaml_path):
    """Find the annotation JSON file from data YAML."""
    if not os.path.exists(data_yaml_path):
        raise FileNotFoundError(f"Data YAML file not found: {data_yaml_path}")
    
    with open(data_yaml_path, 'r') as f:
        data_config = yaml.safe_load(f)
    
    # Check for explicit anno_path
    if 'anno_path' in data_config and data_config['anno_path']:
        anno_path = data_config['anno_path']
        if os.path.isabs(anno_path) or os.path.exists(anno_path):
            return anno_path
        # Try relative to YAML file location
        yaml_dir = os.path.dirname(data_yaml_path)
        relative_path = os.path.join(yaml_dir, anno_path)
        if os.path.exists(relative_path):
            return relative_path
        # Try relative to project root
        if os.path.exists(anno_path):
            return anno_path
    
    # Try to construct from val path
    if 'val' in data_config:
        val_path = data_config['val']
        base_name = os.path.basename(val_path)
        
        # Try multiple possible locations
        possible_paths = []
        
        # 1. Relative to YAML file location
        yaml_dir = os.path.dirname(os.path.abspath(data_yaml_path))
        if not os.path.isabs(val_path):
            # Construct path relative to YAML
            val_full = os.path.join(yaml_dir, val_path)
            dataset_root = os.path.dirname(os.path.dirname(val_full))
            possible_paths.append(os.path.join(dataset_root, 'annotations', f'instances_{base_name}.json'))
        
        # 2. Relative to project root (YOLOv6 directory)
        project_root = os.getcwd()
        if not os.path.isabs(val_path):
            dataset_root = os.path.dirname(os.path.dirname(os.path.join(project_root, val_path)))
            possible_paths.append(os.path.join(dataset_root, 'annotations', f'instances_{base_name}.json'))
        
        # 3. If val_path is absolute
        if os.path.isabs(val_path):
            dataset_root = os.path.dirname(os.path.dirname(val_path))
            possible_paths.append(os.path.join(dataset_root, 'annotations', f'instances_{base_name}.json'))
        
        # 4. Direct check in data directory structure
        yaml_parent = os.path.dirname(yaml_dir)
        possible_paths.append(os.path.join(yaml_parent, base_name, 'annotations', f'instances_{base_name}.json'))
        
        for anno_path in possible_paths:
            if os.path.exists(anno_path):
                return anno_path
    
    raise FileNotFoundError(f"Could not find annotation file. Please check {data_yaml_path}")


def evaluate_predictions(pred_json_path, anno_json_path):
    """Evaluate predictions.json against annotations and print COCO metrics."""
    if not os.path.exists(pred_json_path):
        raise FileNotFoundError(f"Predictions file not found: {pred_json_path}")
    if not os.path.exists(anno_json_path):
        raise FileNotFoundError(f"Annotation file not found: {anno_json_path}")
    
    print(f"Loading ground truth annotations from: {anno_json_path}")
    # Load annotations
    coco_gt = COCO(anno_json_path)
    
    print(f"Loading predictions from: {pred_json_path}")
    # Read predictions
    with open(pred_json_path, 'r') as f:
        pred_data = json.load(f)
    
    # If it's already a list of detections, we need to create a proper COCO format
    if isinstance(pred_data, list):
        # Get categories from ground truth
        cats = coco_gt.loadCats(coco_gt.getCatIds())
        cat_info = [{'id': cat['id'], 'name': cat['name']} for cat in cats]
        
        # Add 'id' and 'area' fields to each prediction if missing
        annotations = []
        for idx, pred in enumerate(pred_data):
            ann = pred.copy()
            if 'id' not in ann:
                ann['id'] = idx + 1  # COCO IDs start at 1
            # Calculate area from bbox [x, y, width, height]
            if 'area' not in ann and 'bbox' in ann:
                bbox = ann['bbox']
                if len(bbox) >= 4:
                    # bbox format: [x, y, width, height]
                    area = bbox[2] * bbox[3]  # width * height
                    ann['area'] = float(area)
            # Ensure iscrowd is set (default to 0)
            if 'iscrowd' not in ann:
                ann['iscrowd'] = 0
            annotations.append(ann)
        
        # Create COCO format structure
        coco_format = {
            'info': {
                'description': 'YOLOv6 predictions',
                'version': '1.0'
            },
            'licenses': [],
            'categories': cat_info,
            'images': [],
            'annotations': annotations
        }
        
        # Write to temporary file and load
        # loadRes expects just a list of annotations, not full COCO format
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp_file:
            json.dump(annotations, tmp_file)
            tmp_path = tmp_file.name
        
        try:
            coco_dt = coco_gt.loadRes(tmp_path)
        finally:
            # Clean up temp file
            try:
                os.unlink(tmp_path)
            except:
                pass
    else:
        # Try loading directly
        coco_dt = coco_gt.loadRes(pred_json_path)
    
    # Create evaluator
    coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
    
    # Set image IDs to evaluate (use all images in ground truth)
    img_ids = sorted(coco_gt.getImgIds())
    coco_eval.params.imgIds = img_ids
    
    print("\nRunning evaluation...")
    # Evaluate
    coco_eval.evaluate()
    coco_eval.accumulate()
    
    # Print results in COCO format (this is what the user wants!)
    print("\n" + "="*80)
    coco_eval.summarize()
    print("="*80 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate predictions.json and display COCO metrics'
    )
    parser.add_argument(
        'predictions_file',
        type=str,
        help='Path to predictions.json file'
    )
    parser.add_argument(
        '--data',
        type=str,
        default='./data/coco.yaml',
        help='Dataset YAML file path (to find annotation file)'
    )
    parser.add_argument(
        '--anno',
        type=str,
        default=None,
        help='Direct path to annotation JSON file (overrides --data)'
    )
    
    args = parser.parse_args()
    
    # Find annotation file
    if args.anno:
        anno_json_path = args.anno
    else:
        try:
            anno_json_path = find_annotation_file(args.data)
            print(f"Found annotation file: {anno_json_path}")
        except Exception as e:
            print(f"Error finding annotation file: {e}")
            print("Please specify --anno with direct path to annotation JSON file")
            return
    
    # Evaluate
    try:
        evaluate_predictions(args.predictions_file, anno_json_path)
    except Exception as e:
        print(f"Error during evaluation: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

