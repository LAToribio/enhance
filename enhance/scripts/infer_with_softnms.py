#!/usr/bin/env python3
"""
Inference script for YOLOv6 with Soft-NMS
Cross-platform Python script
"""

import os
import sys
import subprocess

def main():
    print("=" * 50)
    print("YOLOv6 Inference with Soft-NMS")
    print("=" * 50)
    print()
    
    # Configuration - EDIT THESE VALUES
    config = {
        'weights': 'weights/yolov6n.pt',
        'source': 'data/images',
        'yaml': 'data/data_yolov6/data.yaml',
        'conf_thres': 0.35,
        'iou_thres': 0.4,
        'nms': 'enhanced',
        'nms_method': 'soft',
        'nms_soft_method': 'gaussian',
        'nms_sigma': 0.5,
        'name': 'softnms_inference',
    }
    
    print("Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    print()
    
    # Build command
    cmd = [
        sys.executable,
        'tools/infer.py',
        '--weights', config['weights'],
        '--source', config['source'],
        '--yaml', config['yaml'],
        '--conf-thres', str(config['conf_thres']),
        '--iou-thres', str(config['iou_thres']),
        '--nms', config['nms'],
        '--nms-method', config['nms_method'],
        '--nms-soft-method', config['nms_soft_method'],
        '--nms-sigma', str(config['nms_sigma']),
        '--name', config['name'],
        '--save-txt',
    ]
    
    # Change to YOLOv6 directory if not already there
    script_dir = os.path.dirname(os.path.abspath(__file__))
    yolov6_dir = os.path.dirname(script_dir)
    os.chdir(yolov6_dir)
    
    print("Starting inference with Soft-NMS...")
    print()
    
    try:
        # Run inference
        result = subprocess.run(cmd, check=True)
        
        print()
        print("=" * 50)
        print("Inference Complete!")
        print("=" * 50)
        print(f"Check results in: runs/inference/{config['name']}")
        
        return result.returncode
        
    except subprocess.CalledProcessError as e:
        print(f"\nError during inference: {e}")
        return e.returncode
    except KeyboardInterrupt:
        print("\n\nInference interrupted by user.")
        return 1

if __name__ == '__main__':
    sys.exit(main())

