#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
Script to compare metrics across different image sizes with and without adaptive scaling.

This script evaluates the model at multiple image sizes to determine if adaptive scaling
is enhancing the algorithm's performance.

Usage:
    python tools/compare_adaptive_scale.py --weights <model.pt> --data <data.yaml> [options]
"""

import argparse
import os
import sys
import json
import pandas as pd
from pathlib import Path
import torch

ROOT = os.getcwd()
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from yolov6.core.evaler_enhanced import EvalerEnhanced
from yolov6.utils.events import LOGGER
from yolov6.utils.general import increment_name, check_img_size
from yolov6.utils.config import Config
from yolov6.data.data_load import create_dataloader


def boolean_string(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'


def get_args_parser(add_help=True):
    parser = argparse.ArgumentParser(
        description='Compare metrics across image sizes with/without adaptive scaling',
        add_help=add_help
    )
    parser.add_argument('--data', type=str, required=True, help='dataset.yaml path')
    parser.add_argument('--weights', type=str, required=True, help='model.pt path')
    parser.add_argument('--batch-size', type=int, default=32, help='batch size')
    parser.add_argument('--conf-thres', type=float, default=0.03, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.65, help='NMS IoU threshold')
    parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--half', default=False, action='store_true', help='use fp16 inference')
    parser.add_argument('--save-dir', type=str, default='runs/val/', help='evaluation save dir')
    parser.add_argument('--name', type=str, default='adaptive_scale_comparison', 
                       help='save evaluation results to save_dir/name')
    
    # Image sizes to test
    parser.add_argument('--img-sizes', type=int, nargs='+', 
                       default=[320, 416, 480, 640, 800, 960, 1024, 1280],
                       help='List of image sizes to test')
    
    # Adaptive scale settings
    parser.add_argument('--adaptive-long-side', type=int, default=640,
                       choices=[320, 416, 480, 640, 800, 960, 1000, 1024, 1280],
                       help='Target long-side for adaptive scaling')
    
    # NMS settings
    parser.add_argument('--nms', type=str, default='original', choices=['original', 'enhanced'],
                       help='NMS mode: original or enhanced (default: original)')
    parser.add_argument('--nms-method', type=str, default='soft', choices=['soft', 'diou', 'nms'],
                       help='Enhanced NMS method')
    parser.add_argument('--nms-soft-method', type=str, default='gaussian', choices=['gaussian', 'linear'],
                       help='Soft-NMS method')
    parser.add_argument('--nms-sigma', type=float, default=0.5, help='Sigma for Gaussian Soft-NMS')
    parser.add_argument('--max-det', type=int, default=300, help='Maximum detections per image')
    
    # Output options
    parser.add_argument('--output-format', type=str, default='both', choices=['csv', 'json', 'both'],
                       help='Output format for results')
    parser.add_argument('--verbose', default=False, action='store_true', help='Print detailed results')
    
    return parser


def run_evaluation(data, weights, img_size, adaptive_scale, adaptive_long_side, 
                   batch_size, conf_thres, iou_thres, device, half, save_dir, name,
                   nms_mode, nms_method, nms_soft_method, nms_sigma, max_det):
    """Run a single evaluation with specified parameters."""
    
    # Create unique save directory for this configuration
    config_name = f"img{img_size}_adaptive{adaptive_scale}_long{adaptive_long_side}"
    eval_save_dir = os.path.join(save_dir, name, config_name)
    os.makedirs(eval_save_dir, exist_ok=True)
    
    # Verify img_size is valid
    img_size = check_img_size(img_size, 32, floor=256)
    
    # Initialize evaluator with adaptive_scale settings
    val = EvalerEnhanced(
        data, batch_size, img_size, conf_thres, iou_thres, device, half, eval_save_dir,
        shrink_size=0, infer_on_rect=False, verbose=False, do_coco_metric=True,
        do_pr_metric=True, plot_curve=False, plot_confusion_matrix=False,
        specific_shape=False, height=640, width=640,
        nms_mode=nms_mode, nms_method=nms_method, nms_soft_method=nms_soft_method,
        nms_sigma=nms_sigma, max_det=max_det,
        adaptive_scale=adaptive_scale, adaptive_long_side=adaptive_long_side, adaptive_quality=95
    )
    
    # Load model
    model = val.init_model(None, weights, 'val')
    
    # Initialize dataloader (adaptive_scale is now handled by evaluator)
    dataloader = val.init_data(None, 'val')
    
    # Run evaluation
    task = 'val'  # Define task for evaluation
    model.eval()
    pred_result, vis_outputs, vis_paths = val.predict_model(model, dataloader, task)
    eval_result = val.eval_model(pred_result, model, dataloader, task)
    
    # Get speed metrics
    speed_result = val.speed_result
    n_samples = speed_result[0].item()
    pre_time = 1000 * speed_result[1].cpu().numpy() / n_samples if n_samples > 0 else 0
    inf_time = 1000 * speed_result[2].cpu().numpy() / n_samples if n_samples > 0 else 0
    nms_time = 1000 * speed_result[3].cpu().numpy() / n_samples if n_samples > 0 else 0
    
    map50, map = eval_result
    
    # Get PR metrics if available
    pr_metrics = {}
    if hasattr(val, 'pr_metric_result'):
        pr_map50, pr_map = val.pr_metric_result
        pr_metrics = {
            'pr_map50': pr_map50,
            'pr_map': pr_map
        }
    
    results = {
        'img_size': img_size,
        'adaptive_scale': adaptive_scale,
        'adaptive_long_side': adaptive_long_side if adaptive_scale else None,
        'map50': map50,
        'map': map,
        'preprocess_time_ms': float(pre_time),
        'inference_time_ms': float(inf_time),
        'nms_time_ms': float(nms_time),
        'total_time_ms': float(pre_time + inf_time + nms_time),
        **pr_metrics
    }
    
    return results


def main():
    args = get_args_parser().parse_args()
    
    # Load dataset config
    if isinstance(args.data, str):
        import yaml
        with open(args.data, 'r') as f:
            data = yaml.safe_load(f)
    else:
        data = args.data
    
    # Setup device
    device = EvalerEnhanced.reload_device(args.device, None, 'val')
    half = device.type != 'cpu' and args.half
    
    LOGGER.info("=" * 80)
    LOGGER.info("Adaptive Scale Comparison Study")
    LOGGER.info("=" * 80)
    LOGGER.info(f"Model: {args.weights}")
    LOGGER.info(f"Dataset: {args.data}")
    LOGGER.info(f"Image sizes to test: {args.img_sizes}")
    LOGGER.info(f"Adaptive long-side: {args.adaptive_long_side}")
    LOGGER.info(f"NMS mode: {args.nms}")
    LOGGER.info("=" * 80)
    
    all_results = []
    
    # First: Run ONE evaluation WITH adaptive scaling (handles all sizes automatically)
    LOGGER.info(f"\n{'='*80}")
    LOGGER.info("Running evaluation WITH adaptive scaling")
    LOGGER.info(f"(adaptive_long_side={args.adaptive_long_side} - handles all image sizes)")
    LOGGER.info(f"{'='*80}")
    
    result_adaptive = None
    try:
        # Use the adaptive_long_side as the img_size for adaptive scaling
        result_adaptive = run_evaluation(
            data, args.weights, args.adaptive_long_side, True, args.adaptive_long_side,
            args.batch_size, args.conf_thres, args.iou_thres, device, half,
            args.save_dir, args.name, args.nms, args.nms_method,
            args.nms_soft_method, args.nms_sigma, args.max_det
        )
        result_adaptive['config'] = 'with_adaptive'
        result_adaptive['img_size'] = 'adaptive'  # Mark as adaptive (handles all sizes)
        all_results.append(result_adaptive)
        LOGGER.info(f"  mAP@0.5: {result_adaptive['map50']:.4f}, "
                   f"mAP@0.5:0.95: {result_adaptive['map']:.4f}")
    except Exception as e:
        import traceback
        LOGGER.error(f"  Error in adaptive scaling evaluation: {str(e)}")
        if args.verbose:
            LOGGER.error(f"  Traceback: {traceback.format_exc()}")
    
    # Then: Test each image size WITHOUT adaptive scaling
    LOGGER.info(f"\n{'='*80}")
    LOGGER.info("Running evaluations WITHOUT adaptive scaling (fixed sizes)")
    LOGGER.info(f"{'='*80}")
    
    for img_size in args.img_sizes:
        LOGGER.info(f"\nTesting fixed image size: {img_size}")
        try:
            result_no_adaptive = run_evaluation(
                data, args.weights, img_size, False, args.adaptive_long_side,
                args.batch_size, args.conf_thres, args.iou_thres, device, half,
                args.save_dir, args.name, args.nms, args.nms_method,
                args.nms_soft_method, args.nms_sigma, args.max_det
            )
            result_no_adaptive['config'] = 'no_adaptive'
            all_results.append(result_no_adaptive)
            LOGGER.info(f"  mAP@0.5: {result_no_adaptive['map50']:.4f}, "
                       f"mAP@0.5:0.95: {result_no_adaptive['map']:.4f}")
            
            # Compare with adaptive scaling if available
            if result_adaptive is not None:
                map50_improvement = result_adaptive['map50'] - result_no_adaptive['map50']
                map_improvement = result_adaptive['map'] - result_no_adaptive['map']
                map50_pct = (map50_improvement / result_no_adaptive['map50'] * 100) if result_no_adaptive['map50'] > 0 else 0
                map_pct = (map_improvement / result_no_adaptive['map'] * 100) if result_no_adaptive['map'] > 0 else 0
                
                LOGGER.info(f"  vs Adaptive: mAP@0.5 {map50_improvement:+.4f} ({map50_pct:+.2f}%), "
                           f"mAP@0.5:0.95 {map_improvement:+.4f} ({map_pct:+.2f}%)")
        except Exception as e:
            import traceback
            LOGGER.error(f"  Error: {str(e)}")
            if args.verbose:
                LOGGER.error(f"  Traceback: {traceback.format_exc()}")
            continue
    
    # Save results
    output_dir = os.path.join(args.save_dir, args.name)
    os.makedirs(output_dir, exist_ok=True)
    
    # Convert to DataFrame for easier analysis
    df = pd.DataFrame(all_results)
    
    # Save CSV
    if args.output_format in ['csv', 'both']:
        csv_path = os.path.join(output_dir, 'results.csv')
        df.to_csv(csv_path, index=False)
        LOGGER.info(f"\nResults saved to: {csv_path}")
    
    # Save JSON
    if args.output_format in ['json', 'both']:
        json_path = os.path.join(output_dir, 'results.json')
        with open(json_path, 'w') as f:
            json.dump(all_results, f, indent=2)
        LOGGER.info(f"Results saved to: {json_path}")
    
    # Print summary
    LOGGER.info("\n" + "=" * 80)
    LOGGER.info("SUMMARY")
    LOGGER.info("=" * 80)
    
    if len(df) > 0:
        # Get the adaptive scaling result (single evaluation)
        adaptive_data = df[df['config'] == 'with_adaptive']
        adaptive_result = adaptive_data.iloc[0] if len(adaptive_data) > 0 else None
        
        # Compare adaptive against each fixed size
        summary_data = []
        fixed_size_data = df[df['config'] == 'no_adaptive']
        
        if adaptive_result is not None and len(fixed_size_data) > 0:
            for _, fixed_row in fixed_size_data.iterrows():
                img_size = fixed_row['img_size']
                map50_improvement = adaptive_result['map50'] - fixed_row['map50']
                map_improvement = adaptive_result['map'] - fixed_row['map']
                
                summary_data.append({
                    'img_size': img_size,
                    'fixed_size_map50': fixed_row['map50'],
                    'adaptive_map50': adaptive_result['map50'],
                    'map50_improvement': map50_improvement,
                    'map50_improvement_pct': (map50_improvement / fixed_row['map50'] * 100) if fixed_row['map50'] > 0 else 0,
                    'fixed_size_map': fixed_row['map'],
                    'adaptive_map': adaptive_result['map'],
                    'map_improvement': map_improvement,
                    'map_improvement_pct': (map_improvement / fixed_row['map'] * 100) if fixed_row['map'] > 0 else 0,
                })
        
        summary_df = pd.DataFrame(summary_data)
        
        if len(summary_df) > 0:
            LOGGER.info("\nComparison Summary:")
            LOGGER.info("(Adaptive scaling result compared against each fixed image size)")
            LOGGER.info(summary_df.to_string(index=False))
            
            # Save summary
            summary_path = os.path.join(output_dir, 'summary.csv')
            summary_df.to_csv(summary_path, index=False)
            LOGGER.info(f"\nSummary saved to: {summary_path}")
            
            # Overall statistics
            avg_map50_improvement = summary_df['map50_improvement'].mean()
            avg_map_improvement = summary_df['map_improvement'].mean()
            positive_improvements = (summary_df['map50_improvement'] > 0).sum()
            total_tests = len(summary_df)
            
            LOGGER.info("\n" + "=" * 80)
            LOGGER.info("OVERALL STATISTICS")
            LOGGER.info("=" * 80)
            LOGGER.info(f"Average mAP@0.5 improvement: {avg_map50_improvement:+.4f}")
            LOGGER.info(f"Average mAP@0.5:0.95 improvement: {avg_map_improvement:+.4f}")
            LOGGER.info(f"Image sizes with positive improvement: {positive_improvements}/{total_tests}")
            
            if avg_map50_improvement > 0:
                LOGGER.info("\n✅ CONCLUSION: Adaptive scaling is ENHANCING the algorithm!")
            elif avg_map50_improvement < 0:
                LOGGER.info("\n❌ CONCLUSION: Adaptive scaling is NOT enhancing the algorithm.")
            else:
                LOGGER.info("\n➖ CONCLUSION: Adaptive scaling has NEUTRAL effect.")
    
    LOGGER.info("\n" + "=" * 80)
    LOGGER.info("Comparison complete!")
    LOGGER.info("=" * 80)


if __name__ == "__main__":
    main()

