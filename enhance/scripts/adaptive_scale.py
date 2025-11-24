"""
Adaptive-scale utility for YOLOv6 image preprocessing.

This script resizes images to a target long-side length while maintaining aspect ratio.
It can be used standalone or imported as a module for integration with inference pipelines.

Features:
- Resize images maintaining aspect ratio
- Support for Pillow (PIL) backend
- CLI and programmatic interfaces
- Can integrate with Inferer for on-the-fly preprocessing
"""
from typing import Optional
import argparse
import os
import sys


def adaptive_resize_pil(in_path: str, out_path: str, long_side: int = 640, quality: int = 95):
    """Resize image to target long-side length using Pillow.
    
    Args:
        in_path: Input image file path
        out_path: Output image file path
        long_side: Target long-side length in pixels (320/416/480/640/800/960/1000/1024/1280)
        quality: JPEG quality if saving as JPEG (default 95)
    
    Raises:
        RuntimeError: If Pillow is not installed or image cannot be loaded
        ValueError: If long_side is not positive or paths are invalid
    """
    VALID_SIZES = [320, 416, 480, 640, 800, 960, 1000, 1024, 1280]
    if long_side <= 0:
        raise ValueError(f'long_side must be positive, got {long_side}')
    if long_side not in VALID_SIZES:
        raise ValueError(f'long_side must be one of {VALID_SIZES}, got {long_side}')
    
    try:
        from PIL import Image
    except ImportError:
        raise RuntimeError('Pillow is not installed. Install via: pip install pillow')

    # Validate input
    if not os.path.isfile(in_path):
        raise FileNotFoundError(f'Input file not found: {in_path}')
    
    # Ensure output directory exists
    out_dir = os.path.dirname(out_path)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    # Load and resize
    img = Image.open(in_path)
    w, h = img.size
    
    # If already at target size, just copy
    if max(w, h) == long_side:
        img.save(out_path, quality=quality)
        return
    
    # Calculate scale and resize
    scale = long_side / max(w, h)
    new_w, new_h = int(w * scale), int(h * scale)
    img_resized = img.resize((new_w, new_h), Image.LANCZOS)
    
    # Save with quality setting
    save_kwargs = {'quality': quality} if out_path.lower().endswith(('.jpg', '.jpeg')) else {}
    img_resized.save(out_path, **save_kwargs)


def get_parser(add_help: bool = True):
    """Create and return argument parser for adaptive_scale CLI.
    
    Args:
        add_help: Whether to add -h/--help (default True)
    
    Returns:
        argparse.ArgumentParser configured for adaptive_scale
    """
    parser = argparse.ArgumentParser(
        description='Adaptive image scaler: resize images maintaining aspect ratio',
        add_help=add_help,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python enhance/scripts/adaptive_scale.py input.jpg output.jpg --long-side 640
  python enhance/scripts/adaptive_scale.py input.jpg output.jpg --long-side 416 --quality 90
        """
    )
    parser.add_argument('input', type=str, help='Input image file path')
    parser.add_argument('output', type=str, help='Output image file path (can be .jpg, .png, etc.)')
    parser.add_argument('--long-side', type=int, default=640, choices=[320, 416, 480, 640, 800, 960, 1000, 1024, 1280],
                        help='Target long-side length: 320/416/480/640/800/960/1000/1024/1280 (default: 640)')
    parser.add_argument('--quality', type=int, default=95,
                        help='JPEG quality 1-100 if saving as JPEG (default: 95)')
    return parser


def main(argv: Optional[list] = None):
    """Main entry point for CLI.
    
    Args:
        argv: Command-line arguments (if None, uses sys.argv[1:])
    
    Returns:
        Exit code: 0 on success, non-zero on error
    """
    parser = get_parser()
    args = parser.parse_args(argv)

    try:
        adaptive_resize_pil(args.input, args.output, args.long_side, args.quality)
        print(f'✓ Successfully resized and saved to: {args.output}')
        return 0
    except FileNotFoundError as e:
        print(f'✗ File error: {e}', file=sys.stderr)
        return 2
    except ValueError as e:
        print(f'✗ Validation error: {e}', file=sys.stderr)
        return 1
    except RuntimeError as e:
        print(f'✗ Runtime error: {e}', file=sys.stderr)
        return 3
    except Exception as e:
        print(f'✗ Unexpected error: {e}', file=sys.stderr)
        return 99


if __name__ == '__main__':
    sys.exit(main())
