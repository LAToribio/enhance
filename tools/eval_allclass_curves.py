#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
Evaluation entrypoint that generates PR/F1/P/R curves with only the aggregated
"all classes" line. The original tools/eval.py behavior (per-class lines) is
left untouched.
"""

import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.append(ROOT)
if os.path.dirname(ROOT) not in sys.path:
    sys.path.append(os.path.dirname(ROOT))

import yolov6.utils.metrics as metrics
from eval import get_args_parser, run


def plot_pr_curve_all_only(px, py, ap, save_dir='pr_curve.png', names=()):
    """Plot precision-recall curve showing only the aggregated line."""
    fig, ax = plt.subplots(1, 1, figsize=(9, 6), tight_layout=True)
    py = np.stack(py, axis=1)

    ax.plot(px, py.mean(1), linewidth=3, color='blue',
            label='all classes %.3f mAP@0.5' % ap[:, 0].mean())
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    fig.savefig(Path(save_dir), dpi=250)


def plot_mc_curve_all_only(px, py, save_dir='mc_curve.png', names=(),
                           xlabel='Confidence', ylabel='Metric'):
    """Plot metric-confidence curve showing only the aggregated line."""
    fig, ax = plt.subplots(1, 1, figsize=(9, 6), tight_layout=True)

    y = py.mean(0)
    ax.plot(px, y, linewidth=3, color='blue',
            label=f'all classes {y.max():.2f} at {px[y.argmax()]:.3f}')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    fig.savefig(Path(save_dir), dpi=250)


# Override plotting functions before running evaluation.
metrics.plot_pr_curve = plot_pr_curve_all_only
metrics.plot_mc_curve = plot_mc_curve_all_only


if __name__ == "__main__":
    args = get_args_parser()
    run(**vars(args))

