"""
nms_enhanced.py
Enhanced NMS utilities for YOLOv6 with advanced suppression algorithms.

Features:
- DIoU computation and DIoU-based suppression
- Soft-NMS (linear and gaussian)
- A single API `enhanced_nms` that mirrors basic behavior of original non_max_suppression

This module is intentionally self-contained and importable from inference scripts.
"""
from typing import List, Optional
import math
import numpy as np
import torch


def xyxy_to(xyxy):
    # ensure float
    return xyxy.astype(np.float64) if isinstance(xyxy, np.ndarray) else xyxy


def box_iou_numpy(box1: np.ndarray, box2: np.ndarray):
    """Compute IoU between two boxes in xyxy format (x1,y1,x2,y2).
    box1: (4,) or (N,4)
    box2: (4,) or (M,4)
    returns IoU matrix (N,M)
    """
    box1 = np.atleast_2d(box1)
    box2 = np.atleast_2d(box2)
    # intersection
    inter_x1 = np.maximum(box1[:, None, 0], box2[None, :, 0])
    inter_y1 = np.maximum(box1[:, None, 1], box2[None, :, 1])
    inter_x2 = np.minimum(box1[:, None, 2], box2[None, :, 2])
    inter_y2 = np.minimum(box1[:, None, 3], box2[None, :, 3])
    inter_w = np.maximum(0.0, inter_x2 - inter_x1)
    inter_h = np.maximum(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area1 = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])
    area2 = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])
    union = area1[:, None] + area2[None, :] - inter_area
    iou = inter_area / (union + 1e-12)
    return iou


def diou_numpy(box1: np.ndarray, box2: np.ndarray):
    """Compute DIoU between box1 and box2 (both xyxy). Returns DIoU matrix (N,M).
    DIoU = IoU - (distance_between_centers^2 / c^2)
    where c is diagonal length of smallest enclosing box.
    """
    box1 = np.atleast_2d(box1)
    box2 = np.atleast_2d(box2)
    iou = box_iou_numpy(box1, box2)

    # centers
    cx1 = (box1[:, 0] + box1[:, 2]) / 2.0
    cy1 = (box1[:, 1] + box1[:, 3]) / 2.0
    cx2 = (box2[:, 0] + box2[:, 2]) / 2.0
    cy2 = (box2[:, 1] + box2[:, 3]) / 2.0
    # pairwise squared distance
    dist2 = (cx1[:, None] - cx2[None, :]) ** 2 + (cy1[:, None] - cy2[None, :]) ** 2

    # enclosing box diagonal squared
    enclose_x1 = np.minimum(box1[:, None, 0], box2[None, :, 0])
    enclose_y1 = np.minimum(box1[:, None, 1], box2[None, :, 1])
    enclose_x2 = np.maximum(box1[:, None, 2], box2[None, :, 2])
    enclose_y2 = np.maximum(box1[:, None, 3], box2[None, :, 3])
    c2 = (enclose_x2 - enclose_x1) ** 2 + (enclose_y2 - enclose_y1) ** 2 + 1e-12

    diou = iou - dist2 / c2
    return diou


def soft_nms_numpy(boxes: np.ndarray, scores: np.ndarray, method: str = 'gaussian', iou_thresh: float = 0.5, sigma: float = 0.5, score_thresh: float = 0.001):
    """Soft-NMS (numpy implementation).
    boxes: (N,4) xyxy
    scores: (N,)
    method: 'linear' or 'gaussian'
    returns: indices of kept boxes in original order (sorted by final score desc)
    """
    assert boxes.shape[0] == scores.shape[0]
    boxes = boxes.copy().astype(np.float64)
    scores = scores.copy().astype(np.float64)
    N = boxes.shape[0]
    idxs = np.arange(N)
    keep = []

    while idxs.size > 0:
        # pick top score
        i = np.argmax(scores[idxs])
        cur = idxs[i]
        keep.append(cur)
        # remove current index from idxs
        idxs = np.delete(idxs, i)
        if idxs.size == 0:
            break
        # compute IoU of cur with the rest
        ious = box_iou_numpy(boxes[cur:cur+1], boxes[idxs]).ravel()
        # update scores
        if method == 'linear':
            for j, iou in enumerate(ious):
                if iou > iou_thresh:
                    scores[idxs[j]] = scores[idxs[j]] * (1 - iou)
        elif method == 'gaussian':
            scores[idxs] = scores[idxs] * np.exp(-(ious ** 2) / sigma)
        else:
            # fallback to hard suppression
            for j, iou in enumerate(ious):
                if iou > iou_thresh:
                    scores[idxs[j]] = 0.0
        # filter out low score boxes
        keep_mask = scores[idxs] >= score_thresh
        idxs = idxs[keep_mask]

    # sort keep by original scores desc
    keep = sorted(keep, key=lambda x: -scores[x])
    return keep


def diou_nms_numpy(boxes: np.ndarray, scores: np.ndarray, iou_thresh: float = 0.5, diou_thresh: Optional[float] = None, max_det: int = 300):
    """DIoU-based suppression: keep boxes by decreasing score, suppress boxes whose DIoU > iou_thresh.
    This uses DIoU similarity measure and suppresses more aggressively when DIoU high.
    """
    if diou_thresh is None:
        diou_thresh = iou_thresh
    order = np.argsort(-scores)
    keep = []
    while order.size > 0 and len(keep) < max_det:
        i = order[0]
        keep.append(i)
        if order.size == 1:
            break
        rest = order[1:]
        dious = diou_numpy(boxes[i:i+1], boxes[rest]).ravel()
        # suppress where diou > diou_thresh
        inds = np.where(dious <= diou_thresh)[0]
        order = rest[inds]
    return keep


def enhanced_nms(prediction: torch.Tensor, conf_thres: float = 0.25, iou_thres: float = 0.45, method: str = 'soft', soft_method: str = 'gaussian', sigma: float = 0.5, max_det: int = 300):
    """A convenience wrapper to perform nms per-image like the original non_max_suppression.
    prediction: tensor like [N, 5+num_classes]
    Returns: list of detections per image (tensor Nx6 xyxy, conf, cls)
    method: 'soft', 'diou', or 'nms' (default soft)
    """
    device = prediction.device
    output = [torch.zeros((0, 6), device=device)] * prediction.shape[0]
    for img_idx, x in enumerate(prediction):
        # filter by objectness
        if x.numel() == 0:
            continue
        # compute per-class conf
        box_xywh = x[:, :4].cpu().numpy()
        box_xyxy = np.zeros_like(box_xywh)
        # convert xywh to xyxy
        box_xyxy[:, 0] = box_xywh[:, 0] - box_xywh[:, 2] / 2
        box_xyxy[:, 1] = box_xywh[:, 1] - box_xywh[:, 3] / 2
        box_xyxy[:, 2] = box_xywh[:, 0] + box_xywh[:, 2] / 2
        box_xyxy[:, 3] = box_xywh[:, 1] + box_xywh[:, 3] / 2

        scores_all = (x[:, 4:5] * x[:, 5:]).cpu().numpy()  # (N, num_classes)
        # keep candidates > conf_thres
        candidate_mask = scores_all.max(axis=1) > conf_thres
        if not candidate_mask.any():
            continue
        boxes = box_xyxy[candidate_mask]
        scores = scores_all[candidate_mask]
        class_idx = np.argmax(scores, axis=1)
        scores = scores[np.arange(scores.shape[0]), class_idx]

        # group by class (do per-class suppression)
        keep_indices = []
        for c in np.unique(class_idx):
            cls_mask = np.where(class_idx == c)[0]
            cls_boxes = boxes[cls_mask]
            cls_scores = scores[cls_mask]
            try:
                if method == 'soft':
                    k = soft_nms_numpy(cls_boxes, cls_scores, method=soft_method, iou_thresh=iou_thres, sigma=sigma)
                elif method == 'diou':
                    k = diou_nms_numpy(cls_boxes, cls_scores, iou_thresh=iou_thres, max_det=max_det)
                else:
                    # fallback to torchvision NMS by converting to torch
                    try:
                        import torchvision
                        tb = torch.tensor(cls_boxes, dtype=torch.float32)
                        ts = torch.tensor(cls_scores, dtype=torch.float32)
                        k = torchvision.ops.nms(tb, ts, iou_thres).cpu().numpy().tolist()
                    except ImportError:
                        raise ImportError("torchvision is required for standard NMS method. "
                                        "Install with: pip install torchvision")
                # map back to original indices
                keep_indices.extend(cls_mask[k].tolist())
            except Exception as e:
                # Log error but continue with other classes
                print(f"Warning: NMS failed for class {c}: {str(e)}. Skipping class.")
                continue

        if len(keep_indices) == 0:
            continue
        kept = np.array(keep_indices)
        # build output tensor (xyxy, conf, cls)
        out = torch.zeros((kept.shape[0], 6), device=device)
        out[:, :4] = torch.from_numpy(boxes[kept]).to(device)
        out[:, 4] = torch.from_numpy(scores[kept]).to(device)
        out[:, 5] = torch.from_numpy(class_idx[kept]).to(device)
        # sort by score desc
        out = out[out[:, 4].argsort(descending=True)]
        if out.shape[0] > max_det:
            out = out[:max_det]
        output[img_idx] = out
    return output
