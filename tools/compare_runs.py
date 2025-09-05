#!/usr/bin/env python3
"""
Compare two saved runs (raw_tensors/*.npz) and report:
- Per-head numeric deltas (MAE, max abs, cosine similarity)
- Detection-level matching (IoU>0.5, class-equal) stats per image

Usage:
  python3 tools/compare_runs.py --a outputs/hef_images_test_zzzz --b hailo_model_converter/shared_with_docker/tests/data/golden_run
  python3 tools/compare_runs.py --a outputs/hef_images_match_golden --b hailo_model_converter/shared_with_docker/tests/data/golden_run

This script assumes each directory has raw_tensors/out_XXX.npz files with the
same number of images and corresponding order. It is robust to differing tensor
names as long as shapes match.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np


def list_npz(root: Path) -> List[Path]:
    xs = sorted((root / "raw_tensors").glob("out_*.npz"))
    if not xs:
        raise SystemExit(f"No NPZ files under {root}/raw_tensors")
    return xs


def group_by_shape(d: Dict[str, np.ndarray]) -> Dict[Tuple[int, ...], List[str]]:
    g: Dict[Tuple[int, ...], List[str]] = {}
    for k in d.keys():
        v = d[k]
        g.setdefault(tuple(v.shape), []).append(k)
    return g


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    va = a.reshape(-1).astype(np.float64)
    vb = b.reshape(-1).astype(np.float64)
    denom = np.linalg.norm(va) * np.linalg.norm(vb)
    return float(np.dot(va, vb) / denom) if denom > 0 else float("nan")


def iou(boxes1: np.ndarray, boxes2: np.ndarray) -> np.ndarray:
    if boxes1.size == 0 or boxes2.size == 0:
        return np.zeros((boxes1.shape[0], boxes2.shape[0]))
    x11, y11, x12, y12 = boxes1.T
    x21, y21, x22, y22 = boxes2.T
    area1 = (x12 - x11).clip(0) * (y12 - y11).clip(0)
    area2 = (x22 - x21).clip(0) * (y22 - y21).clip(0)
    out = np.zeros((boxes1.shape[0], boxes2.shape[0]))
    for i in range(boxes1.shape[0]):
        xx1 = np.maximum(x11[i], x21)
        yy1 = np.maximum(y11[i], y21)
        xx2 = np.minimum(x12[i], x22)
        yy2 = np.minimum(y12[i], y22)
        inter = (xx2 - xx1).clip(0) * (yy2 - yy1).clip(0)
        union = area1[i] + area2 - inter + 1e-9
        out[i] = inter / union
    return out


def compare_raw(a_npz: Dict[str, np.ndarray], b_npz: Dict[str, np.ndarray]) -> List[Tuple[Tuple[int, ...], float, float, float]]:
    ga = group_by_shape(a_npz)
    gb = group_by_shape(b_npz)
    shapes = sorted(set(ga.keys()) & set(gb.keys()))
    out: List[Tuple[Tuple[int, ...], float, float, float]] = []
    for shape in shapes:
        ak = sorted(ga[shape])[0]
        bk = sorted(gb[shape])[0]
        a = a_npz[ak].astype(np.float32)
        b = b_npz[bk].astype(np.float32)
        diff = b - a
        mae = float(np.mean(np.abs(diff)))
        maxae = float(np.max(np.abs(diff)))
        cos = cosine(a, b)
        out.append((shape, mae, maxae, cos))
    return out


def _filter_by_score(d: Dict[str, np.ndarray], thresh: float | None) -> Dict[str, np.ndarray]:
    if thresh is None or "boxes" not in d or "scores" not in d:
        return d
    boxes = np.array(d["boxes"]) ; scores = np.array(d["scores"]) ; classes = np.array(d.get("classes", []))
    if boxes.size == 0 or scores.size == 0:
        return d
    m = scores >= float(thresh)
    out = dict(d)
    out["boxes"] = boxes[m]
    if classes.size: out["classes"] = classes[m]
    out["scores"] = scores[m]
    return out


def match_dets(h: Dict[str, np.ndarray], g: Dict[str, np.ndarray], iou_th: float = 0.5) -> Tuple[int, int, int]:
    hb, hc = h.get("boxes", np.zeros((0, 4), np.float32)), h.get("classes", np.zeros((0,), np.int32))
    gb, gc = g.get("boxes", np.zeros((0, 4), np.float32)), g.get("classes", np.zeros((0,), np.int32))
    matched_h: set[int] = set(); matched_g: set[int] = set()
    for cid in np.unique(np.concatenate([hc, gc])):
        hi = np.where(hc == cid)[0]; gi = np.where(gc == cid)[0]
        if hi.size == 0 or gi.size == 0:
            continue
        ious = iou(hb[hi], gb[gi])
        while True:
            ii = np.unravel_index(np.argmax(ious, axis=None), ious.shape)
            if ious[ii] < iou_th:
                break
            matched_h.add(int(hi[ii[0]])); matched_g.add(int(gi[ii[1]]))
            ious[ii[0], :] = -1; ious[:, ii[1]] = -1
    return len(hb), len(gb), len(matched_g)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True, type=Path, help="First run dir (has raw_tensors)")
    ap.add_argument("--b", required=True, type=Path, help="Second run dir (has raw_tensors)")
    ap.add_argument("--iou", type=float, default=0.5, help="IoU threshold for a match")
    ap.add_argument("--score-thresh-a", type=float, default=None, help="Optional score threshold to filter A before matching")
    ap.add_argument("--score-thresh-b", type=float, default=None, help="Optional score threshold to filter B before matching")
    args = ap.parse_args()

    A = list_npz(args.a)
    B = list_npz(args.b)
    if len(A) != len(B):
        print(f"[WARN] Different counts: {len(A)} vs {len(B)}; comparing min overlap")
    n = min(len(A), len(B))

    raw_stats: Dict[Tuple[int, ...], List[Tuple[float, float, float]]] = {}
    dets: List[Tuple[int, int, int]] = []
    for i in range(n):
        a = dict(np.load(A[i]))
        b = dict(np.load(B[i]))
        a = _filter_by_score(a, args.score_thresh_a)
        b = _filter_by_score(b, args.score_thresh_b)
        for shape, mae, maxae, cos in compare_raw(a, b):
            raw_stats.setdefault(shape, []).append((mae, maxae, cos))
        dets.append(match_dets(a, b, iou_th=float(args.iou)))

    print("Raw head deltas (mean over images):")
    for shape in sorted(raw_stats.keys()):
        arr = np.array(raw_stats[shape])
        print(f"  {str(shape):>14}: MAE {arr[:,0].mean():.6f}  MaxAE {arr[:,1].mean():.6f}  Cos {arr[:,2].mean():.6f}")

    if dets:
        rec = [m / max(1, g) for h, g, m in dets]
        prec = [m / max(1, h) for h, g, m in dets]
        f1 = [0 if (r+p)==0 else 2*r*p/(r+p) for r, p in zip(rec, prec)]
        print("\nDetection matching (IoU>0.5, same class):")
        for i, (h, g, m) in enumerate(dets):
            r = m / max(1, g); p = m / max(1, h); F = 0 if (r+p)==0 else 2*r*p/(r+p)
            print(f"  img {i}: HEF {h:3d}, GOLD {g:3d}, match {m:3d} | R {r:.2f} P {p:.2f} F1 {F:.2f}")
        print(
            f"  Averages: R {np.mean(rec):.3f}  P {np.mean(prec):.3f}  F1 {np.mean(f1):.3f}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
