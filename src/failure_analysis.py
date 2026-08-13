# coding=utf-8
"""
Failure analysis для детектора: зіставляє предикти з GT (TP/FP/FN/TN),
виводить наближені per-frame атрибути (LR/FM/SV) з gt_rect і рахує
precision/recall/F1/mean IoU у розрізі офіційних атрибутів Anti-UAV
(label_new/*.json).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

ATTRIBUTE_CODES = ["FM", "LI", "LR", "OC", "OV", "SV", "TC"]

LR_AREA_PX = 400  
FM_SHIFT_PX = 60  
SV_RATIO_LOW, SV_RATIO_HIGH = 0.66, 1.5 


def iou_xywh(box_a, box_b) -> float:
    """IoU між двома боксами [x, y, w, h] (top-left кут + розмір)."""
    ax, ay, aw, ah = box_a
    bx, by, bw, bh = box_b
    ax2, ay2 = ax + aw, ay + ah
    bx2, by2 = bx + bw, by + bh
    ix1, iy1 = max(ax, bx), max(ay, by)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def load_sequence_attributes(label_new_path: Path) -> dict[str, list[str]]:
    with open(label_new_path, encoding="utf-8") as f:
        return json.load(f)


def seq_id_from_filename(filename: str) -> str:
    stem = Path(filename).stem
    for marker in ("_infrared_", "_visible_"):
        if marker in stem:
            return stem.split(marker)[0]
    raise ValueError(f"cannot parse sequence id from filename: {filename}")


def frame_index_from_filename(filename: str) -> int:
    stem = Path(filename).stem
    return int(stem.rsplit("_", 1)[-1])


def load_yolo_gt_box(label_path: Path, img_w: int, img_h: int) -> list[float] | None:
    if not label_path.exists():
        return None
    text = label_path.read_text().strip()
    if not text:
        return None
    _, xc, yc, w, h = (float(v) for v in text.split())
    pw, ph = w * img_w, h * img_h
    px, py = xc * img_w - pw / 2, yc * img_h - ph / 2
    return [px, py, pw, ph]


def pred_boxes_from_result(result) -> list[tuple[list[float], float]]:
    """Витягує (box_xywh_pixels, confidence) з Ultralytics Results."""
    if result.boxes is None or len(result.boxes) == 0:
        return []
    boxes = []
    for xyxy, conf in zip(result.boxes.xyxy.tolist(), result.boxes.conf.tolist()):
        x1, y1, x2, y2 = xyxy
        boxes.append(([x1, y1, x2 - x1, y2 - y1], conf))
    return boxes


@dataclass
class FrameResult:
    outcome: str  # "TP" | "FP" | "FN" | "TN"
    iou: float


def match_frame(gt_box_xywh, pred_boxes, iou_thresh: float = 0.5) -> FrameResult:
    """Однооб'єктний/одноклассовий датасет -> береться лише найвпевненіший предикт.
    Низький IoU при наявному GT рахується як FN (пропуск), а не окремо FP+FN."""
    best_pred = max(pred_boxes, key=lambda p: p[1]) if pred_boxes else None

    if gt_box_xywh is None:
        return FrameResult("TN", 0.0) if best_pred is None else FrameResult("FP", 0.0)

    if best_pred is None:
        return FrameResult("FN", 0.0)

    iou = iou_xywh(gt_box_xywh, best_pred[0])
    return FrameResult("TP", iou) if iou >= iou_thresh else FrameResult("FN", iou)


def derive_lr(gt_box_xywh) -> bool:
    """Low Resolution проксі: площа GT-боксу < 400 px^2 (означення зі статті)."""
    _, _, w, h = gt_box_xywh
    return (w * h) < LR_AREA_PX


def derive_fm_sv(prev_gt_box_xywh, curr_gt_box_xywh) -> tuple[bool, bool]:
    if prev_gt_box_xywh is None or curr_gt_box_xywh is None:
        return False, False

    px, py, pw, ph = prev_gt_box_xywh
    cx, cy, cw, ch = curr_gt_box_xywh
    prev_center = (px + pw / 2, py + ph / 2)
    curr_center = (cx + cw / 2, cy + ch / 2)
    shift = ((curr_center[0] - prev_center[0]) ** 2 + (curr_center[1] - prev_center[1]) ** 2) ** 0.5
    fm = shift > FM_SHIFT_PX

    prev_area, curr_area = pw * ph, cw * ch
    ratio = curr_area / prev_area if prev_area > 0 else float("inf")
    sv = not (SV_RATIO_LOW <= ratio <= SV_RATIO_HIGH)
    return fm, sv


def state_accuracy(gt_exist: list[int], gt_rect: list[list[float]],
                    frame_preds: list[list[tuple[list[float], float]]]) -> float:
    T = len(gt_exist)
    T_star = sum(1 for v in gt_exist if v > 0)
    total = 0.0
    penalty_sum = 0.0
    for t in range(T):
        visible = gt_exist[t] > 0
        preds = frame_preds[t]
        has_pred = len(preds) > 0
        p_t = 0 if has_pred else 1
        if visible:
            iou_t = max((iou_xywh(gt_rect[t], box) for box, _ in preds), default=0.0)
            total += iou_t
            penalty_sum += p_t
        else:
            total += p_t
    penalty = 0.2 * ((penalty_sum / T_star) ** 0.3) if T_star else 0.0
    return total / T - penalty


def _safe_div(n: float, d: float) -> float:
    return n / d if d else float("nan")


def _frame_metrics(df: pd.DataFrame) -> dict:
    tp = int((df["outcome"] == "TP").sum())
    fp = int((df["outcome"] == "FP").sum())
    fn = int((df["outcome"] == "FN").sum())
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    f1 = _safe_div(2 * precision * recall, precision + recall) if precision == precision and recall == recall else float("nan")
    mean_iou = df.loc[df["outcome"].isin(["TP", "FN"]), "iou"].mean()
    return {"n_frames": len(df), "precision": precision, "recall": recall, "f1": f1, "mean_iou": mean_iou}


def attribute_metrics_table(records: pd.DataFrame, seq_attributes: dict[str, list[str]]) -> pd.DataFrame:
    rows = {"overall": _frame_metrics(records)}
    for code in ATTRIBUTE_CODES:
        seqs_with_tag = {seq for seq, tags in seq_attributes.items() if code in tags}
        subset = records[records["seq_id"].isin(seqs_with_tag)]
        if len(subset) == 0:
            continue
        rows[code] = _frame_metrics(subset)

    return pd.DataFrame(rows).T


def self_test() -> None:
    assert iou_xywh([0, 0, 10, 10], [0, 0, 10, 10]) == 1.0
    assert iou_xywh([0, 0, 10, 10], [20, 20, 10, 10]) == 0.0

    assert match_frame(None, []).outcome == "TN"
    assert match_frame(None, [([0, 0, 5, 5], 0.9)]).outcome == "FP"
    assert match_frame([0, 0, 10, 10], []).outcome == "FN"
    assert match_frame([0, 0, 10, 10], [([0, 0, 10, 10], 0.9)]).outcome == "TP"

    assert seq_id_from_filename("20190925_111757_1_10_infrared_000000.jpg") == "20190925_111757_1_10"
    assert frame_index_from_filename("20190925_111757_1_10_infrared_000045.jpg") == 45

    assert derive_lr([0, 0, 10, 10]) is True 
    assert derive_lr([0, 0, 30, 30]) is False 
    fm, sv = derive_fm_sv([0, 0, 10, 10], [100, 100, 10, 10])
    assert fm is True
    fm2, sv2 = derive_fm_sv([0, 0, 10, 10], [0, 0, 10, 10])
    assert fm2 is False and sv2 is False

    perfect = pd.DataFrame([
        {"seq_id": "s1", "outcome": "TP", "iou": 1.0},
        {"seq_id": "s1", "outcome": "TP", "iou": 1.0},
        {"seq_id": "s1", "outcome": "TN", "iou": 0.0},
    ])
    m = attribute_metrics_table(perfect, {"s1": []})
    assert m.loc["overall", "precision"] == 1.0
    assert m.loc["overall", "recall"] == 1.0
    assert m.loc["overall", "mean_iou"] == 1.0

    all_misses = pd.DataFrame([
        {"seq_id": "s2", "outcome": "FN", "iou": 0.0},
        {"seq_id": "s2", "outcome": "FN", "iou": 0.0},
    ])
    m2 = attribute_metrics_table(all_misses, {"s2": []})
    assert m2.loc["overall", "recall"] == 0.0
    assert m2.loc["overall", "mean_iou"] == 0.0

    perfect_sa = state_accuracy(
        gt_exist=[1, 1, 0],
        gt_rect=[[0, 0, 10, 10], [0, 0, 10, 10], [0, 0, 0, 0]],
        frame_preds=[[([0, 0, 10, 10], 0.9)], [([0, 0, 10, 10], 0.9)], []],
    )
    assert abs(perfect_sa - 1.0) < 1e-9

    always_empty_correct = state_accuracy(
        gt_exist=[0, 0], gt_rect=[[0, 0, 0, 0], [0, 0, 0, 0]], frame_preds=[[], []],
    )
    assert abs(always_empty_correct - 1.0) < 1e-9

    all_misses_sa = state_accuracy(
        gt_exist=[1, 1], gt_rect=[[0, 0, 10, 10], [0, 0, 10, 10]], frame_preds=[[], []],
    )
    assert abs(all_misses_sa - (-0.2)) < 1e-9

    print("self_test: OK")


if __name__ == "__main__":
    self_test()
