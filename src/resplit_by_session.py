# coding=utf-8
"""
Перебудовує train/val спліт yolo_dataset по СЕСІЯХ, а не по кліпах.

Стара версія (src/dataset.py) наслідувала фізичний поділ тек train/ і val/,
де кліпи однієї сесії запису (спільний фон/дрон/траєкторія) розкидані по
обох теках -> витік даних (26 з 27 val-сесій присутні в train, див. EDA).

Цей скрипт бере ВСІ послідовності з train/ і val/ разом, групує їх за
базовою сесією (YYYYMMDD_HHMMSS_N), і переприсвоює сесії цілком одному
спліту так, щоб жодна сесія не потрапляла і в train, і в val.
"""

from __future__ import annotations

import argparse
import random
import re
from pathlib import Path

from dataset import build_split, write_data_yaml

SESSION_RE = re.compile(r"(\d{8}_\d{6}_\d+)_\d+$")


def collect_sessions(root: Path) -> dict[str, list[Path]]:
    sessions: dict[str, list[Path]] = {}
    for split_dir in ("train", "val"):
        for seq_dir in sorted((root / split_dir).iterdir()):
            if not seq_dir.is_dir():
                continue
            m = SESSION_RE.match(seq_dir.name)
            if not m:
                continue
            sessions.setdefault(m.group(1), []).append(seq_dir)
    return sessions


def split_sessions(sessions: dict[str, list[Path]], val_frac: float, seed: int):
    names = sorted(sessions)
    rng = random.Random(seed)
    rng.shuffle(names)

    total_seqs = sum(len(v) for v in sessions.values())
    target_val = total_seqs * val_frac

    val_names, running = [], 0
    for name in names:
        if running >= target_val:
            break
        val_names.append(name)
        running += len(sessions[name])
    val_names = set(val_names)

    train_seqs = [d for n, ds in sessions.items() if n not in val_names for d in ds]
    val_seqs = [d for n, ds in sessions.items() if n in val_names for d in ds]
    return train_seqs, val_seqs, sorted(val_names)


def build_from_seq_list(seq_dirs: list[Path], images_dir: Path, labels_dir: Path,
                         modality: str = "infrared") -> tuple[int, int]:
    """Як dataset.build_split, але для довільного списку теки-послідовностей."""
    import json
    import shutil

    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    n_images = n_boxes = 0
    for seq_dir in seq_dirs:
        mod_dir = seq_dir / modality
        manifest_path = mod_dir / "_sampled.json"
        if not manifest_path.exists():
            continue
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)

        for entry in manifest["frames"]:
            img_src = mod_dir / entry["file"]
            if not img_src.exists():
                continue
            img_dst = images_dir / entry["file"]
            shutil.copyfile(img_src, img_dst)

            from dataset import _image_size
            label_path = labels_dir / (img_dst.stem + ".txt")
            lines = []
            if entry.get("exist") and entry.get("gt_rect"):
                x, y, w, h = entry["gt_rect"]
                img_w, img_h = _image_size(img_src)
                if w > 0 and h > 0:
                    xc, yc = (x + w / 2) / img_w, (y + h / 2) / img_h
                    nw, nh = w / img_w, h / img_h
                    lines.append(f"0 {xc:.6f} {yc:.6f} {nw:.6f} {nh:.6f}")
                    n_boxes += 1
            label_path.write_text("\n".join(lines), encoding="utf-8")
            n_images += 1
    return n_images, n_boxes


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".", help="теки з train/ і val/ (сирі послідовності)")
    ap.add_argument("--out", default="./yolo_dataset")
    ap.add_argument("--modality", default="infrared")
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    root = Path(args.root)
    out = Path(args.out)

    sessions = collect_sessions(root)
    train_seqs, val_seqs, val_session_names = split_sessions(sessions, args.val_frac, args.seed)

    print(f"сесій усього: {len(sessions)}, val-сесій: {len(val_session_names)}")
    print(f"val-сесії: {val_session_names}")
    print(f"train-послідовностей: {len(train_seqs)}, val-послідовностей: {len(val_seqs)}")

    n_tr_img, n_tr_box = build_from_seq_list(
        train_seqs, out / "images" / "train", out / "labels" / "train", args.modality)
    n_va_img, n_va_box = build_from_seq_list(
        val_seqs, out / "images" / "val", out / "labels" / "val", args.modality)

    print(f"[train] {n_tr_img} зображень, {n_tr_box} боксів")
    print(f"[val]   {n_va_img} зображень, {n_va_box} боксів")

    write_data_yaml(out, out / "images" / "train", out / "images" / "val")


if __name__ == "__main__":
    main()
