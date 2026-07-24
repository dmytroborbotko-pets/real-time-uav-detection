# coding=utf-8
"""
Будує YOLO-сумісний датасет (Ultralytics detect format) з уже нарізаних
кадрів Anti-UAV-RGBT (train/*/infrared, val/*/infrared).
"""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from pathlib import Path

CLASS_NAMES = ["uav"]


def build_split(seq_root: Path, images_dir: Path, labels_dir: Path, modality: str = "infrared") -> tuple[int, int]:
    """Копіює кадри й пише YOLO-мітки для всіх послідовностей у seq_root."""
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    n_images = 0
    n_boxes = 0
    for seq_dir in sorted(p for p in seq_root.iterdir() if p.is_dir()):
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

            label_path = labels_dir / (img_dst.stem + ".txt")
            lines = []
            if entry.get("exist") and entry.get("gt_rect"):
                x, y, w, h = entry["gt_rect"]
                img_w, img_h = _image_size(img_src)
                if w > 0 and h > 0:
                    xc = (x + w / 2) / img_w
                    yc = (y + h / 2) / img_h
                    nw = w / img_w
                    nh = h / img_h
                    lines.append(f"0 {xc:.6f} {yc:.6f} {nw:.6f} {nh:.6f}")
                    n_boxes += 1
            label_path.write_text("\n".join(lines), encoding="utf-8")
            n_images += 1

    return n_images, n_boxes


def _image_size(path: Path) -> tuple[int, int]:
    from PIL import Image

    with Image.open(path) as im:
        return im.size


def write_data_yaml(out_dir: Path, train_images: Path, val_images: Path) -> Path:
    yaml_path = out_dir / "data.yaml"
    names_block = "\n".join(f"  {i}: {name}" for i, name in enumerate(CLASS_NAMES))
    train_rel = train_images.relative_to(out_dir)
    val_rel = val_images.relative_to(out_dir)
    content = (
        f"train: {train_rel}\n"
        f"val: {val_rel}\n"
        f"nc: {len(CLASS_NAMES)}\n"
        f"names:\n{names_block}\n"
    )
    yaml_path.write_text(content, encoding="utf-8")
    return yaml_path


def build_dataset(root: Path, out: Path, modality: str = "infrared") -> None:
    out.mkdir(parents=True, exist_ok=True)

    for split, src_name in (("train", "train"), ("val", "val")):
        seq_root = root / src_name
        images_dir = out / "images" / split
        labels_dir = out / "labels" / split
        n_images, n_boxes = build_split(seq_root, images_dir, labels_dir, modality)
        print(f"[{split}] {n_images} зображень, {n_boxes} боксів -> {images_dir}")

    yaml_path = write_data_yaml(out, out / "images" / "train", out / "images" / "val")
    print(f"data.yaml -> {yaml_path}")

def _make_fake_sequence(seq_dir: Path, modality: str, n_frames: int = 3) -> None:
    from PIL import Image

    mod_dir = seq_dir / modality
    mod_dir.mkdir(parents=True)

    manifest = {"stride": 1, "frames_source": n_frames, "negatives": 0, "frames": []}
    for i in range(n_frames):
        name = f"seq_{modality}_{i:06d}.jpg"
        Image.new("RGB", (64, 48), color=(i * 10, 0, 0)).save(mod_dir / name)
        has_target = i != 1
        entry = {"file": name, "frame_index": i, "exist": int(has_target)}
        if has_target:
            entry["gt_rect"] = [10, 5, 20, 10]
        manifest["frames"].append(entry)

    (mod_dir / "_sampled.json").write_text(json.dumps(manifest), encoding="utf-8")


def self_test() -> None:
    """Тест перед реальним датасетом."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        root = tmp / "root"
        out = tmp / "out"

        _make_fake_sequence(root / "train" / "seq1", "infrared", n_frames=3)
        _make_fake_sequence(root / "val" / "seq2", "infrared", n_frames=2)

        build_dataset(root, out)

        train_images = sorted((out / "images" / "train").glob("*.jpg"))
        val_images = sorted((out / "images" / "val").glob("*.jpg"))
        assert len(train_images) == 3, train_images
        assert len(val_images) == 2, val_images

        label_pos = (out / "labels" / "train" / "seq_infrared_000000.txt").read_text()
        cls, xc, yc, w, h = (float(v) if i else int(v) for i, v in enumerate(label_pos.split()))
        assert cls == 0
        expected_xc, expected_yc = (10 + 20 / 2) / 64, (5 + 10 / 2) / 48
        assert abs(xc - expected_xc) < 1e-6 and abs(yc - expected_yc) < 1e-6
        assert 0 < w < 1 and 0 < h < 1

        label_neg_path = out / "labels" / "train" / "seq_infrared_000001.txt"
        assert label_neg_path.exists()
        assert label_neg_path.read_text() == ""

        assert (out / "data.yaml").exists()

    print("self_test: OK")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="./Anti-UAV-RGBT",
                     help="теки з train/ і val/, кожна зі своїми послідовностями")
    ap.add_argument("--out", default="./Anti-UAV-RGBT/yolo_dataset",
                     help="куди писати images/, labels/, data.yaml")
    ap.add_argument("--modality", default="infrared", choices=["infrared", "visible"])
    ap.add_argument("--skip-self-test", action="store_true",
                     help="пропустити прогін на синтетичних даних перед реальною конвертацією")
    args = ap.parse_args()

    if not args.skip_self_test:
        self_test()

    build_dataset(Path(args.root), Path(args.out), args.modality)


if __name__ == "__main__":
    main()
