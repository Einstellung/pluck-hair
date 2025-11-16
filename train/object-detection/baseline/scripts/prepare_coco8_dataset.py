#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List

import yaml
from PIL import Image


def _load_names(names_yaml: Path) -> Dict[int, str]:
    with names_yaml.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    names = data.get("names")
    if isinstance(names, dict):
        return {int(k): v for k, v in names.items()}
    if isinstance(names, list):
        return {idx: name for idx, name in enumerate(names)}
    raise ValueError(f"Unsupported names format in {names_yaml}")


def _ensure_symlink(src: Path, dst: Path) -> None:
    if dst.exists():
        if dst.is_symlink() or dst.is_file():
            dst.unlink()
        elif dst.is_dir():
            return
    dst.parent.mkdir(parents=True, exist_ok=True)
    os.symlink(src.resolve(), dst)


def _convert_split(
    split: str,
    src_images: Path,
    src_labels: Path,
    dst_images: Path,
    dst_annotations: Path,
    cat_map: Dict[int, int],
    categories: List[Dict[str, str]],
) -> None:
    _ensure_symlink(src_images, dst_images)

    images = []
    annotations = []
    img_id = 1
    ann_id = 1
    for img_path in sorted(src_images.glob("*.jpg")):
        with Image.open(img_path) as im:
            width, height = im.size
        images.append({"id": img_id, "file_name": img_path.name, "width": width, "height": height})
        label_path = src_labels / f"{img_path.stem}.txt"
        if label_path.exists():
            with label_path.open("r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) != 5:
                        continue
                    cls = int(parts[0])
                    if cls not in cat_map:
                        continue
                    x_center, y_center, box_w_rel, box_h_rel = map(float, parts[1:])
                    box_w = box_w_rel * width
                    box_h = box_h_rel * height
                    x_min = max(0.0, x_center * width - box_w / 2)
                    y_min = max(0.0, y_center * height - box_h / 2)
                    annotations.append(
                        {
                            "id": ann_id,
                            "image_id": img_id,
                            "category_id": cat_map[cls],
                            "bbox": [x_min, y_min, box_w, box_h],
                            "area": box_w * box_h,
                            "iscrowd": 0,
                            "segmentation": [],
                        }
                    )
                    ann_id += 1
        img_id += 1

    dst_annotations.parent.mkdir(parents=True, exist_ok=True)
    with dst_annotations.open("w", encoding="utf-8") as f:
        json.dump({"images": images, "annotations": annotations, "categories": categories}, f)
    print(f"[prepare_coco8] Wrote {dst_annotations} ({len(images)} images, {len(annotations)} annotations)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert YOLO-format coco8 to COCO JSON for Detectron2")
    parser.add_argument("--source-root", default="dataset/ultra/coco8", help="YOLO-format coco8 root containing images/labels")
    parser.add_argument("--target-root", default="dataset/coco8", help="Destination root for Detectron2-ready dataset")
    parser.add_argument("--names-yaml", default=None, help="Path to coco8.yaml (if not auto-detected)")
    parser.add_argument("--allow-missing-classes", action="store_true", help="Skip labels whose class ids are out of range")
    args = parser.parse_args()

    source_root = Path(args.source_root).resolve()
    target_root = Path(args.target_root).resolve()
    names_yaml = Path(args.names_yaml).resolve() if args.names_yaml else (source_root.parent / "coco8.yaml").resolve()

    name_dict = _load_names(names_yaml)
    categories = [{"id": idx + 1, "name": name} for idx, name in sorted(name_dict.items())]
    cat_map = {cls: cls + 1 for cls in name_dict.keys()}

    for split in ("train", "val"):
        src_images = source_root / "images" / split
        src_labels = source_root / "labels" / split
        dst_images = target_root / "images" / split
        dst_ann = target_root / "annotations" / f"instances_{split}.json"
        if not src_images.exists():
            raise FileNotFoundError(f"Missing source images: {src_images}")
        if not src_labels.exists():
            raise FileNotFoundError(f"Missing source labels: {src_labels}")
        _convert_split(split, src_images, src_labels, dst_images, dst_ann, cat_map, categories)


if __name__ == "__main__":
    main()
