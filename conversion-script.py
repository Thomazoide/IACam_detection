import argparse
import json
import os
import random
import shutil
from pathlib import Path


def load_coco(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def build_class_map(categories: list[dict]) -> tuple[dict, list[str]]:
    categories_sorted = sorted(categories, key=lambda c: c["id"])
    id_to_idx = {c["id"]: i for i, c in enumerate(categories_sorted)}
    names = [c["name"] for c in categories_sorted]
    return id_to_idx, names


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def safe_link_or_copy(src: Path, dst: Path) -> None:
    if dst.exists():
        return
    ensure_dir(dst.parent)
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


def coco_to_yolo(
    coco: dict,
    images_root: Path,
    out_images_dir: Path,
    out_labels_dir: Path,
    id_to_idx: dict,
    skip_ignore: bool,
    skip_uncertain: bool,
    skip_crowd: bool,
) -> tuple[int, int]:
    images = coco.get("images", [])
    annotations = coco.get("annotations", [])

    anns_by_image: dict[int, list[dict]] = {}
    for ann in annotations:
        if skip_ignore and ann.get("ignore", False):
            continue
        if skip_uncertain and ann.get("uncertain", False):
            continue
        if skip_crowd and ann.get("iscrowd", 0):
            continue
        anns_by_image.setdefault(ann["image_id"], []).append(ann)

    missing_images = 0
    written_labels = 0

    for img in images:
        file_name = img["file_name"]
        src_img = images_root / file_name
        rel_path = Path(file_name)
        dst_img = out_images_dir / rel_path
        if src_img.exists():
            safe_link_or_copy(src_img, dst_img)
        else:
            missing_images += 1

        w = img["width"]
        h = img["height"]
        label_lines = []
        for ann in anns_by_image.get(img["id"], []):
            cat_id = ann["category_id"]
            if cat_id not in id_to_idx:
                continue
            x, y, bw, bh = ann["bbox"]
            cx = x + bw / 2.0
            cy = y + bh / 2.0
            cx_n = clamp01(cx / w)
            cy_n = clamp01(cy / h)
            bw_n = clamp01(bw / w)
            bh_n = clamp01(bh / h)
            label_lines.append(
                f"{id_to_idx[cat_id]} {cx_n:.6f} {cy_n:.6f} {bw_n:.6f} {bh_n:.6f}"
            )

        ensure_dir(out_labels_dir)
        label_path = out_labels_dir / rel_path.with_suffix(".txt")
        ensure_dir(label_path.parent)
        label_path.write_text("\n".join(label_lines), encoding="utf-8")
        written_labels += 1

    return missing_images, written_labels


def write_data_yaml(out_dir: Path, names: list[str]) -> Path:
    data_yaml = out_dir / "data.yaml"
    lines = [
        f"path: {out_dir.as_posix()}",
        "train: images/train",
        "val: images/val",
        f"nc: {len(names)}",
        "names:",
    ]
    for i, n in enumerate(names):
        lines.append(f"  {i}: {n}")
    data_yaml.write_text("\n".join(lines), encoding="utf-8")
    return data_yaml


def split_images(images: list[dict], val_split: float, seed: int) -> tuple[list[dict], list[dict]]:
    rng = random.Random(seed)
    images = images[:]
    rng.shuffle(images)
    n_val = max(1, int(len(images) * val_split))
    val = images[:n_val]
    train = images[n_val:]
    return train, val


def filter_coco_by_images(coco: dict, images_subset: list[dict]) -> dict:
    image_ids = {img["id"] for img in images_subset}
    annotations = [ann for ann in coco.get("annotations", []) if ann["image_id"] in image_ids]
    return {
        "type": coco.get("type", "instance"),
        "images": images_subset,
        "annotations": annotations,
        "categories": coco.get("categories", []),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert COCO to YOLO and train yolov8s.")
    parser.add_argument(
        "--train-coco",
        type=Path,
        default=Path("tiny_set/annotations/tiny_set_train.json"),
        help="Path to COCO train JSON.",
    )
    parser.add_argument(
        "--val-coco",
        type=Path,
        default=Path("tiny_set/annotations/tiny_set_test.json"),
        help="Path to COCO val JSON. If missing, a split will be created.",
    )
    parser.add_argument(
        "--images-root",
        type=Path,
        default=Path("tiny_set"),
        help="Root folder where COCO file_name paths are relative to.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("tiny_set/yolo8"),
        help="Output folder for YOLO images/labels and data.yaml.",
    )
    parser.add_argument("--val-split", type=float, default=0.2, help="Val split ratio.")
    parser.add_argument("--seed", type=int, default=1337, help="Random seed for split.")
    parser.add_argument("--keep-uncertain", action="store_true", help="Keep uncertain labels.")
    parser.add_argument("--keep-ignore", action="store_true", help="Keep ignore labels.")
    parser.add_argument("--keep-crowd", action="store_true", help="Keep iscrowd labels.")
    parser.add_argument("--no-train", action="store_true", help="Only convert, do not train.")
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs.")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size.")
    parser.add_argument("--batch", type=int, default=16, help="Batch size.")
    parser.add_argument("--device", type=str, default="", help="Device, e.g. 0 or cpu.")
    parser.add_argument("--project", type=str, default="runs/train", help="Ultralytics project dir.")
    parser.add_argument("--name", type=str, default="yolov8s_tiny_set", help="Run name.")
    args = parser.parse_args()

    train_coco = load_coco(args.train_coco)
    if args.val_coco.exists():
        val_coco = load_coco(args.val_coco)
    else:
        val_coco = None

    id_to_idx, names = build_class_map(train_coco.get("categories", []))

    out_dir = args.out_dir
    train_images_dir = out_dir / "images/train"
    train_labels_dir = out_dir / "labels/train"
    val_images_dir = out_dir / "images/val"
    val_labels_dir = out_dir / "labels/val"
    ensure_dir(train_images_dir)
    ensure_dir(train_labels_dir)
    ensure_dir(val_images_dir)
    ensure_dir(val_labels_dir)

    if val_coco is None:
        images = train_coco.get("images", [])
        train_images, val_images = split_images(images, args.val_split, args.seed)
        full_coco = train_coco
        train_coco = filter_coco_by_images(full_coco, train_images)
        val_coco = filter_coco_by_images(full_coco, val_images)

    skip_ignore = not args.keep_ignore
    skip_uncertain = not args.keep_uncertain
    skip_crowd = not args.keep_crowd

    train_missing, _ = coco_to_yolo(
        train_coco,
        args.images_root,
        train_images_dir,
        train_labels_dir,
        id_to_idx,
        skip_ignore,
        skip_uncertain,
        skip_crowd,
    )
    val_missing, _ = coco_to_yolo(
        val_coco,
        args.images_root,
        val_images_dir,
        val_labels_dir,
        id_to_idx,
        skip_ignore,
        skip_uncertain,
        skip_crowd,
    )

    data_yaml = write_data_yaml(out_dir, names)

    if train_missing or val_missing:
        print(f"[WARN] Missing images: train={train_missing}, val={val_missing}")
        print("       Make sure you extracted the images (e.g., from train.tar.gz/test.tar.gz).")

    if args.no_train:
        print(f"Conversion complete. data.yaml at: {data_yaml}")
        return

    from ultralytics import YOLO

    model = YOLO("yolov8s.pt")
    model.train(
        data=str(data_yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=args.project,
        name=args.name,
    )


if __name__ == "__main__":
    main()
