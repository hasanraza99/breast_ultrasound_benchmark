#!/usr/bin/env python3
import argparse
from pathlib import Path
import re
import sys
from collections import defaultdict

import numpy as np
import cv2 as cv
from tqdm import tqdm

EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}

# Suffix handling
rx_union_suffix = re.compile(r"_ui_dash_union$", re.IGNORECASE)
rx_mask_suffix  = re.compile(r"(_mask|_lesionmask|_seg)$", re.IGNORECASE)
rx_fovcrop      = re.compile(r"_fovcrop$", re.IGNORECASE)


def union_key(stem: str) -> str:
    """
    Build matching key from a union mask filename stem.
    Example:
      benign_1_ui_dash_union   -> benign_1
      malignant_53_lesion1_ui_dash_union -> malignant_53_lesion1
    """
    s = stem.lower()
    s = rx_union_suffix.sub("", s)
    s = rx_fovcrop.sub("", s)
    s = re.sub(r"_+", "_", s)
    return s


def lesion_key(stem: str) -> str:
    """
    Build matching key from a lesion mask filename stem.
    Example:
      benign_1_mask            -> benign_1
      bus_0002_l_mask          -> bus_0002_l
      malignant_53_lesion1_mask -> malignant_53_lesion1
    """
    s = stem.lower()
    s = rx_mask_suffix.sub("", s)
    s = rx_fovcrop.sub("", s)
    s = re.sub(r"_+", "_", s)
    return s


def load_mask_binary(path: Path) -> np.ndarray:
    """Load mask as binary uint8 {0,1}."""
    m = cv.imread(str(path), cv.IMREAD_GRAYSCALE)
    if m is None:
        raise RuntimeError(f"Failed to read mask: {path}")
    return (m > 0).astype(np.uint8)


def build_lesion_index(dataset_root: Path):
    """
    Build index: (dataset_rel, key) -> lesion_mask_path

    dataset_rel is path from dataset_root down to parent of 'masks', e.g.:
      external/BUSI/benign
      internal/BUS-BRA/malignant
    """
    index = {}
    collisions = 0
    total = 0

    dataset_root = dataset_root.resolve()

    for p in dataset_root.rglob("*"):
        if not p.is_file():
            continue
        if p.suffix.lower() not in EXTS:
            continue
        if "masks" not in p.parts:
            continue
        if ".ipynb_checkpoints" in p.parts:
            continue

        total += 1
        rel = p.relative_to(dataset_root)
        parts = rel.parts
        try:
            m_idx = parts.index("masks")
        except ValueError:
            continue

        dataset_rel = Path(*parts[:m_idx])  # e.g. external/BUSI/benign
        key = lesion_key(p.stem)
        k = (dataset_rel.as_posix(), key)

        if k in index:
            collisions += 1
        index[k] = p

    print(f"[INFO] Lesion index: {total} files, {len(index)} unique keys.")
    if collisions:
        print(f"[WARN] Lesion index had {collisions} key collisions (last one wins).")

    return index


def main():
    parser = argparse.ArgumentParser(
        description="Measure overlap between ui_dash_union_masks and lesion masks."
    )
    parser.add_argument(
        "--union-root",
        type=Path,
        default=Path("/workspace/ui_dash_union_masks"),
        help="Root directory containing *_ui_dash_union.png masks.",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("/workspace/breastdataset_NORMALIZED/breastdataset_NORMALIZED"),
        help="Root of normalized breast dataset (with **/masks/*.png).",
    )
    parser.add_argument(
        "--out-csv",
        type=Path,
        default=Path("/workspace/ui_dash_vs_lesion_overlap.csv"),
        help="Output CSV path.",
    )

    args = parser.parse_args()
    union_root = args.union_root.resolve()
    dataset_root = args.dataset_root.resolve()
    out_csv = args.out_csv.resolve()

    print(f"[INFO] union_root   = {union_root}")
    print(f"[INFO] dataset_root = {dataset_root}")
    print(f"[INFO] out_csv      = {out_csv}")

    lesion_index = build_lesion_index(dataset_root)

    # Collect union masks
    union_files = []
    for p in union_root.rglob("*"):
        if not p.is_file():
            continue
        if p.suffix.lower() not in EXTS:
            continue
        if ".ipynb_checkpoints" in p.parts:
            continue
        union_files.append(p)

    if not union_files:
        print("[ERROR] No union masks found under union_root.", file=sys.stderr)
        sys.exit(1)

    print(f"[INFO] Found {len(union_files)} union masks.")

    # Prepare CSV
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    f = out_csv.open("w", encoding="utf-8")
    f.write(
        "dataset_rel,class,key,union_path,lesion_path,"
        "lesion_px,union_px,inter_px,iou,"
        "frac_lesion_covered,frac_union_inside_lesion\n"
    )

    unmatched_union = 0
    stats_by_dataset = defaultdict(lambda: {"pairs": 0, "unmatched_union": 0})

    for upath in tqdm(union_files, desc="Measuring overlaps"):
        rel = upath.relative_to(union_root)
        parts = rel.parts

        if len(parts) < 2:
            # expect something like internal/BUS-BRA/benign/xxx.png
            continue

        # dataset_rel = internal/BUS-BRA/benign, external/BUSI/malignant, etc.
        dataset_rel = Path(*parts[:-1])  # all except filename
        dataset_str = dataset_rel.as_posix()
        # crude "class" = last component (benign/malignant/normal)
        class_name = dataset_rel.parts[-1] if len(dataset_rel.parts) >= 1 else ""

        key = union_key(upath.stem)
        lesion_path = lesion_index.get((dataset_str, key), None)

        if lesion_path is None:
            unmatched_union += 1
            stats_by_dataset[dataset_str]["unmatched_union"] += 1
            # still log row with empty lesion path and zeros
            f.write(
                f"{dataset_str},{class_name},{key},{upath},,"
                f"0,0,0,0,0,0\n"
            )
            continue

        stats_by_dataset[dataset_str]["pairs"] += 1

        union_mask = load_mask_binary(upath)
        lesion_mask = load_mask_binary(lesion_path)

        # Resize union to lesion shape if needed
        if union_mask.shape != lesion_mask.shape:
            h, w = lesion_mask.shape[:2]
            union_mask = cv.resize(
                union_mask, (w, h), interpolation=cv.INTER_NEAREST
            )

        # Pixel counts
        lesion_px = int(lesion_mask.sum())
        union_px = int(union_mask.sum())
        inter = (lesion_mask & union_mask)
        inter_px = int(inter.sum())

        # Convert to "number of pixels" if you prefer counts of 1s not 255s
        # but since both are {0,1}, sums are already counts.

        denom_iou = lesion_px + union_px - inter_px
        iou = inter_px / denom_iou if denom_iou > 0 else 0.0

        frac_lesion_covered = (
            inter_px / lesion_px if lesion_px > 0 else 0.0
        )
        frac_union_inside_lesion = (
            inter_px / union_px if union_px > 0 else 0.0
        )

        f.write(
            f"{dataset_str},{class_name},{key},{upath},{lesion_path},"
            f"{lesion_px},{union_px},{inter_px},"
            f"{iou:.6f},{frac_lesion_covered:.6f},{frac_union_inside_lesion:.6f}\n"
        )

    f.close()

    print("\n[SUMMARY] Overlap stats by dataset_rel:")
    for ds, d in sorted(stats_by_dataset.items()):
        print(
            f"  {ds}: pairs={d['pairs']}, "
            f"unmatched_union={d['unmatched_union']}"
        )

    print(f"\n[INFO] Total unmatched union masks: {unmatched_union}")
    print(f"[OK] Overlap CSV written to: {out_csv}")


if __name__ == "__main__":
    main()
