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

# Suffixes to strip
rx_dash_suffix   = re.compile(r"_dash$", re.IGNORECASE)
rx_ui_mask       = re.compile(r"_ui_mask$", re.IGNORECASE)
rx_uimask        = re.compile(r"_uimask$", re.IGNORECASE)
rx_ui_suffix     = re.compile(r"_ui$", re.IGNORECASE)
rx_mask_suffix   = re.compile(r"_mask$", re.IGNORECASE)
rx_fovcrop       = re.compile(r"_fovcrop$", re.IGNORECASE)

def dash_key(stem: str) -> str:
    """
    Build matching key from a dash mask filename stem.
    - remove '_dash'
    - optionally remove '_fovcrop'
    - KEEP '_lesionX' if present
    """
    s = stem.lower()
    s = rx_dash_suffix.sub("", s)
    s = rx_fovcrop.sub("", s)
    s = re.sub(r"_+", "_", s)
    return s

def ui_key(stem: str) -> str:
    """
    Build matching key from a UI mask filename stem.
    - remove '_ui_mask' / '_uimask' / '_ui' / '_mask'
    - optionally remove '_fovcrop'
    - KEEP '_lesionX' if present
    """
    s = stem.lower()
    s = rx_ui_mask.sub("", s)
    s = rx_uimask.sub("", s)
    s = rx_ui_suffix.sub("", s)
    s = rx_mask_suffix.sub("", s)
    s = rx_fovcrop.sub("", s)
    s = re.sub(r"_+", "_", s)
    return s

def load_mask_binary(path: Path) -> np.ndarray:
    """Load mask as binary uint8 {0,255}."""
    m = cv.imread(str(path), cv.IMREAD_GRAYSCALE)
    if m is None:
        raise RuntimeError(f"Failed to read mask: {path}")
    m = (m > 0).astype(np.uint8) * 255
    return m

def build_ui_index(ui_root: Path):
    """
    Build index: (dataset_rel, key) -> [ui_mask_paths...]

    dataset_rel is path from ui_root down to parent of 'ui_masks', e.g.:
      external/BUSI/malignant
      internal/USG/benign
    """
    index: dict[tuple[str, str], list[Path]] = defaultdict(list)

    ui_root = ui_root.resolve()
    total_files = 0

    for p in ui_root.rglob("*"):
        if not p.is_file():
            continue
        if p.suffix.lower() not in EXTS:
            continue
        if "ui_masks" not in p.parts:
            continue

        total_files += 1
        rel = p.relative_to(ui_root)
        parts = rel.parts
        try:
            ui_idx = parts.index("ui_masks")
        except ValueError:
            continue

        dataset_rel = Path(*parts[:ui_idx])  # e.g. external/BUSI/malignant
        key = ui_key(p.stem)
        index[(dataset_rel.as_posix(), key)].append(p)

    num_keys = len(index)
    multi_keys = sum(1 for v in index.values() if len(v) > 1)

    print(f"[INFO] UI index: {total_files} files, {num_keys} unique keys.")
    if multi_keys:
        print(
            f"[WARN] {multi_keys} keys have multiple UI masks; "
            f"they will be OR'ed together per key."
        )

    return index

def main():
    parser = argparse.ArgumentParser(
        description="Create per-lesion union of UI and dash masks into ui_dash_union_masks."
    )
    parser.add_argument(
        "--dash-root",
        type=Path,
        default=Path("/workspace/dash_preds_v2"),
        help="Root directory containing masks_dashes_pred folders.",
    )
    parser.add_argument(
        "--ui-root",
        type=Path,
        default=Path("/workspace/ui_masks_all"),
        help="Root directory containing ui_masks folders.",
    )
    parser.add_argument(
        "--out-root",
        type=Path,
        default=Path("/workspace/ui_dash_union_masks"),
        help="Output root for union masks.",
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=Path("/workspace/ui_dash_union_metadata.csv"),
        help="CSV metadata logfile.",
    )

    args = parser.parse_args()
    dash_root = args.dash_root.resolve()
    ui_root   = args.ui_root.resolve()
    out_root  = args.out_root.resolve()

    print(f"[INFO] dash_root = {dash_root}")
    print(f"[INFO] ui_root   = {ui_root}")
    print(f"[INFO] out_root  = {out_root}")

    ui_index = build_ui_index(ui_root)

    # Collect dash masks
    dash_files = []
    for p in dash_root.rglob("*"):
        if not p.is_file():
            continue
        if p.suffix.lower() not in EXTS:
            continue
        if "masks_dashes_pred" not in p.parts:
            continue
        dash_files.append(p)

    if not dash_files:
        print("[ERROR] No dash masks found under dash_root.", file=sys.stderr)
        sys.exit(1)

    print(f"[INFO] Found {len(dash_files)} dash masks.")

    # For summary
    stats = defaultdict(lambda: {"total": 0, "ui_matched": 0, "dash_only": 0})

    # Prepare metadata CSV
    args.metadata.parent.mkdir(parents=True, exist_ok=True)
    with args.metadata.open("w", encoding="utf-8") as f_meta:
        f_meta.write(
            "dataset_rel,dash_path,ui_paths,out_path,"
            "dash_sum,ui_sum,union_sum,match_type,key\n"
        )

        for dash_path in tqdm(dash_files, desc="Processing dash masks"):
            rel = dash_path.relative_to(dash_root)
            parts = rel.parts
            try:
                dash_idx = parts.index("masks_dashes_pred")
            except ValueError:
                continue

            dataset_rel = Path(*parts[:dash_idx])  # e.g. external/BUSI/malignant
            dataset_key = dataset_rel.as_posix()

            dash_img = load_mask_binary(dash_path)
            dash_sum = int(dash_img.sum() // 255)

            key = dash_key(dash_path.stem)
            ui_paths = ui_index.get((dataset_key, key), [])

            if ui_paths:
                ui_img = None
                for up in ui_paths:
                    m = load_mask_binary(up)
                    if m.shape != dash_img.shape:
                        h, w = dash_img.shape[:2]
                        m = cv.resize(m, (w, h), interpolation=cv.INTER_NEAREST)
                    if ui_img is None:
                        ui_img = m
                    else:
                        ui_img = cv.bitwise_or(ui_img, m)

                match_type = f"dash+ui({len(ui_paths)})"
                ui_sum = int(ui_img.sum() // 255)
                union = cv.bitwise_or(dash_img, ui_img)
                stats[dataset_key]["ui_matched"] += 1
            else:
                match_type = "dash_only"
                ui_sum = 0
                union = dash_img.copy()
                stats[dataset_key]["dash_only"] += 1

            union_sum = int(union.sum() // 255)
            stats[dataset_key]["total"] += 1

            # Output path: keep lesion info, drop only '_dash'
            base = rx_dash_suffix.sub("", dash_path.stem)
            out_dir = out_root / dataset_rel
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / f"{base}_ui_dash_union.png"

            cv.imwrite(str(out_path), union)

            ui_paths_str = ";".join(str(p) for p in ui_paths)
            f_meta.write(
                f"{dataset_key},{dash_path},{ui_paths_str},{out_path},"
                f"{dash_sum},{ui_sum},{union_sum},{match_type},{key}\n"
            )

    print("\n[SUMMARY] per dataset_rel:")
    for dataset_key, d in sorted(stats.items()):
        print(
            f"  {dataset_key}: total={d['total']}, "
            f"ui_matched={d['ui_matched']}, dash_only={d['dash_only']}"
        )

    print(f"\n[OK] Union masks written under: {out_root}")
    print(f"[OK] Metadata CSV: {args.metadata}")

if __name__ == "__main__":
    main()
