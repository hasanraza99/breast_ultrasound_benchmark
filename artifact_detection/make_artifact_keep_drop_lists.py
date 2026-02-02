#!/usr/bin/env python3
import argparse
from pathlib import Path
import re

import pandas as pd


def lesion_mask_to_image_path(lesion_path: str) -> str:
    """
    Convert lesion mask path to corresponding image path.

    Example:
      /workspace/breastdataset_NORMALIZED/breastdataset_NORMALIZED/external/BUSI/benign/masks/benign_1_mask.png
      -> .../external/BUSI/benign/images/benign_1.png
    """
    p = Path(lesion_path)
    # swap masks -> images
    if "masks" not in p.parts:
        return lesion_path  # fallback: return as-is
    parts = list(p.parts)
    idx = parts.index("masks")
    parts[idx] = "images"
    img_dir = Path(*parts[:-1])

    # strip _mask / _lesionmask / _seg suffixes from stem
    stem = p.stem
    stem = re.sub(r"(_mask|_lesionmask|_seg)$", "", stem, flags=re.IGNORECASE)

    # assume PNG images
    img_path = img_dir / f"{stem}.png"
    return str(img_path)


def main():
    parser = argparse.ArgumentParser(
        description="Create kept/dropped image lists based on artifact-overlap thresholds."
    )
    parser.add_argument(
        "--overlap-csv",
        type=Path,
        default=Path("/workspace/ui_dash_vs_lesion_overlap.csv"),
        help="CSV with overlap stats between union (artifact) and lesion masks.",
    )
    parser.add_argument(
        "--out-root",
        type=Path,
        default=Path("/workspace/artifact_filter_v1"),
        help="Output root for kept/dropped lists and summary.",
    )
    parser.add_argument(
        "--frac-th",
        type=float,
        default=0.02,
        help="Threshold on frac_lesion_covered; drop if value > frac_th.",
    )
    parser.add_argument(
        "--min-inter-px",
        type=int,
        default=50,
        help="Minimum intersection pixels for a sample to be considered for dropping.",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.overlap_csv)

    # Only consider rows with a lesion mask (BUSI normals with no lesion are irrelevant here)
    df = df[df["lesion_path"].notna() & (df["lesion_path"] != "")]
    n_total = len(df)
    print(f"[INFO] Loaded {n_total} overlap rows with lesion masks.")

    if "frac_lesion_covered" not in df.columns or "inter_px" not in df.columns:
        raise ValueError("overlap CSV must contain 'frac_lesion_covered' and 'inter_px' columns.")

    # Ensure numeric
    df["frac_lesion_covered"] = pd.to_numeric(df["frac_lesion_covered"], errors="coerce").fillna(0.0)
    df["inter_px"] = pd.to_numeric(df["inter_px"], errors="coerce").fillna(0.0)

    # Drop rule: artifact covers more than frac_th of lesion AND intersection is non-trivial
    drop_mask = (df["frac_lesion_covered"] > args.frac_th) & (df["inter_px"] >= args.min_inter_px)
    df["drop"] = drop_mask.astype(int)

    n_drop = int(drop_mask.sum())
    n_keep = n_total - n_drop
    pct_drop = 100.0 * n_drop / n_total if n_total > 0 else 0.0

    print(
        f"[INFO] Using frac_th={args.frac_th}, min_inter_px={args.min_inter_px}: "
        f"drop {n_drop}/{n_total} ({pct_drop:.2f}%)."
    )

    # Map lesion masks to image paths
    df["image_path"] = df["lesion_path"].apply(lesion_mask_to_image_path)

    # Output directory
    out_root = args.out_root.resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    kept_paths = df.loc[df["drop"] == 0, "image_path"].tolist()
    dropped_paths = df.loc[df["drop"] == 1, "image_path"].tolist()

    kept_txt = out_root / "kept_images.txt"
    dropped_txt = out_root / "dropped_images.txt"
    summary_csv = out_root / "filter_summary.csv"

    with kept_txt.open("w", encoding="utf-8") as f:
        for p in kept_paths:
            f.write(f"{p}\n")

    with dropped_txt.open("w", encoding="utf-8") as f:
        for p in dropped_paths:
            f.write(f"{p}\n")

    # Save full summary CSV (overlap + drop flag + thresholds used)
    df["frac_th"] = args.frac_th
    df["min_inter_px_th"] = args.min_inter_px
    df.to_csv(summary_csv, index=False)

    print(f"[OK] Kept images list:   {kept_txt} (n={len(kept_paths)})")
    print(f"[OK] Dropped images list:{dropped_txt} (n={len(dropped_paths)})")
    print(f"[OK] Filter summary CSV: {summary_csv}")


if __name__ == "__main__":
    main()
