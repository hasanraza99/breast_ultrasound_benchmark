#!/usr/bin/env python3
import json
from pathlib import Path

ROOT = Path("/workspace/breastdataset_NORMALIZED/breastdataset_NORMALIZED")

INTERNAL_NAMES = ["BUS-BRA", "BUS_UC", "BUS-UCLM", "USG", "UDIAT"]
EXTERNAL_NAMES = ["BUSI", "QAMEBI"]


def count_images(ds_root: Path) -> int:
    """Count PNG images under benign/images and malignant/images."""
    total = 0
    for cls in ["benign", "malignant"]:
        img_dir = ds_root / cls / "images"
        if not img_dir.exists():
            continue
        total += sum(1 for _ in img_dir.glob("*.png"))
    return total


def main():
    if not ROOT.exists():
        raise SystemExit(f"Dataset root not found: {ROOT}")

    internal_datasets = {}
    external_datasets = {}
    source_sizes = {}

    # Internal: paths + sizes
    for name in INTERNAL_NAMES:
        ds_root = ROOT / "internal" / name
        internal_datasets[name] = str(ds_root)
        n = count_images(ds_root)
        source_sizes[name] = n
        print(f"[INTERNAL] {name}: path={ds_root}, images={n}")

    # External: paths only (no source_sizes needed)
    for name in EXTERNAL_NAMES:
        ds_root = ROOT / "external" / name
        external_datasets[name] = str(ds_root)
        print(f"[EXTERNAL] {name}: path={ds_root}")

    config = {
        "internal_datasets": internal_datasets,
        "external_datasets": external_datasets,
        "source_sizes": source_sizes,
    }

    out_path = Path("dataset_config.json").resolve()
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    print(f"\n[OK] Wrote config to {out_path}")


if __name__ == "__main__":
    main()
