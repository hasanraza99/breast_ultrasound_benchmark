#!/usr/bin/env python3
import argparse
from pathlib import Path

import breast_us_benchmark_final_v3 as bench

# Root for internal datasets in your NORMALIZED tree
INTERNAL_ROOT = Path("/workspace/breastdataset_NORMALIZED/breastdataset_NORMALIZED/internal")

# Your chosen models for ROI ablation
ROI_MODELS = [
    "swin_t",
    "convnext_tiny",
    "deit_tiny_distilled_patch16_224",
    "densenet121",
    "efficientnet_b0",
]


def load_keep_set(path: Path | None):
    if path is None:
        return None
    path = path.resolve()
    if not path.exists():
        raise FileNotFoundError(f"kept_images.txt not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        lines = [ln.strip() for ln in f if ln.strip()]
    keep = {str(Path(p).resolve().as_posix()) for p in lines}
    print(f"[ARTIFACT] Loaded {len(keep)} kept image paths from {path}")
    return keep


def is_internal_image(path_str: str) -> bool:
    """
    Return True if this image path is under the internal root
    (/workspace/breastdataset_NORMALIZED/.../internal).
    """
    p = Path(path_str).resolve()
    try:
        p.relative_to(INTERNAL_ROOT)
        return True
    except ValueError:
        return False


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Run ROI ABLATION (different ROI crops) on selected models "
            "with 5% artifact filtering on internal data."
        )
    )
    parser.add_argument(
        "--keep-list",
        type=Path,
        default=Path("/workspace/artifact_filter_lenient_005/kept_images.txt"),
        help="Path to kept_images.txt for the 5% artifact regime.",
    )
    parser.add_argument(
        "--num-folds",
        type=int,
        default=None,
        help="Override number of CV folds (default: use bench.NUM_FOLDS = 5).",
    )
    args = parser.parse_args()

    keep_set = load_keep_set(args.keep_list)

    # ---- Configure benchmark for ROI ABLATION ----
    # Use the predefined ROI ablation experiments (uniform/bottom/no_mask)
    bench.EXPERIMENTS = bench.ABLATION_EXPERIMENTS

    # Use your chosen model suite
    bench.MODEL_NAMES = ROI_MODELS

    # Keep default 5 folds unless overridden
    if args.num_folds is not None:
        bench.NUM_FOLDS = args.num_folds

    # Tag SCRIPT_VERSION so results are clearly labeled
    bench.SCRIPT_VERSION = f"{bench.SCRIPT_VERSION}_roi_ablation_5pct"

    print("\n====== ROI ABLATION with 5% artifact filtering ======")
    print(f"Models        : {bench.MODEL_NAMES}")
    print(f"Experiments   : {[e['name'] for e in bench.EXPERIMENTS]}")
    print(f"NUM_FOLDS     : {bench.NUM_FOLDS}")
    print(f"SCRIPT_VERSION: {bench.SCRIPT_VERSION}")
    print(f"[ARTIFACT] Using kept_images list with {len(keep_set)} entries.\n")

    # ---- Patch load_data_with_source to filter ONLY internal images ----
    orig_load = bench.load_data_with_source

    def patched_load_data_with_source(datasets):
        images, labels, sources = orig_load(datasets)

        if keep_set is None:
            return images, labels, sources

        filtered_imgs = []
        filtered_lbls = []
        filtered_srcs = []

        dropped = 0
        kept = 0

        for im, y, s in zip(images, labels, sources):
            im_resolved = str(Path(im).resolve().as_posix())

            # If this is an internal image, enforce the keep_set
            if is_internal_image(im_resolved):
                if im_resolved in keep_set:
                    filtered_imgs.append(im)
                    filtered_lbls.append(y)
                    filtered_srcs.append(s)
                    kept += 1
                else:
                    dropped += 1
            else:
                # External image: always keep (no artifact filtering on test data)
                filtered_imgs.append(im)
                filtered_lbls.append(y)
                filtered_srcs.append(s)
                kept += 1

        if kept == 0:
            raise RuntimeError(
                "[ARTIFACT] Filter removed all images! "
                "Check that kept_images.txt paths match dataset_config.json."
            )

        print(
            f"[ARTIFACT] After filtering: kept={kept}, dropped={dropped} "
            "(internal only; external test untouched)."
        )

        return filtered_imgs, filtered_lbls, filtered_srcs

    bench.load_data_with_source = patched_load_data_with_source

    # ---- Run the normal benchmark pipeline (now in ROI ablation mode) ----
    bench.validate_datasets()
    bench.run_experiments()


if __name__ == "__main__":
    main()
