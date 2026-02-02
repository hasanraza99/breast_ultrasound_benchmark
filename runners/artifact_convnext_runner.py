#!/usr/bin/env python3
import argparse
from pathlib import Path

import breast_us_benchmark_final_v3 as bench


# Root for internal datasets in your NORMALIZED tree
INTERNAL_ROOT = Path("/workspace/breastdataset_NORMALIZED/breastdataset_NORMALIZED/internal")


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
            "Run ConvNeXt-Tiny breast US experiment under different artifact filters "
            "using breast_us_benchmark_final_v3."
        )
    )
    parser.add_argument(
        "--filter-mode",
        choices=[
            "none",
            "lenient_005",
            "moderate_002",
            "strict_any",
            "egregious_010",
        ],
        default="none",
        help=(
            "Artifact filter regime:\n"
            "  none          = no artifact filtering\n"
            "  lenient_005   = drop frac_lesion_covered > 0.05\n"
            "  moderate_002  = drop frac_lesion_covered > 0.02\n"
            "  strict_any    = drop any non-zero overlap above min_inter_px\n"
            "  egregious_010 = drop frac_lesion_covered > 0.10\n"
            "  (min_inter_px threshold is baked into each kept_images list)"
        ),
    )
    parser.add_argument(
        "--keep-list",
        type=Path,
        default=None,
        help="Optional explicit path to kept_images.txt. If omitted, a default is inferred from filter-mode.",
    )
    parser.add_argument(
        "--num-folds",
        type=int,
        default=3,
        help="Number of CV folds (default: 3 for this artifact experiment).",
    )
    args = parser.parse_args()

    # Default kept_images.txt locations for each regime
    default_keep_paths = {
        "lenient_005": Path("/workspace/artifact_filter_lenient_005/kept_images.txt"),
        "moderate_002": Path("/workspace/artifact_filter_moderate_002/kept_images.txt"),
        "strict_any": Path("/workspace/artifact_filter_strict_any/kept_images.txt"),
        "egregious_010": Path("/workspace/artifact_filter_010/kept_images.txt"),
    }

    if args.filter_mode == "none":
        keep_list_path = None
    else:
        keep_list_path = args.keep_list or default_keep_paths[args.filter_mode]

    keep_set = load_keep_set(keep_list_path) if keep_list_path is not None else None

    # ---- Configure benchmark for this experiment ----
    # Single model: ConvNeXt-Tiny
    bench.MODEL_NAMES = ["convnext_tiny"]

    # Strict lesion-ROI cropping, 0% border, but name encodes filter regime
    bench.EXPERIMENTS = [
        {"ratio": 0.0, "type": "uniform", "name": f"baseline_convnext_artifacts_{args.filter_mode}"}
    ]

    # Use 3 folds for this side experiment
    bench.NUM_FOLDS = args.num_folds

    # Tag SCRIPT_VERSION so results are separate from ResNet runs
    bench.SCRIPT_VERSION = f"{bench.SCRIPT_VERSION}_convnext_tiny_artifacts_{args.filter_mode}"

    print("\n====== Artifact-aware ConvNeXt-Tiny experiment ======")
    print(f"Filter mode    : {args.filter_mode}")
    print(f"Model          : convnext_tiny")
    print(f"NUM_FOLDS      : {bench.NUM_FOLDS}")
    print(f"SCRIPT_VERSION : {bench.SCRIPT_VERSION}")
    if keep_set is None:
        print("[ARTIFACT] No artifact filtering (all internal images kept).")
    else:
        print(f"[ARTIFACT] Using kept_images list with {len(keep_set)} entries.")

    # ---- Patch load_data_with_source to filter ONLY internal images ----
    orig_load = bench.load_data_with_source

    def patched_load_data_with_source(datasets):
        images, labels, sources = orig_load(datasets)

        if keep_set is None:
            # No filtering: return as-is
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
                # External image: always keep
                filtered_imgs.append(im)
                filtered_lbls.append(y)
                filtered_srcs.append(s)
                kept += 1

        if kept == 0:
            raise RuntimeError(
                "[ARTIFACT] Filter removed all images! "
                "Check that your kept_images.txt paths match the dataset_config."
            )

        print(
            f"[ARTIFACT] After filtering in mode '{args.filter_mode}': "
            f"kept={kept}, dropped={dropped}"
        )

        return filtered_imgs, filtered_lbls, filtered_srcs

    bench.load_data_with_source = patched_load_data_with_source

    # ---- Run the normal benchmark pipeline ----
    bench.validate_datasets()
    bench.run_experiments()


if __name__ == "__main__":
    main()
