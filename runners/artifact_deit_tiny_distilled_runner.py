#!/usr/bin/env python3
import argparse
from pathlib import Path

import breast_us_benchmark_final_v3 as bench


_SCRIPT_DIR = Path(__file__).resolve().parent
_BASE_SCRIPT_VERSION = bench.SCRIPT_VERSION
_BASE_LOAD_DATA = bench.load_data_with_source
_DEFAULT_DATA_ROOT = "/workspace/breastdataset_NORMALIZED/breastdataset_NORMALIZED"
_LOCAL_DATA_ROOT = _SCRIPT_DIR / "breastdataset_NORMALIZED" / "breastdataset_NORMALIZED"
_USE_LOCAL_ROOT = _LOCAL_DATA_ROOT.exists()


def remap_path(path_str: str) -> str:
    if _USE_LOCAL_ROOT and path_str.startswith(_DEFAULT_DATA_ROOT):
        rel = path_str[len(_DEFAULT_DATA_ROOT) :].lstrip("/\\")
        return str((_LOCAL_DATA_ROOT / rel).resolve())
    return path_str


def remap_dataset_paths(dsets: dict) -> dict:
    return {name: remap_path(path) for name, path in dsets.items()}


# Root for internal datasets in your NORMALIZED tree
_DATA_ROOT = _LOCAL_DATA_ROOT if _USE_LOCAL_ROOT else Path(_DEFAULT_DATA_ROOT)
INTERNAL_ROOT = _DATA_ROOT / "internal"

# Prefer local repo copy; fall back to /workspace if that's where your data lives.
KEEP_LIST_BASE = _SCRIPT_DIR
_DEFAULT_KEEP_BASE = Path("/workspace")


FILTER_MODES = [
    "none",
    "moderate_002",
    "lenient_005",
    "egregious_010",
    "egregious_020",
    "strict_any",
]

DEFAULT_KEEP_PATHS = {
    "lenient_005": KEEP_LIST_BASE / "artifact_filter_lenient_005" / "kept_images.txt",
    "moderate_002": KEEP_LIST_BASE / "artifact_filter_moderate_002" / "kept_images.txt",
    "strict_any": KEEP_LIST_BASE / "artifact_filter_strict_any" / "kept_images.txt",
    "egregious_010": KEEP_LIST_BASE / "artifact_filter_010" / "kept_images.txt",
    "egregious_020": KEEP_LIST_BASE / "artifact_filter_020" / "kept_images.txt",
}

FALLBACK_KEEP_PATHS = {
    "lenient_005": _DEFAULT_KEEP_BASE / "artifact_filter_lenient_005" / "kept_images.txt",
    "moderate_002": _DEFAULT_KEEP_BASE / "artifact_filter_moderate_002" / "kept_images.txt",
    "strict_any": _DEFAULT_KEEP_BASE / "artifact_filter_strict_any" / "kept_images.txt",
    "egregious_010": _DEFAULT_KEEP_BASE / "artifact_filter_010" / "kept_images.txt",
    "egregious_020": _DEFAULT_KEEP_BASE / "artifact_filter_020" / "kept_images.txt",
}


def load_keep_set(path: Path | None):
    if path is None:
        return None
    path = path.resolve()
    if not path.exists():
        raise FileNotFoundError(f"kept_images.txt not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        lines = [ln.strip() for ln in f if ln.strip()]
    keep = {str(Path(remap_path(p)).resolve().as_posix()) for p in lines}
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


def resolve_keep_list(filter_mode: str) -> Path | None:
    if filter_mode == "none":
        return None
    path = DEFAULT_KEEP_PATHS[filter_mode]
    if path.exists():
        return path
    return FALLBACK_KEEP_PATHS[filter_mode]


def run_one(filter_mode: str, keep_list_path: Path | None, num_folds: int):
    keep_set = load_keep_set(keep_list_path) if keep_list_path else None

    # Remap dataset roots when running on a local Windows checkout.
    if _USE_LOCAL_ROOT:
        bench.INTERNAL_DATASETS = remap_dataset_paths(bench.INTERNAL_DATASETS)
        bench.EXTERNAL_DATASETS = remap_dataset_paths(bench.EXTERNAL_DATASETS)

    # ---- Configure benchmark for this experiment ----
    # Single model: DeiT-Tiny distilled
    bench.MODEL_NAMES = ["deit_tiny_distilled_patch16_224"]

    # Strict lesion-ROI cropping, 0% border, but name encodes filter regime
    bench.EXPERIMENTS = [
        {
            "ratio": 0.0,
            "type": "uniform",
            "name": f"baseline_deit_tiny_distilled_artifacts_{filter_mode}",
        }
    ]

    # Use 3 folds for this side experiment (matching other artifact runs)
    bench.NUM_FOLDS = num_folds

    # Tag SCRIPT_VERSION so results are separate from CNN runs
    bench.SCRIPT_VERSION = (
        f"{_BASE_SCRIPT_VERSION}_deit_tiny_distilled_artifacts_{filter_mode}"
    )

    print("\n====== Artifact-aware DeiT-Tiny Distilled experiment ======")
    print(f"Filter mode    : {filter_mode}")
    print(f"Model          : deit_tiny_distilled_patch16_224")
    print(f"NUM_FOLDS      : {bench.NUM_FOLDS}")
    print(f"SCRIPT_VERSION : {bench.SCRIPT_VERSION}")
    if keep_set is None:
        print("[ARTIFACT] No artifact filtering (all internal images kept).")
    else:
        print(f"[ARTIFACT] Using kept_images list with {len(keep_set)} entries.")

    # Windows-safe: avoid multiprocessing workers that re-import torch/CUDA
    _orig_dl = bench.DataLoader

    def _dl(*args, **kwargs):
        kwargs["num_workers"] = 0
        kwargs["persistent_workers"] = False
        return _orig_dl(*args, **kwargs)

    bench.DataLoader = _dl

    # ---- Patch load_data_with_source to filter ONLY internal images ----
    orig_load = _BASE_LOAD_DATA

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
            f"[ARTIFACT] After filtering in mode '{filter_mode}': "
            f"kept={kept}, dropped={dropped}"
        )

        return filtered_imgs, filtered_lbls, filtered_srcs

    bench.load_data_with_source = patched_load_data_with_source

    # ---- Run the normal benchmark pipeline ----
    bench.validate_datasets()
    bench.run_experiments()
    # Restore base loader to avoid stacking patches across modes
    bench.load_data_with_source = _BASE_LOAD_DATA


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Run DeiT-Tiny distilled artifact experiments using breast_us_benchmark_final_v3."
        )
    )
    parser.add_argument(
        "--filter-mode",
        choices=FILTER_MODES + ["all"],
        default="all",
        help=(
            "Artifact filter regime. Use 'all' to run: "
            "none, moderate_002, lenient_005, egregious_010, egregious_020, strict_any."
        ),
    )
    parser.add_argument(
        "--keep-list",
        type=Path,
        default=None,
        help="Optional explicit path to kept_images.txt (only used when filter-mode != all).",
    )
    parser.add_argument(
        "--num-folds",
        type=int,
        default=3,
        help="Number of CV folds (default: 3 for this artifact experiment).",
    )
    args = parser.parse_args()

    if args.filter_mode == "all":
        for mode in FILTER_MODES:
            keep_path = resolve_keep_list(mode)
            run_one(mode, keep_path, args.num_folds)
    else:
        keep_path = args.keep_list or resolve_keep_list(args.filter_mode)
        run_one(args.filter_mode, keep_path, args.num_folds)
