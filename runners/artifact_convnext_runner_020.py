#!/usr/bin/env python3
from pathlib import Path

import breast_us_benchmark_final_v3 as bench


_SCRIPT_DIR = Path(__file__).resolve().parent
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

# Fixed 0.20 artifact regime (ConvNeXt-Tiny only)
FILTER_MODE = "egregious_020"
# Prefer local repo copy; fall back to /workspace if that's where your data lives.
KEEP_LIST_PATH = _SCRIPT_DIR / "artifact_filter_020" / "kept_images.txt"


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


def main():
    keep_set = load_keep_set(KEEP_LIST_PATH)

    # Remap dataset roots when running on a local Windows checkout.
    if _USE_LOCAL_ROOT:
        bench.INTERNAL_DATASETS = remap_dataset_paths(bench.INTERNAL_DATASETS)
        bench.EXTERNAL_DATASETS = remap_dataset_paths(bench.EXTERNAL_DATASETS)

    # ---- Configure benchmark for this experiment ----
    # Single model: ConvNeXt-Tiny
    bench.MODEL_NAMES = ["convnext_tiny"]

    # Strict lesion-ROI cropping, 0% border, but name encodes filter regime
    bench.EXPERIMENTS = [
        {"ratio": 0.0, "type": "uniform", "name": f"baseline_convnext_artifacts_{FILTER_MODE}"}
    ]

    # Use 3 folds for this side experiment (matching other artifact runs)
    bench.NUM_FOLDS = 3

    # Tag SCRIPT_VERSION so results are separate from ResNet runs
    bench.SCRIPT_VERSION = f"{bench.SCRIPT_VERSION}_convnext_tiny_artifacts_{FILTER_MODE}"

    print("\n====== Artifact-aware ConvNeXt-Tiny experiment ======")
    print(f"Filter mode    : {FILTER_MODE}")
    print(f"Model          : convnext_tiny")
    print(f"NUM_FOLDS      : {bench.NUM_FOLDS}")
    print(f"SCRIPT_VERSION : {bench.SCRIPT_VERSION}")
    print(f"[ARTIFACT] Using kept_images list with {len(keep_set)} entries.")

    # Windows-safe: avoid multiprocessing workers that re-import torch/CUDA
    _orig_dl = bench.DataLoader

    def _dl(*args, **kwargs):
        kwargs["num_workers"] = 0
        kwargs["persistent_workers"] = False
        return _orig_dl(*args, **kwargs)

    bench.DataLoader = _dl

    # ---- Patch load_data_with_source to filter ONLY internal images ----
    orig_load = bench.load_data_with_source

    def patched_load_data_with_source(datasets):
        images, labels, sources = orig_load(datasets)

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
            f"[ARTIFACT] After filtering in mode '{FILTER_MODE}': "
            f"kept={kept}, dropped={dropped}"
        )

        return filtered_imgs, filtered_lbls, filtered_srcs

    bench.load_data_with_source = patched_load_data_with_source

    # ---- Run the normal benchmark pipeline ----
    bench.validate_datasets()
    bench.run_experiments()


if __name__ == "__main__":
    main()
