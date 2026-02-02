#!/usr/bin/env python3
"""
Inference-only external AUROC + bootstrap CIs (no retraining).

Outputs per architecture:
  - fold_auc_mean, fold_auc_sd (across fold checkpoints)
  - fold_auc_boot_ci95_l/u (bootstrap CI on mean-of-folds AUROC)
  - ensemble_auc, ensemble_boot_ci95_l/u (bootstrap CI on ensemble AUROC)
  - n_external, n_pos, n_neg

Saved artifacts per architecture:
  - probs_by_fold.npy          shape (K, N)
  - ensemble_probs.npy         shape (N,)
  - labels.npy                 shape (N,)
  - bootstrap_auc_samples.npz  arrays: fold_mean_auc, ensemble_auc

Notes:
  - Bootstrap assumes independent test cases; if BUSI has duplicate patients,
    CIs may be optimistic. Keep that caveat in your supplement.
  - Uses the same preprocessing as the benchmark (PadSquareAndResize + normalization).
  - Fails fast on any image/ROI I/O failure to preserve label alignment.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.transforms import functional as F
from torchvision.transforms import InterpolationMode
from torchvision.models import (
    vgg16,
    resnet50,
    densenet121,
    convnext_tiny,
    mobilenet_v3_small,
    efficientnet_b0,
    swin_t,
    maxvit_t,
)
import timm
import cv2
from PIL import Image
from sklearn.metrics import roc_auc_score

# --------------------------------------------------------------------
# Dataset loading + preprocessing (copied from benchmark for inference)
# --------------------------------------------------------------------

KNOWN_MODELS = [
    "deit_tiny_distilled_patch16_224",
    "mobilenet_v3_small",
    "convnext_tiny",
    "efficientnet_b0",
    "densenet121",
    "regnety_008",
    "resnet50",
    "swin_t",
    "maxvit_t",
    "vgg16",
]


def load_json(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def resolve_external_datasets(
    external_split: str | None, external_pairs: List[str] | None
) -> Dict[str, str]:
    if external_pairs:
        out = {}
        for pair in external_pairs:
            if "=" not in pair:
                raise ValueError(
                    f"Invalid --external format: {pair} (expected NAME=PATH)"
                )
            name, path = pair.split("=", 1)
            out[name.strip()] = path.strip()
        return out

    if external_split:
        p = Path(external_split)
        if p.exists() and p.suffix.lower() in {".json"}:
            cfg = load_json(p)
            if "external_datasets" in cfg:
                return cfg["external_datasets"]
            if isinstance(cfg, dict):
                return cfg
        if external_split.lower() in {"config", "dataset_config.json", "default"}:
            cfg_path = Path("dataset_config.json")
            if cfg_path.exists():
                cfg = load_json(cfg_path)
                return cfg.get("external_datasets", {})
        raise ValueError(
            "Unable to resolve --external_split. Provide a JSON file or use --external NAME=PATH."
        )

    # Default: dataset_config.json if present
    cfg_path = Path("dataset_config.json")
    if cfg_path.exists():
        cfg = load_json(cfg_path)
        return cfg.get("external_datasets", {})

    raise ValueError(
        "No external datasets specified. Provide --external or --external_split."
    )


def find_mask_file(image_path: str) -> str | None:
    p = Path(image_path)
    mp = p.parent.parent / "masks" / f"{p.stem}_mask.png"
    return str(mp) if mp.exists() else None


def process_mask_to_binary(mask):
    if mask is None:
        return None
    if len(mask.shape) == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    return binary


def extract_roi(image_path, mask_path, border_ratio=0.0, border_type="uniform"):
    try:
        img = cv2.imread(image_path)
        if border_type == "no_mask":
            return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        mask = cv2.imread(mask_path)
        if img is None or mask is None:
            return None
        binary = process_mask_to_binary(mask)
        if binary is None:
            return None
        contours, _ = cv2.findContours(
            binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            return None
        x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
        if border_ratio > 0:
            if border_type == "uniform":
                bw, bh = int(w * border_ratio), int(h * border_ratio)
                x = max(0, x - bw)
                y = max(0, y - bh)
                w = min(img.shape[1] - x, w + 2 * bw)
                h = min(img.shape[0] - y, h + 2 * bh)
            elif border_type == "bottom_only":
                bh = int(h * border_ratio)
                h = min(img.shape[0] - y, h + bh)
        roi = img[y : y + h, x : x + w]
        return Image.fromarray(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
    except Exception:
        return None


class PadSquareAndResize:
    def __init__(self, target_size=224, small_threshold=150, fill=0):
        self.target_size = target_size
        self.small_threshold = small_threshold
        self.fill = fill

    def __call__(self, img):
        w, h = img.size
        if w != h:
            m = max(w, h)
            img = F.pad(
                img,
                (
                    (m - w) // 2,
                    (m - h) // 2,
                    m - w - (m - w) // 2,
                    m - h - (m - h) // 2,
                ),
                fill=self.fill,
                padding_mode="constant",
            )
            w = h = m
        if w < self.small_threshold:
            pad = (self.target_size - w) // 2
            img = F.pad(img, pad, fill=self.fill)
            if img.size[0] != self.target_size:
                img = transforms.Resize(
                    (self.target_size, self.target_size),
                    interpolation=InterpolationMode.BICUBIC,
                )(img)
        elif w != self.target_size:
            img = transforms.Resize(
                (self.target_size, self.target_size),
                interpolation=InterpolationMode.BICUBIC,
            )(img)
        return img


class BUSDataset(Dataset):
    def __init__(self, images, labels, transform=None, border_ratio=0.0, border_type="uniform"):
        self.images = images
        self.labels = labels
        self.transform = transform
        self.border_ratio = border_ratio
        self.border_type = border_type
        self.cache = {}

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]
        key = f"{img_path}_{self.border_ratio}_{self.border_type}"

        img = self.cache.get(key)
        if img is None:
            try:
                if self.border_type == "no_mask":
                    img = Image.open(img_path).convert("RGB")
                else:
                    mp = find_mask_file(img_path)
                    if not mp:
                        raise RuntimeError(f"Missing ROI mask for {img_path}")
                    img = extract_roi(img_path, mp, self.border_ratio, self.border_type)
                    if img is None:
                        raise RuntimeError(f"ROI extraction failed for {img_path}")
                self.cache[key] = img
            except Exception as e:
                raise RuntimeError(f"Failed to load/ROI image: {img_path}") from e

        if self.transform:
            img = self.transform(img)
        return img, label


def build_eval_transform(model_name: str):
    target = 299 if "inception" in model_name.lower() else 224
    padresize = PadSquareAndResize(target_size=target, small_threshold=150, fill=0)
    return transforms.Compose(
        [
            padresize,
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )


def load_data_with_source(datasets: Dict[str, str]) -> Tuple[List[str], List[int], List[str]]:
    images, labels, sources = [], [], []
    for name, path in datasets.items():
        if not os.path.exists(path):
            print(f"WARNING: {name} not found at {path}")
            continue
        for label_name, label_val in [("benign", 0), ("malignant", 1)]:
            img_dir = os.path.join(path, label_name, "images")
            if not os.path.exists(img_dir):
                img_dir = os.path.join(path, label_name)
            if not os.path.exists(img_dir):
                print(f"WARNING: Missing directory: {img_dir}")
                continue
            for fname in sorted(os.listdir(img_dir)):
                if fname.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")):
                    images.append(os.path.join(img_dir, fname))
                    labels.append(label_val)
                    sources.append(name)
    return images, labels, sources


# -----------------------------
# Models
# -----------------------------

def get_model(name: str) -> nn.Module:
    if name == "vgg16":
        m = vgg16(weights=None)
        m.classifier[6] = nn.Linear(m.classifier[6].in_features, 2)
    elif name == "resnet50":
        m = resnet50(weights=None)
        m.fc = nn.Linear(m.fc.in_features, 2)
    elif name == "densenet121":
        m = densenet121(weights=None)
        m.classifier = nn.Linear(m.classifier.in_features, 2)
    elif name == "regnety_008":
        m = timm.create_model("regnety_008", pretrained=False)
        m.head.fc = nn.Linear(m.head.fc.in_features, 2)
    elif name == "convnext_tiny":
        m = convnext_tiny(weights=None)
        m.classifier[2] = nn.Linear(m.classifier[2].in_features, 2)
    elif name == "mobilenet_v3_small":
        m = mobilenet_v3_small(weights=None)
        m.classifier[3] = nn.Linear(m.classifier[3].in_features, 2)
    elif name == "efficientnet_b0":
        m = efficientnet_b0(weights=None)
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, 2)
    elif name == "deit_tiny_distilled_patch16_224":
        m = timm.create_model("deit_tiny_distilled_patch16_224", pretrained=False)
        if hasattr(m, "head") and isinstance(m.head, nn.Linear):
            m.head = nn.Linear(m.head.in_features, 2)
        if hasattr(m, "head_dist") and isinstance(m.head_dist, nn.Linear):
            m.head_dist = nn.Linear(m.head_dist.in_features, 2)
    elif name == "swin_t":
        m = swin_t(weights=None)
        m.head = nn.Linear(m.head.in_features, 2)
    elif name == "maxvit_t":
        m = maxvit_t(weights=None)
        m.classifier[5] = nn.Linear(m.classifier[5].in_features, 2)
    else:
        raise ValueError(f"Unknown model name: {name}")
    return m


# -----------------------------
# Bootstrap CI
# -----------------------------

def bootstrap_auc_indices(
    y_true: np.ndarray, n_boot: int, seed: int
) -> List[np.ndarray]:
    rng = np.random.RandomState(seed)
    y_true = np.asarray(y_true)
    pos = np.where(y_true == 1)[0]
    neg = np.where(y_true == 0)[0]
    n_pos, n_neg = len(pos), len(neg)
    if n_pos == 0 or n_neg == 0:
        return []
    idx_list = []
    for _ in range(n_boot):
        idx = np.concatenate(
            [rng.choice(pos, n_pos, True), rng.choice(neg, n_neg, True)]
        )
        idx_list.append(idx)
    return idx_list


def percentile_ci(samples: np.ndarray, alpha: float = 0.95) -> Tuple[float, float]:
    lo = float(np.percentile(samples, (1 - alpha) / 2 * 100))
    hi = float(np.percentile(samples, (1 + alpha) / 2 * 100))
    return lo, hi


# -----------------------------
# Checkpoint discovery
# -----------------------------

def parse_checkpoint_name(name: str, model_names: List[str]):
    """
    Parse: <prefix>_<model>_<exp>_f<fold>_best.pth
    Returns (prefix, model, exp, fold) or None.
    """
    # Prefer longest model names first to avoid partial matches.
    model_names = sorted(model_names, key=len, reverse=True)
    for model in model_names:
        pat = rf"^(?P<prefix>.+)_{re.escape(model)}_(?P<exp>.+)_f(?P<fold>\d+)_best\.pth$"
        m = re.match(pat, name)
        if m:
            gd = m.groupdict()
            return gd["prefix"], model, gd["exp"], int(gd["fold"])
    return None


def discover_checkpoints(
    ckpt_dir: Path,
    model_names: List[str],
    glob_pattern: str | None,
    experiment_filter: str | None,
):
    files = list(ckpt_dir.glob(glob_pattern or "*.pth"))
    parsed = []
    for p in files:
        res = parse_checkpoint_name(p.name, model_names)
        if res is None:
            continue
        prefix, model, exp, fold = res
        if experiment_filter and exp != experiment_filter:
            continue
        parsed.append((p, prefix, model, exp, fold))
    return parsed


# -----------------------------
# Inference
# -----------------------------

def run_fold_inference(
    model_name: str,
    ckpt_path: Path,
    loader: DataLoader,
    device: torch.device,
) -> np.ndarray:
    model = get_model(model_name)
    state = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(state)
    model = model.to(device)
    model.eval()

    probs_all = []
    with torch.no_grad():
        for x, _ in loader:
            x = x.to(device, non_blocking=True)
            logits = model(x)
            probs = torch.softmax(logits, dim=1)[:, 1]
            probs_all.append(probs.cpu().numpy())
    probs = np.concatenate(probs_all, axis=0)

    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return probs


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Inference-only external AUROC + bootstrap CIs (no retraining)."
    )
    ap.add_argument("--checkpoints_root", required=True, type=Path, help="Root folder with checkpoints.")
    ap.add_argument("--external_split", default=None, help="JSON path or 'config' to use dataset_config.json.")
    ap.add_argument("--external", nargs="*", default=None, help="Override external datasets as NAME=PATH pairs.")
    ap.add_argument("--architectures", default=None, help="Comma-separated list of model names.")
    ap.add_argument("--expected_folds", type=int, default=5, help="Expected number of folds per architecture.")
    ap.add_argument("--bootstrap_iters", type=int, default=2000, help="Bootstrap iterations.")
    ap.add_argument("--seed", type=int, default=42, help="Bootstrap RNG seed.")
    ap.add_argument("--device", default=None, help="cuda/cpu override.")
    ap.add_argument("--out_dir", default="external_bootstrap_out", help="Output directory.")
    ap.add_argument("--out_csv", default="external_auc_bootstrap_summary.csv", help="Summary CSV filename.")
    ap.add_argument("--out_xlsx", default=None, help="Optional XLSX summary filename.")
    ap.add_argument("--glob_pattern", default=None, help="Glob pattern to filter checkpoints (e.g., '*_baseline_f*_best.pth').")
    ap.add_argument("--experiment", default=None, help="Filter checkpoints by experiment name (between model and fold).")
    ap.add_argument("--border_ratio", type=float, default=0.0, help="ROI border ratio (default 0.0).")
    ap.add_argument("--border_type", default="uniform", choices=["uniform", "bottom_only", "no_mask"], help="ROI border type.")
    ap.add_argument("--batch_size", type=int, default=32, help="Batch size for inference.")
    ap.add_argument("--num_workers", type=int, default=0, help="DataLoader workers (default 0).")
    ap.add_argument("--expected_n", type=int, default=None, help="Expected number of external samples (optional).")
    ap.add_argument("--strict", action="store_true", help="Error on missing folds or mismatched N.")
    args = ap.parse_args()

    ckpt_root = args.checkpoints_root.expanduser().resolve()
    if not ckpt_root.exists():
        raise FileNotFoundError(f"Checkpoints root not found: {ckpt_root}")

    device = torch.device(args.device) if args.device else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print(f"Device: {device}")

    # Resolve external datasets and load once for consistent ordering.
    external_datasets = resolve_external_datasets(args.external_split, args.external)
    images, labels, _ = load_data_with_source(external_datasets)
    if not images:
        raise RuntimeError("No external images found. Check dataset paths.")
    labels = np.array(labels, dtype=np.int64)

    n_external = int(labels.shape[0])
    n_pos = int((labels == 1).sum())
    n_neg = int((labels == 0).sum())
    print(f"External N={n_external} (pos={n_pos}, neg={n_neg})")

    if args.expected_n is not None and n_external != args.expected_n:
        msg = f"Expected N={args.expected_n}, got N={n_external}"
        if args.strict:
            raise RuntimeError(msg)
        print("WARNING:", msg)

    # Determine architectures to process.
    if args.architectures:
        model_names = [m.strip() for m in args.architectures.split(",") if m.strip()]
    else:
        model_names = KNOWN_MODELS

    parsed = discover_checkpoints(
        ckpt_root, model_names, args.glob_pattern, args.experiment
    )
    if not parsed:
        raise RuntimeError("No checkpoints matched. Use --glob_pattern or --experiment.")

    # Group by (model, prefix, exp)
    by_model: Dict[str, Dict[Tuple[str, str], List[Tuple[Path, int]]]] = {}
    for p, prefix, model, exp, fold in parsed:
        by_model.setdefault(model, {}).setdefault((prefix, exp), []).append((p, fold))

    # Resolve unique (prefix, exp) per model
    resolved = {}
    for model, groups in by_model.items():
        if len(groups) > 1:
            msg = (
                f"Multiple checkpoint groups for model '{model}': "
                + ", ".join([f"{k[0]}|{k[1]}" for k in groups.keys()])
            )
            if args.strict:
                raise RuntimeError(msg)
            print("WARNING:", msg, "Use --glob_pattern or --experiment to disambiguate. Skipping.")
            continue
        (prefix, exp), items = next(iter(groups.items()))
        resolved[model] = (prefix, exp, items)

    if not resolved:
        raise RuntimeError("No unambiguous models resolved. Use --glob_pattern or --experiment.")

    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = []

    # Pre-generate bootstrap indices
    boot_indices = bootstrap_auc_indices(labels, args.bootstrap_iters, args.seed)
    if not boot_indices:
        raise RuntimeError("Bootstrap indices could not be generated (need both classes).")

    for model, (prefix, exp, items) in sorted(resolved.items()):
        folds = sorted(items, key=lambda x: x[1])
        fold_ids = [f for _, f in folds]
        fold_id_set = set(fold_ids)
        expected_set_1 = set(range(1, args.expected_folds + 1))
        expected_set_0 = set(range(args.expected_folds))

        if len(fold_id_set) != len(fold_ids):
            msg = f"{model}: duplicate fold IDs found ({fold_ids})"
            if args.strict:
                raise RuntimeError(msg)
            print("WARNING:", msg, "Skipping.")
            continue

        if fold_id_set not in (expected_set_1, expected_set_0):
            msg = f"{model}: unexpected fold IDs {sorted(fold_id_set)} (expected {sorted(expected_set_1)} or {sorted(expected_set_0)})"
            if args.strict:
                raise RuntimeError(msg)
            print("WARNING:", msg, "Skipping.")
            continue

        if len(folds) != args.expected_folds:
            msg = f"{model}: expected {args.expected_folds} folds, found {len(folds)} ({fold_ids})"
            if args.strict:
                raise RuntimeError(msg)
            print("WARNING:", msg, "Skipping.")
            continue

        # Build model-specific dataset/loader
        tf = build_eval_transform(model)
        ds = BUSDataset(images, labels.tolist(), tf, args.border_ratio, args.border_type)
        loader = DataLoader(
            ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=(device.type == "cuda"),
        )

        # Inference for each fold
        probs_by_fold = []
        fold_aucs = []
        for ckpt_path, fold in folds:
            probs = run_fold_inference(model, ckpt_path, loader, device)
            if probs.shape[0] != n_external:
                raise RuntimeError(
                    f"{model} fold {fold}: probs length {probs.shape[0]} != {n_external}"
                )
            auc = float(roc_auc_score(labels, probs))
            probs_by_fold.append(probs)
            fold_aucs.append(auc)

        probs_by_fold = np.stack(probs_by_fold, axis=0)  # (K, N)
        ensemble_probs = probs_by_fold.mean(axis=0)

        # Point estimates
        fold_auc_mean = float(np.mean(fold_aucs))
        fold_auc_sd = float(np.std(fold_aucs, ddof=1)) if len(fold_aucs) > 1 else 0.0
        ensemble_auc = float(roc_auc_score(labels, ensemble_probs))

        # Bootstrap CIs
        boot_fold_mean = []
        boot_ensemble = []
        for idx in boot_indices:
            aucs_b = [roc_auc_score(labels[idx], probs_by_fold[i, idx]) for i in range(probs_by_fold.shape[0])]
            boot_fold_mean.append(float(np.mean(aucs_b)))
            boot_ensemble.append(float(roc_auc_score(labels[idx], ensemble_probs[idx])))

        boot_fold_mean = np.array(boot_fold_mean, dtype=np.float32)
        boot_ensemble = np.array(boot_ensemble, dtype=np.float32)

        fold_ci_l, fold_ci_u = percentile_ci(boot_fold_mean, alpha=0.95)
        ens_ci_l, ens_ci_u = percentile_ci(boot_ensemble, alpha=0.95)

        # Save artifacts
        model_dir = out_dir / model
        model_dir.mkdir(parents=True, exist_ok=True)
        np.save(model_dir / "probs_by_fold.npy", probs_by_fold)
        np.save(model_dir / "ensemble_probs.npy", ensemble_probs)
        np.save(model_dir / "labels.npy", labels)
        np.savez(
            model_dir / "bootstrap_auc_samples.npz",
            fold_mean_auc=boot_fold_mean,
            ensemble_auc=boot_ensemble,
        )

        summary_rows.append(
            {
                "architecture": model,
                "n_external": n_external,
                "n_pos": n_pos,
                "n_neg": n_neg,
                "fold_auc_mean": fold_auc_mean,
                "fold_auc_sd": fold_auc_sd,
                "fold_auc_boot_ci95_l": fold_ci_l,
                "fold_auc_boot_ci95_u": fold_ci_u,
                "ensemble_auc": ensemble_auc,
                "ensemble_boot_ci95_l": ens_ci_l,
                "ensemble_boot_ci95_u": ens_ci_u,
                "expected_folds": args.expected_folds,
                "prefix": prefix,
                "experiment": exp,
                "border_ratio": args.border_ratio,
                "border_type": args.border_type,
            }
        )

        print(
            f"{model}: fold_mean={fold_auc_mean:.4f} (sd={fold_auc_sd:.4f}) "
            f"CI[{fold_ci_l:.4f},{fold_ci_u:.4f}] | "
            f"ensemble={ensemble_auc:.4f} CI[{ens_ci_l:.4f},{ens_ci_u:.4f}]"
        )

    if not summary_rows:
        raise RuntimeError("No models processed. Check filters and checkpoints.")

    import pandas as pd

    df = pd.DataFrame(summary_rows).sort_values("fold_auc_mean", ascending=False)
    out_csv = Path(args.out_csv)
    if not out_csv.is_absolute():
        out_csv = out_dir / out_csv
    df.to_csv(out_csv, index=False)
    print(f"Saved summary: {out_csv}")

    if args.out_xlsx:
        out_xlsx = Path(args.out_xlsx)
        if not out_xlsx.is_absolute():
            out_xlsx = out_dir / out_xlsx
        df.to_excel(out_xlsx, index=False)
        print(f"Saved XLSX: {out_xlsx}")


if __name__ == "__main__":
    main()
