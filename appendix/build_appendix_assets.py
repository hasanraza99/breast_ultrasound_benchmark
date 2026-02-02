
import os
import json
import math
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

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
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedGroupKFold

try:
    from scipy.stats import ttest_rel
except Exception:
    ttest_rel = None

try:
    import psutil
except Exception:
    psutil = None

ROOT = Path.cwd()
OUT_DIR = ROOT / "appendix"
OUT_DIR.mkdir(parents=True, exist_ok=True)

ARCH_NAMES = {
    "swin_t": "Swin-T",
    "convnext_tiny": "ConvNeXt-T",
    "deit_tiny_distilled_patch16_224": "DeiT-T",
    "vgg16": "VGG-16",
    "regnety_008": "RegNetY",
    "efficientnet_b0": "EffNet-B0",
    "densenet121": "DenseNet",
    "maxvit_t": "MaxViT-T",
    "mobilenet_v3_small": "MobNetV3",
    "resnet50": "ResNet-50",
}

ARCH_ORDER = [
    "Swin-T",
    "ConvNeXt-T",
    "DeiT-T",
    "VGG-16",
    "RegNetY",
    "EffNet-B0",
    "DenseNet",
    "MaxViT-T",
    "MobNetV3",
    "ResNet-50",
]

BENCHMARK_MODELS = [
    "vgg16",
    "resnet50",
    "densenet121",
    "regnety_008",
    "convnext_tiny",
    "mobilenet_v3_small",
    "efficientnet_b0",
    "deit_tiny_distilled_patch16_224",
    "swin_t",
    "maxvit_t",
]

SEED = 42
NUM_FOLDS = 5
BATCH_SIZE = 32


def resolve_dataset_path(path_str: str) -> Path:
    p = Path(path_str)
    if p.exists():
        return p
    s = str(path_str)
    if "/workspace/" in s:
        rel = s.split("/workspace/")[-1]
        candidate = ROOT / Path(rel)
        if candidate.exists():
            return candidate
    candidate = Path(s.replace("/", os.sep))
    if candidate.exists():
        return candidate
    return Path(s)


def load_config(config_path: Path):
    if not config_path.exists():
        raise FileNotFoundError(f"Missing config: {config_path}")
    cfg = json.loads(config_path.read_text())
    internal = {k: resolve_dataset_path(v) for k, v in cfg["internal_datasets"].items()}
    external = {k: resolve_dataset_path(v) for k, v in cfg["external_datasets"].items()}
    source_sizes = cfg.get("source_sizes", {})
    return internal, external, source_sizes


def find_mask_file(image_path: str):
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
            return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
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
        original_idx = idx
        max_retries = 10
        img = None
        label = self.labels[idx]

        for _ in range(max_retries):
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
                        if mp:
                            img = extract_roi(img_path, mp, self.border_ratio, self.border_type)
                    if img is not None:
                        self.cache[key] = img
                        break
                except Exception:
                    pass
            else:
                break
            idx = (idx + 1) % len(self)

        if img is None:
            print(
                f"WARNING: Failed to load image after {max_retries} attempts (original idx={original_idx})"
            )
            img = Image.new("RGB", (224, 224), color=0)

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

def load_data_with_source_internal(datasets):
    images, labels, sources = [], [], []
    for name, path in datasets.items():
        if not path.exists():
            print(f"WARNING: {name} not found at {path}")
            continue
        for label_name, label_val in [("benign", 0), ("malignant", 1)]:
            img_dir = path / label_name / "images"
            if not img_dir.exists():
                print(f"WARNING: Missing directory: {img_dir}")
                continue
            for fname in os.listdir(img_dir):
                if fname.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")):
                    images.append(str(img_dir / fname))
                    labels.append(label_val)
                    sources.append(name)
    return images, labels, sources


def load_data_with_source_external_sorted(datasets):
    images, labels, sources = [], [], []
    for name, path in datasets.items():
        if not path.exists():
            print(f"WARNING: {name} not found at {path}")
            continue
        for label_name, label_val in [("benign", 0), ("malignant", 1)]:
            img_dir = path / label_name / "images"
            if not img_dir.exists():
                img_dir = path / label_name
            if not img_dir.exists():
                print(f"WARNING: Missing directory: {img_dir}")
                continue
            for fname in sorted(os.listdir(img_dir)):
                if fname.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")):
                    images.append(str(img_dir / fname))
                    labels.append(label_val)
                    sources.append(name)
    return images, labels, sources


BUS_BRA_PATIENT_CSV = ROOT / "bus-bra_data.csv"
BUS_UCLM_PATIENT_CSV = ROOT / "bus_uclm_patient_ids.csv"


def load_patient_id_maps():
    bus_bra_map, bus_uclm_map = {}, {}
    if BUS_BRA_PATIENT_CSV.exists():
        try:
            df = pd.read_csv(BUS_BRA_PATIENT_CSV)
            if "ID" in df.columns and "Case" in df.columns:
                bus_bra_map = {
                    str(row["ID"]).strip(): str(row["Case"]).strip()
                    for _, row in df.iterrows()
                }
        except Exception:
            pass
    if BUS_UCLM_PATIENT_CSV.exists():
        try:
            df = pd.read_csv(BUS_UCLM_PATIENT_CSV)
            if "file_name" in df.columns and "patient_id" in df.columns:
                bus_uclm_map = {
                    str(row["file_name"]).strip(): str(row["patient_id"]).strip()
                    for _, row in df.iterrows()
                }
        except Exception:
            pass
    return bus_bra_map, bus_uclm_map


def build_patient_groups(images, sources, bus_bra_map, bus_uclm_map):
    groups = []
    for img_path, src in zip(images, sources):
        p = Path(img_path)
        if src == "BUS-BRA" and bus_bra_map:
            key = p.stem
            case = bus_bra_map.get(key)
            if case is not None:
                group = f"{src}:case{case}"
            else:
                group = f"{src}:{key}"
        elif src == "BUS-UCLM" and bus_uclm_map:
            key = p.name
            pid = bus_uclm_map.get(key)
            if pid is not None:
                group = f"{src}:{pid}"
            else:
                group = f"{src}:{p.stem}"
        else:
            group = f"{src}:{p.stem}"
        groups.append(group)
    return np.array(groups)


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
        raise ValueError(f"Unknown model: {name}")
    return m


def load_checkpoint(model: nn.Module, ckpt_path: Path):
    state = torch.load(ckpt_path, map_location="cpu")
    if isinstance(state, dict) and all(k.startswith("module.") for k in state.keys()):
        state = {k.replace("module.", ""): v for k, v in state.items()}
    try:
        model.load_state_dict(state, strict=True)
    except Exception:
        model.load_state_dict(state, strict=False)
    return model


def compute_youden_metrics(y_true, y_prob):
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    if len(np.unique(y_true)) < 2:
        return np.nan, np.nan, np.nan, np.nan
    fpr, tpr, thr = roc_curve(y_true, y_prob)
    j = tpr - fpr
    idx = np.argmax(j)
    sens = tpr[idx]
    spec = 1.0 - fpr[idx]
    bal = 0.5 * (sens + spec)
    return sens, spec, bal, thr[idx]


def compute_auc(y_true, y_prob):
    y_true = np.asarray(y_true)
    if len(np.unique(y_true)) < 2:
        return np.nan
    return roc_auc_score(y_true, y_prob)


def run_bus_uc_sensitivity(internal_datasets):
    print("Running BUS_UC sensitivity analysis (inference)...")
    train_images, train_labels, train_sources = load_data_with_source_internal(internal_datasets)
    train_labels = np.array(train_labels)

    bus_bra_map, bus_uclm_map = load_patient_id_maps()
    train_groups = build_patient_groups(train_images, train_sources, bus_bra_map, bus_uclm_map)

    sgkf = StratifiedGroupKFold(n_splits=NUM_FOLDS, shuffle=True, random_state=SEED)
    splits = list(sgkf.split(train_images, train_labels, groups=train_groups))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    results_rows = []
    per_fold_cache = {}

    for model_name in BENCHMARK_MODELS:
        print(f"  Model: {model_name}")
        fold_metrics_full = []
        fold_metrics_no_uc = []
        fold_metrics_no_uclm_ml = []

        for fold_idx, (_, va_idx) in enumerate(splits, 1):
            ckpt = ROOT / "checkpoints" / f"v2.0_final_artifacts_5pct_{model_name}_baseline_f{fold_idx}_best.pth"
            if not ckpt.exists():
                print(f"    WARNING: checkpoint missing: {ckpt}")
                continue

            X_val = [train_images[i] for i in va_idx]
            y_val = [train_labels[i] for i in va_idx]
            s_val = [train_sources[i] for i in va_idx]

            cache_key = (model_name, fold_idx)
            if cache_key in per_fold_cache:
                probs, labels, paths, sources = per_fold_cache[cache_key]
            else:
                val_tf = build_eval_transform(model_name)
                val_ds = BUSDataset(X_val, y_val, val_tf, border_ratio=0.0, border_type="uniform")
                val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

                model = get_model(model_name)
                model = load_checkpoint(model, ckpt)
                model.to(device)
                model.eval()

                probs_list = []
                labels_list = []
                with torch.no_grad():
                    for xb, yb in val_loader:
                        xb = xb.to(device)
                        out = model(xb)
                        if out.shape[-1] == 2:
                            pb = torch.softmax(out, dim=1)[:, 1]
                        else:
                            pb = torch.sigmoid(out.squeeze())
                        probs_list.append(pb.cpu().numpy())
                        labels_list.append(yb.numpy())

                probs = np.concatenate(probs_list)
                labels = np.concatenate(labels_list)
                paths = np.array(X_val)
                sources = np.array(s_val)
                per_fold_cache[cache_key] = (probs, labels, paths, sources)

            auc_full = compute_auc(labels, probs)
            sens_full, spec_full, bal_full, _ = compute_youden_metrics(labels, probs)
            fold_metrics_full.append((auc_full, sens_full, spec_full, bal_full))

            mask_no_uc = sources != "BUS_UC"
            labels_no_uc = labels[mask_no_uc]
            probs_no_uc = probs[mask_no_uc]
            auc_no_uc = compute_auc(labels_no_uc, probs_no_uc)
            sens_no_uc, spec_no_uc, bal_no_uc, _ = compute_youden_metrics(labels_no_uc, probs_no_uc)
            fold_metrics_no_uc.append((auc_no_uc, sens_no_uc, spec_no_uc, bal_no_uc))

            ml_mask = np.array(["_lesion" in Path(p).stem for p in paths])
            mask_no_ml = ~((sources == "BUS-UCLM") & ml_mask)
            labels_no_ml = labels[mask_no_ml]
            probs_no_ml = probs[mask_no_ml]
            auc_no_ml = compute_auc(labels_no_ml, probs_no_ml)
            sens_no_ml, spec_no_ml, bal_no_ml, _ = compute_youden_metrics(labels_no_ml, probs_no_ml)
            fold_metrics_no_uclm_ml.append((auc_no_ml, sens_no_ml, spec_no_ml, bal_no_ml))

        def summarize(metrics):
            arr = np.array(metrics, dtype=float)
            return np.nanmean(arr, axis=0), np.nanstd(arr, axis=0, ddof=1)

        mean_full, sd_full = summarize(fold_metrics_full)
        mean_no_uc, sd_no_uc = summarize(fold_metrics_no_uc)
        mean_no_ml, sd_no_ml = summarize(fold_metrics_no_uclm_ml)

        for metric_idx, metric_name in enumerate(["AUC", "Sensitivity", "Specificity", "Balanced_Accuracy"]):
            results_rows.append(
                {
                    "Model": model_name,
                    "Architecture": ARCH_NAMES.get(model_name, model_name),
                    "Metric": metric_name,
                    "Full_mean": mean_full[metric_idx],
                    "Full_sd": sd_full[metric_idx],
                    "No_BUS_UC_mean": mean_no_uc[metric_idx],
                    "No_BUS_UC_sd": sd_no_uc[metric_idx],
                    "Delta_No_BUS_UC": mean_no_uc[metric_idx] - mean_full[metric_idx],
                    "No_UCLM_ML_mean": mean_no_ml[metric_idx],
                    "No_UCLM_ML_sd": sd_no_ml[metric_idx],
                    "Delta_No_UCLM_ML": mean_no_ml[metric_idx] - mean_full[metric_idx],
                }
            )

    df = pd.DataFrame(results_rows)
    df.to_csv(OUT_DIR / "table_A1_bus_uc_sensitivity_metrics.csv", index=False)

    df_auc = df[df["Metric"] == "AUC"].copy()
    df_auc = df_auc.sort_values("Full_mean", ascending=False)
    fig, ax = plt.subplots(figsize=(9, 6))
    y = np.arange(len(df_auc))
    ax.hlines(y, df_auc["Full_mean"], df_auc["No_BUS_UC_mean"], color="#888", linewidth=1)
    ax.scatter(df_auc["Full_mean"], y, label="Full validation", color="#1f77b4")
    ax.scatter(df_auc["No_BUS_UC_mean"], y, label="Validation excluding BUS_UC", color="#ff7f0e")
    ax.set_yticks(y)
    ax.set_yticklabels(df_auc["Architecture"])
    ax.set_xlabel("Validation AUROC")
    ax.set_title("Appendix A: BUS_UC Sensitivity (Validation AUROC)")
    ax.legend(loc="lower right", frameon=False)
    ax.grid(axis="x", alpha=0.2)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig_A1_bus_uc_sensitivity_auc.png", dpi=300)
    plt.close(fig)

    return per_fold_cache, df

def list_multilesion_images(dataset_path: Path):
    entries = []
    for label_name in ["benign", "malignant"]:
        img_dir = dataset_path / label_name / "images"
        if not img_dir.exists():
            img_dir = dataset_path / label_name
        if not img_dir.exists():
            continue
        for fname in sorted(os.listdir(img_dir)):
            if "_lesion" in fname.lower():
                stem = Path(fname).stem
                base = stem.split("_lesion")[0]
                entries.append((base, fname, label_name))
    return entries


def run_multi_lesion_analysis(external_datasets, per_fold_cache):
    print("Running multi-lesion analysis...")
    rows = []
    for ds_name in ["BUS-UCLM", "BUSI"]:
        path = external_datasets.get(ds_name) if ds_name in external_datasets else None
        if ds_name == "BUS-UCLM":
            path = resolve_dataset_path(
                str(ROOT / "breastdataset_NORMALIZED" / "breastdataset_NORMALIZED" / "internal" / "BUS-UCLM")
            )
        if path is None or not Path(path).exists():
            continue
        entries = list_multilesion_images(Path(path))
        grouped = defaultdict(list)
        for base, fname, _ in entries:
            grouped[base].append(fname)
        for base, fnames in grouped.items():
            rows.append(
                {
                    "Dataset": ds_name,
                    "Source_ID": base,
                    "Lesion_Files": ";".join(sorted(fnames)),
                    "Num_Lesions": len(fnames),
                }
            )

    df_list = pd.DataFrame(rows)
    df_list.to_csv(OUT_DIR / "table_B1_multi_lesion_list.csv", index=False)

    ext_images, ext_labels, ext_sources = load_data_with_source_external_sorted(external_datasets)
    ext_sources = np.array(ext_sources)
    ext_paths = np.array(ext_images)
    ml_mask_ext = np.array(["_lesion" in Path(p).stem for p in ext_paths])

    impacts = []
    ext_boot_dir = ROOT / "results" / "external_bootstrap_artifacts_5pct"
    for model_dir in sorted([p for p in ext_boot_dir.iterdir() if p.is_dir()]):
        model_name = model_dir.name
        probs_by_fold = np.load(model_dir / "probs_by_fold.npy")
        labels = np.load(model_dir / "labels.npy")
        ensemble_probs = np.load(model_dir / "ensemble_probs.npy")

        busi_mask = ext_sources == "BUSI"
        busi_mask_no_ml = busi_mask & (~ml_mask_ext)

        fold_aucs = []
        fold_aucs_no_ml = []
        for fold_idx in range(probs_by_fold.shape[0]):
            p = probs_by_fold[fold_idx]
            auc_full = compute_auc(labels[busi_mask], p[busi_mask])
            auc_no_ml = compute_auc(labels[busi_mask_no_ml], p[busi_mask_no_ml])
            fold_aucs.append(auc_full)
            fold_aucs_no_ml.append(auc_no_ml)

        mean_full = np.nanmean(fold_aucs)
        sd_full = np.nanstd(fold_aucs, ddof=1)
        mean_no_ml = np.nanmean(fold_aucs_no_ml)
        sd_no_ml = np.nanstd(fold_aucs_no_ml, ddof=1)

        ens_full = compute_auc(labels[busi_mask], ensemble_probs[busi_mask])
        ens_no_ml = compute_auc(labels[busi_mask_no_ml], ensemble_probs[busi_mask_no_ml])

        impacts.append(
            {
                "Model": model_name,
                "Architecture": ARCH_NAMES.get(model_name, model_name),
                "Dataset": "BUSI",
                "Fold_AUC_mean": mean_full,
                "Fold_AUC_sd": sd_full,
                "Fold_AUC_no_ml_mean": mean_no_ml,
                "Fold_AUC_no_ml_sd": sd_no_ml,
                "Delta_Fold_AUC": mean_no_ml - mean_full,
                "Ensemble_AUC": ens_full,
                "Ensemble_AUC_no_ml": ens_no_ml,
                "Delta_Ensemble_AUC": ens_no_ml - ens_full,
            }
        )

    df_impact = pd.DataFrame(impacts)
    df_impact.to_csv(OUT_DIR / "table_B2_multi_lesion_impact_external.csv", index=False)

    df_plot = df_impact.copy().sort_values("Fold_AUC_mean", ascending=False)
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(df_plot["Architecture"], df_plot["Delta_Fold_AUC"], color="#9467bd")
    ax.axvline(0, color="#333", linewidth=1)
    ax.set_xlabel("Delta AUROC (Exclude multi-lesion BUSI images)")
    ax.set_title("Appendix B: BUSI Multi-Lesion Impact (Fold-Mean AUROC)")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig_B1_multi_lesion_impact_busi.png", dpi=300)
    plt.close(fig)

    uclm_impacts = []
    for (model_name, fold_idx), (probs, labels, paths, sources) in per_fold_cache.items():
        sources = np.array(sources)
        paths = np.array(paths)
        uclm_mask = sources == "BUS-UCLM"
        if not np.any(uclm_mask):
            continue
        ml_mask = np.array(["_lesion" in Path(p).stem for p in paths])
        uclm_no_ml = uclm_mask & (~ml_mask)
        auc_full = compute_auc(labels[uclm_mask], probs[uclm_mask])
        auc_no_ml = compute_auc(labels[uclm_no_ml], probs[uclm_no_ml])
        uclm_impacts.append(
            {
                "Model": model_name,
                "Architecture": ARCH_NAMES.get(model_name, model_name),
                "Fold": fold_idx,
                "AUC_full": auc_full,
                "AUC_no_ml": auc_no_ml,
                "Delta": auc_no_ml - auc_full,
            }
        )

    if uclm_impacts:
        df_uclm = pd.DataFrame(uclm_impacts)
        agg = df_uclm.groupby(["Model", "Architecture"], as_index=False).agg(
            AUC_full_mean=("AUC_full", "mean"),
            AUC_full_sd=("AUC_full", "std"),
            AUC_no_ml_mean=("AUC_no_ml", "mean"),
            AUC_no_ml_sd=("AUC_no_ml", "std"),
            Delta=("Delta", "mean"),
        )
        agg.to_csv(OUT_DIR / "table_B3_multi_lesion_impact_uclm_internal.csv", index=False)

    return df_list, df_impact


def run_roi_ablation_table():
    print("Building ROI ablation table...")
    roi_path = ROOT / "results" / "summary_v2.0_final_roi_ablation_5pct.csv"
    df = pd.read_csv(roi_path)

    exp_order = [
        "baseline",
        "uniform_5",
        "uniform_7_5",
        "uniform_10",
        "uniform_15",
        "uniform_20",
        "bottom_5",
        "bottom_10",
        "bottom_15",
        "no_mask",
    ]

    df["Experiment"] = pd.Categorical(df["Experiment"], categories=exp_order, ordered=True)

    agg = df.groupby(["Model", "Experiment"], as_index=False).agg(
        AUC_mean=("Test_AUC", "mean"),
        AUC_sd=("Test_AUC", "std"),
    )
    agg["Architecture"] = agg["Model"].map(ARCH_NAMES).fillna(agg["Model"])

    sig_rows = []
    for model_name in df["Model"].unique():
        df_m = df[df["Model"] == model_name]
        base = df_m[df_m["Experiment"] == "baseline"]["Test_AUC"].values
        for exp in exp_order:
            df_e = df_m[df_m["Experiment"] == exp]["Test_AUC"].values
            pval = np.nan
            if len(base) == len(df_e) and len(base) > 1:
                if ttest_rel is not None:
                    try:
                        pval = ttest_rel(df_e, base).pvalue
                    except Exception:
                        pval = np.nan
            sig = "ns"
            if np.isfinite(pval):
                if pval < 0.001:
                    sig = "***"
                elif pval < 0.01:
                    sig = "**"
                elif pval < 0.05:
                    sig = "*"
            sig_rows.append(
                {
                    "Model": model_name,
                    "Experiment": exp,
                    "p_value_vs_baseline": pval,
                    "sig": sig,
                }
            )

    sig_df = pd.DataFrame(sig_rows)
    full = agg.merge(sig_df, on=["Model", "Experiment"], how="left")
    full = full.sort_values(["Architecture", "Experiment"])
    full.to_csv(OUT_DIR / "table_C1_roi_ablation_auc.csv", index=False)

    pivot = full.pivot(index="Architecture", columns="Experiment", values="AUC_mean")
    pivot_sd = full.pivot(index="Architecture", columns="Experiment", values="AUC_sd")
    pivot_sig = full.pivot(index="Architecture", columns="Experiment", values="sig")

    arch_order = [a for a in ARCH_ORDER if a in pivot.index]
    pivot = pivot.loc[arch_order]
    pivot_sd = pivot_sd.loc[arch_order]
    pivot_sig = pivot_sig.loc[arch_order]

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.axis("off")
    table_data = []
    for arch in pivot.index:
        row = []
        for exp in exp_order:
            val = pivot.loc[arch, exp]
            sd = pivot_sd.loc[arch, exp]
            sig = pivot_sig.loc[arch, exp]
            if pd.isna(val):
                row.append("")
            else:
                row.append(f"{val:.3f} +/-{sd:.3f} {sig}")
        table_data.append(row)

    col_labels = exp_order
    row_labels = list(pivot.index)
    tbl = ax.table(cellText=table_data, rowLabels=row_labels, colLabels=col_labels, loc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(7)
    tbl.scale(1, 1.3)
    ax.set_title("Appendix C: ROI Ablation AUROC (mean +/- SD, vs baseline)")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig_C1_roi_ablation_table.png", dpi=300)
    plt.close(fig)

    return full

def run_bootstrap_figures():
    print("Generating bootstrap distribution figures...")
    ext_boot_dir = ROOT / "results" / "external_bootstrap_artifacts_5pct"
    arch_dirs = [p for p in ext_boot_dir.iterdir() if p.is_dir()]

    boot_data = {}
    for p in arch_dirs:
        npz = p / "bootstrap_auc_samples.npz"
        if not npz.exists():
            continue
        with np.load(npz) as data:
            boot_data[p.name] = {
                "fold_mean": data["fold_mean_auc"],
                "ensemble": data["ensemble_auc"],
            }

    names_sorted = [k for k in ARCH_NAMES.keys() if k in boot_data]
    n = len(names_sorted)
    cols = 2
    rows = int(math.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(12, 3 * rows), sharex=False)
    axes = np.atleast_1d(axes).flatten()

    for idx, model_name in enumerate(names_sorted):
        ax = axes[idx]
        data = boot_data[model_name]
        ax.hist(data["fold_mean"], bins=30, alpha=0.6, label="Fold-mean", color="#1f77b4")
        ax.hist(data["ensemble"], bins=30, alpha=0.6, label="Ensemble", color="#ff7f0e")
        ax.axvline(np.mean(data["fold_mean"]), color="#1f77b4", linestyle="--", linewidth=1)
        ax.axvline(np.mean(data["ensemble"]), color="#ff7f0e", linestyle="--", linewidth=1)
        ax.set_title(ARCH_NAMES.get(model_name, model_name))
        ax.set_xlabel("AUROC")
        ax.set_ylabel("Count")
        if idx == 0:
            ax.legend(frameon=False)
    for j in range(idx + 1, len(axes)):
        axes[j].axis("off")

    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig_D1_bootstrap_histograms.png", dpi=300)
    plt.close(fig)

    rows = []
    for model_name in names_sorted:
        data = boot_data[model_name]
        fold = data["fold_mean"]
        ens = data["ensemble"]
        rows.append(
            {
                "Model": model_name,
                "Architecture": ARCH_NAMES.get(model_name, model_name),
                "Fold_mean": np.mean(fold),
                "Fold_ci_l": np.quantile(fold, 0.025),
                "Fold_ci_u": np.quantile(fold, 0.975),
                "Ensemble_mean": np.mean(ens),
                "Ensemble_ci_l": np.quantile(ens, 0.025),
                "Ensemble_ci_u": np.quantile(ens, 0.975),
            }
        )
    df = pd.DataFrame(rows)
    df = df.sort_values("Fold_mean", ascending=False)
    df.to_csv(OUT_DIR / "table_D1_bootstrap_summary.csv", index=False)

    fig, ax = plt.subplots(figsize=(9, 6))
    y = np.arange(len(df))
    ax.errorbar(
        df["Fold_mean"],
        y + 0.12,
        xerr=[df["Fold_mean"] - df["Fold_ci_l"], df["Fold_ci_u"] - df["Fold_mean"]],
        fmt="o",
        color="#1f77b4",
        label="Fold-mean",
    )
    ax.errorbar(
        df["Ensemble_mean"],
        y - 0.12,
        xerr=[df["Ensemble_mean"] - df["Ensemble_ci_l"], df["Ensemble_ci_u"] - df["Ensemble_mean"]],
        fmt="o",
        color="#ff7f0e",
        label="Ensemble",
    )
    ax.set_yticks(y)
    ax.set_yticklabels(df["Architecture"])
    ax.set_xlabel("AUROC (bootstrap 95% CI)")
    ax.set_title("Appendix D: Ensemble vs Fold-Mean AUROC")
    ax.legend(frameon=False)
    ax.grid(axis="x", alpha=0.2)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig_D2_ensemble_vs_foldmean.png", dpi=300)
    plt.close(fig)


def run_external_breakdown(external_datasets):
    print("Computing per-dataset external breakdown...")
    ext_images, ext_labels, ext_sources = load_data_with_source_external_sorted(external_datasets)
    ext_labels = np.array(ext_labels)
    ext_sources = np.array(ext_sources)

    ext_boot_dir = ROOT / "results" / "external_bootstrap_artifacts_5pct"
    rows = []
    for model_dir in sorted([p for p in ext_boot_dir.iterdir() if p.is_dir()]):
        model_name = model_dir.name
        probs_by_fold = np.load(model_dir / "probs_by_fold.npy")
        labels = np.load(model_dir / "labels.npy")
        ensemble_probs = np.load(model_dir / "ensemble_probs.npy")

        for ds in ["BUSI", "QAMEBI"]:
            mask = ext_sources == ds
            if not np.any(mask):
                continue
            fold_aucs = []
            for fold_idx in range(probs_by_fold.shape[0]):
                p = probs_by_fold[fold_idx]
                fold_aucs.append(compute_auc(labels[mask], p[mask]))
            rows.append(
                {
                    "Model": model_name,
                    "Architecture": ARCH_NAMES.get(model_name, model_name),
                    "Dataset": ds,
                    "n": int(mask.sum()),
                    "Fold_AUC_mean": np.nanmean(fold_aucs),
                    "Fold_AUC_sd": np.nanstd(fold_aucs, ddof=1),
                    "Ensemble_AUC": compute_auc(labels[mask], ensemble_probs[mask]),
                }
            )

    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "table_E1_external_by_dataset.csv", index=False)

    df_plot = df.copy()
    df_plot = df_plot.sort_values(["Dataset", "Fold_AUC_mean"], ascending=[True, False])
    fig, ax = plt.subplots(figsize=(9, 6))
    for i, ds in enumerate(["BUSI", "QAMEBI"]):
        sub = df_plot[df_plot["Dataset"] == ds]
        y = np.arange(len(sub)) + (i * 0.15)
        ax.errorbar(sub["Fold_AUC_mean"], y, xerr=sub["Fold_AUC_sd"], fmt="o", label=ds)
    ax.set_yticks(np.arange(len(sub)))
    ax.set_yticklabels(sub["Architecture"])
    ax.set_xlabel("External AUROC (mean +/- SD across folds)")
    ax.set_title("Appendix E: External Performance by Dataset")
    ax.legend(frameon=False)
    ax.grid(axis="x", alpha=0.2)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig_E1_external_by_dataset.png", dpi=300)
    plt.close(fig)


def write_ai_usage_doc():
    content = """# Generative AI Usage Documentation

This appendix documents the use of generative AI tools in the dissertation workflow.

## Tools Used
- ChatGPT 5.2
- Claude Opus 4.5
- Gemini 3
- Claude Code
- ChatGPT Codex

## Learning Assistance Prompts (replace with your actual prompts)
- [PROMPT 1] "..."
- [PROMPT 2] "..."

## Code Refinement Examples (replace with your actual prompts)
- [PROMPT] "..."
- [BEFORE/AFTER] Describe what was changed and why.

## Research Summarisation Outputs (replace with your actual prompts)
- [PROMPT] "Summarise paper X for Y"
- [SUMMARY] "..."

> NOTE: Replace placeholder text with the specific prompts and outputs used in your project.
"""
    (OUT_DIR / "generative_ai_usage.md").write_text(content, encoding="utf-8")


def write_reproducibility_docs():
    hyperparams = [
        ("SEED", SEED),
        ("NUM_FOLDS", NUM_FOLDS),
        ("BATCH_SIZE", BATCH_SIZE),
        ("NUM_EPOCHS", 30),
        ("PATIENCE", 7),
        ("LEARNING_RATE", 1e-4),
        ("WEIGHT_DECAY", 1e-4),
        ("THRESHOLD_STRATEGY", "youden"),
        ("F_BETA", 1.0),
        ("COST_FP", 1.0),
        ("COST_FN", 1.0),
        ("FIXED_THRESHOLD", 0.5),
        ("ROI_BORDER_RATIO", 0.0),
        ("ROI_BORDER_TYPE", "uniform"),
    ]
    hp_df = pd.DataFrame(hyperparams, columns=["Parameter", "Value"])
    hp_df.to_csv(OUT_DIR / "hyperparameters.csv", index=False)

    pkg_rows = []
    def add_pkg(name, module):
        try:
            ver = module.__version__
        except Exception:
            ver = "unknown"
        pkg_rows.append({"Package": name, "Version": ver})

    import sklearn
    import PIL
    import matplotlib
    import torchvision

    add_pkg("python", sys.version.split()[0])
    add_pkg("numpy", np)
    add_pkg("pandas", pd)
    add_pkg("scikit-learn", sklearn)
    add_pkg("opencv-python", cv2)
    add_pkg("Pillow", PIL)
    add_pkg("matplotlib", matplotlib)
    add_pkg("timm", timm)
    add_pkg("torch", torch)
    add_pkg("torchvision", torchvision)

    pd.DataFrame(pkg_rows).to_csv(OUT_DIR / "software_versions.csv", index=False)

    hw_lines = []
    hw_lines.append(f"OS: {os.name}")
    hw_lines.append(f"Platform: {os.sys.platform}")
    if psutil is not None:
        hw_lines.append(f"CPU cores (logical): {psutil.cpu_count(logical=True)}")
        hw_lines.append(f"CPU cores (physical): {psutil.cpu_count(logical=False)}")
        mem = psutil.virtual_memory()
        hw_lines.append(f"RAM total (GB): {mem.total / (1024**3):.2f}")
    else:
        hw_lines.append(f"CPU cores (logical): {os.cpu_count()}")
    if torch.cuda.is_available():
        hw_lines.append("CUDA: available")
        hw_lines.append(f"GPU: {torch.cuda.get_device_name(0)}")
    else:
        hw_lines.append("CUDA: not available")

    (OUT_DIR / "hardware_specs.txt").write_text("\n".join(hw_lines), encoding="utf-8")

    repro_md = """# Reproducibility Information

This appendix documents the software, hardware, and training hyperparameters used to generate the results.

## Hardware
See `hardware_specs.txt`.

## Software Versions
See `software_versions.csv`.

## Hyperparameters
See `hyperparameters.csv`.
"""
    (OUT_DIR / "reproducibility.md").write_text(repro_md, encoding="utf-8")


def main():
    internal, external, _ = load_config(ROOT / "dataset_config.json")
    per_fold_cache, _ = run_bus_uc_sensitivity(internal)
    run_multi_lesion_analysis(external, per_fold_cache)
    run_roi_ablation_table()
    run_bootstrap_figures()
    run_external_breakdown(external)
    write_ai_usage_doc()
    write_reproducibility_docs()


if __name__ == "__main__":
    main()
