import subprocess, sys

subprocess.call(
    [
        sys.executable,
        "-m",
        "pip",
        "install",
        "numpy>=1.24.0,<2.0.0",
        "pandas",
        "scikit-learn",
        "opencv-python",
        "timm",
        "scipy",
        "matplotlib",
        "pillow",
        "--quiet",
    ]
)

import os, json, random, warnings, gc, math
from time import perf_counter
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader, ConcatDataset, WeightedRandomSampler
from torch.cuda.amp import autocast, GradScaler
from PIL import Image
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
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold
from sklearn.metrics import (
    confusion_matrix,
    roc_auc_score,
    matthews_corrcoef,
    cohen_kappa_score,
    brier_score_loss,
    f1_score,
    roc_curve,
    precision_recall_curve,
    average_precision_score,
)
from scipy.stats import beta as beta_dist
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ==== VERSION & CONFIG ====
SCRIPT_VERSION = "v2.0_final"  # Increment when making significant changes
warnings.filterwarnings("ignore")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 Script Version: {SCRIPT_VERSION}")
print(f"Device: {device}")


def load_config(config_path="dataset_config.json"):
    """
    Load dataset configuration from JSON file.
    Falls back to hardcoded defaults if config not found.
    """
    if os.path.exists(config_path):
        try:
            with open(config_path, "r") as f:
                config = json.load(f)
            print(f"✓ Loaded config from {config_path}")
            return config
        except Exception as e:
            print(f"⚠️ Error loading config: {e}")
            print("Using hardcoded defaults...")
            return None
    else:
        print(f"ℹ️ Config file not found: {config_path}")
        print(
            "Using hardcoded defaults. Create 'dataset_config.json' to customize paths."
        )
        return None


# Load configuration
config = load_config()

if config:
    INTERNAL_DATASETS = config["internal_datasets"]
    EXTERNAL_DATASETS = config["external_datasets"]
    SOURCE_SIZES = config["source_sizes"]
else:
    # Fallback to hardcoded defaults (FIXED: consistent naming)
    INTERNAL_DATASETS = {
        "BUS-BRA": "/workspace/Datasets/internal datasets/BUS-BRA",
        "BUS_UC": "/workspace/Datasets/internal datasets/BUS_UC",
        "UDIAT": "/workspace/Datasets/internal datasets/UDIAT",
        "BUS-UCLM": "/workspace/Datasets/internal datasets/BUS-UCLM",  # FIXED: was "UCLM"
        "USG": "/workspace/Datasets/internal datasets/USG",
    }
    EXTERNAL_DATASETS = {
        "BUSI": "/workspace/Datasets/external datasets/BUSI",
        "QAMEBI": "/workspace/Datasets/external datasets/QAMEBI",
    }
    SOURCE_SIZES = {
        "BUS-BRA": 1875,
        "BUS_UC": 811,
        "BUS-UCLM": 264,  # FIXED: was "BUS-UCLM" with key "UCLM" in datasets
        "USG": 252,
        "UDIAT": 163,
    }

# Validate that keys match between INTERNAL_DATASETS and SOURCE_SIZES
missing_sizes = set(INTERNAL_DATASETS.keys()) - set(SOURCE_SIZES.keys())
if missing_sizes:
    print(f"⚠️ WARNING: These datasets lack source size definitions: {missing_sizes}")
    print(f"   Will use default size (500). Add them to SOURCE_SIZES or config file.")

# ===== EXPERIMENT CONFIGURATION =====
# Change this to switch between modes: "ablation", "benchmark", or "both"
RUN_MODE = "benchmark"  # ← CHANGE THIS TO SWITCH MODES

# ROI Ablation experiments - test different border expansion strategies
ABLATION_EXPERIMENTS = [
    {"ratio": 0.0, "type": "uniform", "name": "baseline"},
    {"ratio": 0.05, "type": "uniform", "name": "uniform_5"},
    {"ratio": 0.075, "type": "uniform", "name": "uniform_7_5"},
    {"ratio": 0.10, "type": "uniform", "name": "uniform_10"},
    {"ratio": 0.15, "type": "uniform", "name": "uniform_15"},
    {"ratio": 0.20, "type": "uniform", "name": "uniform_20"},
    {"ratio": 0.05, "type": "bottom_only", "name": "bottom_5"},
    {"ratio": 0.10, "type": "bottom_only", "name": "bottom_10"},
    {"ratio": 0.15, "type": "bottom_only", "name": "bottom_15"},
    {"ratio": 0.0, "type": "no_mask", "name": "no_mask"},
]

# Baseline benchmark - just evaluate models on strict 0% crop
BENCHMARK_EXPERIMENTS = [
    {"ratio": 0.0, "type": "uniform", "name": "baseline"},
]

# Model configurations
ABLATION_MODELS = [
    "mobilenet_v3_small",  # Fast, lightweight
    "convnext_tiny",  # Modern architecture
    "efficientnet_b0",  # Balanced efficiency
]

BENCHMARK_MODELS = [
    # Classical CNNs
    "vgg16",
    # Residual/Modern ConvNets
    "resnet50",
    "densenet121",
    "regnety_008",
    "convnext_tiny",
    # Efficient CNNs
    "mobilenet_v3_small",
    "efficientnet_b0",
    # Transformers and Hybrids
    "deit_tiny_distilled_patch16_224",
    "swin_t",
    "maxvit_t",
]

# Select experiments and models based on run mode
if RUN_MODE == "ablation":
    EXPERIMENTS = ABLATION_EXPERIMENTS
    MODEL_NAMES = ABLATION_MODELS
    print(f"\n🔬 Running ROI ABLATION mode")
    print(f"   Testing border expansion strategies on {len(MODEL_NAMES)} models")
elif RUN_MODE == "benchmark":
    EXPERIMENTS = BENCHMARK_EXPERIMENTS
    MODEL_NAMES = BENCHMARK_MODELS
    print(f"\n📊 Running BASELINE BENCHMARK mode")
    print(f"   Evaluating {len(MODEL_NAMES)} models on strict 0% crop")
elif RUN_MODE == "both":
    EXPERIMENTS = ABLATION_EXPERIMENTS + BENCHMARK_EXPERIMENTS
    MODEL_NAMES = list(set(ABLATION_MODELS + BENCHMARK_MODELS))
    print(f"\n🔬📊 Running BOTH modes sequentially")
    print(f"   Phase 1: ROI ablation on {len(ABLATION_MODELS)} models")
    print(f"   Phase 2: Benchmark on {len(BENCHMARK_MODELS)} models")
else:
    raise ValueError(
        f"Invalid RUN_MODE: {RUN_MODE}. Must be 'ablation', 'benchmark', or 'both'"
    )

print(f"   Experiments: {len(EXPERIMENTS)}")
print(f"   Models: {len(MODEL_NAMES)}")
print(
    f"   Total training runs: {len(EXPERIMENTS) * len(MODEL_NAMES) * 5} (5-fold CV)\n"
)

# Training parameters
SEED, BATCH_SIZE, NUM_EPOCHS, PATIENCE = 42, 32, 30, 7
LEARNING_RATE, WEIGHT_DECAY = 1e-4, 1e-4
NUM_FOLDS = 5

USE_SOURCE_SAMPLER = True

THRESHOLD_STRATEGY = "youden"
F_BETA = 1.0
COST_FP, COST_FN = 1.0, 1.0
FIXED_THRESHOLD = 0.5

SKIP_COMPLETED = True
SAVE_CURVES = True


# ==== UTILITY ====
def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def clear_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


set_seed()


def done_marker_path(model_name, exp_name, fold):
    Path("checkpoints/done").mkdir(parents=True, exist_ok=True)
    return Path(
        f"checkpoints/done/{SCRIPT_VERSION}_{model_name}_{exp_name}_f{fold}.done"
    )


def fold_is_done(model_name, exp_name, fold):
    best = Path(
        f"checkpoints/{SCRIPT_VERSION}_{model_name}_{exp_name}_f{fold}_best.pth"
    )
    done = done_marker_path(model_name, exp_name, fold)
    return best.exists() and done.exists()


def load_previous_results():
    """Load previous results from versioned file to prevent mixing old/new runs."""
    try:
        results_file = f"results/raw_results_{SCRIPT_VERSION}.json"
        with open(results_file, "r") as f:
            prev = json.load(f)
        idx = {(r["model"], r["experiment"], r["fold"]): r for r in prev}
        print(f"✓ Loaded {len(prev)} previous results from {results_file}")
        return prev, idx
    except Exception:
        print(f"ℹ️ No previous results found for {SCRIPT_VERSION}, starting fresh")
        return [], {}


# ==== DATA LOADING ====
def find_mask_file(image_path: str):
    """
    Find corresponding mask file for an image.
    Expected: images in [dataset]/[benign|malignant]/images/
              masks in [dataset]/masks/
    Mask naming: image001.png -> image001_mask.png
    """
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
        contours, _ = cv2.findContours(
            binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
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


def load_data_with_source(datasets):
    """
    Load images with expected directory structure:
    [dataset]/
        benign/
            images/
                image001.png
                image002.png
        malignant/
            images/
                image001.png
        masks/  (optional)
            image001_mask.png
    """
    images, labels, sources = [], [], []
    for name, path in datasets.items():
        if not os.path.exists(path):
            print(f"⚠️ {name} not found at {path}")
            continue
        for label_name, label_val in [("benign", 0), ("malignant", 1)]:
            img_dir = os.path.join(path, label_name, "images")
            if not os.path.exists(img_dir):
                print(f"⚠️ Missing directory: {img_dir}")
                continue
            for fname in os.listdir(img_dir):
                if fname.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")):
                    images.append(os.path.join(img_dir, fname))
                    labels.append(label_val)
                    sources.append(name)
    return images, labels, sources


# ==== PATIENT-LEVEL GROUP CONSTRUCTION FOR CROSS-VALIDATION ====
# BUS-BRA and BUS-UCLM expose patient identifiers; other internal datasets do not.
# We use StratifiedGroupKFold with:
#   - true patient-level groups where IDs are available
#   - degenerate 1-image groups elsewhere (equivalent to stratified image-level CV)

BUS_BRA_PATIENT_CSV = "bus-bra_data.csv"
BUS_UCLM_PATIENT_CSV = "bus_uclm_patient_ids.csv"


def load_patient_id_maps():
    """Load patient ID lookup tables for BUS-BRA and BUS-UCLM if CSVs are available."""
    bus_bra_map, bus_uclm_map = {}, {}

    # BUS-BRA: columns ['ID', 'Case'] where ID matches image stem (e.g. 'bus_0001-l')
    if os.path.exists(BUS_BRA_PATIENT_CSV):
        try:
            df = pd.read_csv(BUS_BRA_PATIENT_CSV)
            if "ID" in df.columns and "Case" in df.columns:
                bus_bra_map = {
                    str(row["ID"]).strip(): str(row["Case"]).strip()
                    for _, row in df.iterrows()
                }
                print(
                    f"✓ Loaded BUS-BRA patient IDs from {BUS_BRA_PATIENT_CSV} ({len(bus_bra_map)} entries)"
                )
            else:
                print(
                    f"⚠️ BUS-BRA CSV {BUS_BRA_PATIENT_CSV} missing 'ID'/'Case' columns; using image-level groups for BUS-BRA"
                )
        except Exception as e:
            print(
                f"⚠️ Failed to load BUS-BRA patient CSV: {e}; using image-level groups for BUS-BRA"
            )
    else:
        print(
            f"ℹ️ BUS-BRA patient CSV not found at {BUS_BRA_PATIENT_CSV}; using image-level groups for BUS-BRA"
        )

    # BUS-UCLM: columns ['file_name', 'patient_id'] where file_name matches image filename
    if os.path.exists(BUS_UCLM_PATIENT_CSV):
        try:
            df = pd.read_csv(BUS_UCLM_PATIENT_CSV)
            if "file_name" in df.columns and "patient_id" in df.columns:
                bus_uclm_map = {
                    str(row["file_name"]).strip(): str(row["patient_id"]).strip()
                    for _, row in df.iterrows()
                }
                print(
                    f"✓ Loaded BUS-UCLM patient IDs from {BUS_UCLM_PATIENT_CSV} ({len(bus_uclm_map)} entries)"
                )
            else:
                print(
                    f"⚠️ BUS-UCLM CSV {BUS_UCLM_PATIENT_CSV} missing 'file_name'/'patient_id' columns; using image-level groups for BUS-UCLM"
                )
        except Exception as e:
            print(
                f"⚠️ Failed to load BUS-UCLM patient CSV: {e}; using image-level groups for BUS-UCLM"
            )
    else:
        print(
            f"ℹ️ BUS-UCLM patient CSV not found at {BUS_UCLM_PATIENT_CSV}; using image-level groups for BUS-UCLM"
        )

    return bus_bra_map, bus_uclm_map


def build_patient_groups(images, sources, bus_bra_map, bus_uclm_map):
    """Build group labels for StratifiedGroupKFold.

    - BUS-BRA: group by anonymised patient case ID (multiple views per patient).
    - BUS-UCLM: group by anonymised patient ID from CSV (multiple frames per patient).
    - All other internal datasets: fall back to unique per-image groups (image-level CV).
    """
    groups = []
    for img_path, src in zip(images, sources):
        p = Path(img_path)

        if src == "BUS-BRA" and bus_bra_map:
            key = p.stem  # e.g. 'bus_0001-l'
            case = bus_bra_map.get(key)
            if case is not None:
                group = f"{src}:case{case}"
            else:
                # Fallback to image-level grouping if ID not found
                group = f"{src}:{key}"

        elif src == "BUS-UCLM" and bus_uclm_map:
            key = p.name  # e.g. 'alwi_000.png'
            pid = bus_uclm_map.get(key)
            if pid is not None:
                group = f"{src}:{pid}"
            else:
                group = f"{src}:{p.stem}"

        else:
            # Datasets without explicit patient IDs (BUS_UC, UDIAT, USG)
            # or 1:1 patient:image → unique per-image group
            group = f"{src}:{p.stem}"

        groups.append(group)

    return np.array(groups)


# ==== AUGMENTATION (Literature-aligned: removed speckle noise) ====
class SourceAwareMedicalAugmentation:
    """
    Source-aware augmentation that scales intensity based on dataset size.
    Smaller datasets get weaker augmentation to prevent overfitting.
    """

    def __init__(self, source_name, source_sizes, p_augment=0.8):
        self.source_name = source_name
        self.source_size = source_sizes.get(source_name, 500)
        self.scale = 1.0 / max(np.sqrt(self.source_size), 1.0)
        self.p_augment = p_augment
        self.num_ops_range = (1, 3)

        rot = 5.0 * self.scale
        # Literature-aligned augmentations (horizontal flip, rotation, brightness, contrast)
        # Removed speckle noise as per SOTA papers
        self.augmentations = [
            ("horizontal_flip", 0.5),
            ("rotation", rot),
            ("brightness", 0.2),
            ("contrast", 0.2),
        ]

    def __call__(self, img):
        if random.random() > self.p_augment:
            return img
        k = random.randint(*self.num_ops_range)
        for op_name, param in random.sample(self.augmentations, k=k):
            if op_name == "horizontal_flip" and random.random() < param:
                img = F.hflip(img)
            elif op_name == "rotation":
                img = F.rotate(
                    img,
                    random.uniform(-param, param),
                    interpolation=InterpolationMode.BICUBIC,
                    fill=0,
                )
            elif op_name == "brightness":
                img = F.adjust_brightness(img, random.uniform(1 - param, 1 + param))
            elif op_name == "contrast":
                img = F.adjust_contrast(img, random.uniform(1 - param, 1 + param))
        return img


class PadSquareAndResize:
    """Pad image to square, then resize to target size."""

    def __init__(self, target_size=224, small_threshold=150, fill=0):
        self.target_size, self.small_threshold, self.fill = (
            target_size,
            small_threshold,
            fill,
        )

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


# ==== DATASET (Safe retry logic for missing masks) ====
class BUSDataset(Dataset):
    def __init__(
        self, images, labels, transform=None, border_ratio=0.0, border_type="uniform"
    ):
        self.images, self.labels, self.transform = images, labels, transform
        self.border_ratio, self.border_type = border_ratio, border_type
        self.cache = {}

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        original_idx = idx
        MAX_RETRIES = 10
        img = None
        label = self.labels[idx]

        for attempt in range(MAX_RETRIES):
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
                            img = extract_roi(
                                img_path, mp, self.border_ratio, self.border_type
                            )

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
                f"WARNING: Failed to load image after {MAX_RETRIES} attempts (original idx={original_idx})"
            )
            img = Image.new("RGB", (224, 224), color=0)

        if self.transform:
            img = self.transform(img)
        return img, label


def build_source_aware_train_dataset(
    X, y, src, border_ratio, border_type, model_name, source_sizes
):
    target = 299 if "inception" in model_name.lower() else 224
    padresize = PadSquareAndResize(target_size=target, small_threshold=150, fill=0)

    groups = defaultdict(lambda: {"images": [], "labels": []})
    for p, lbl, s in zip(X, y, src):
        groups[s]["images"].append(p)
        groups[s]["labels"].append(lbl)

    subdatasets, concat_weights = [], []
    for sname, data in groups.items():
        aug = SourceAwareMedicalAugmentation(sname, source_sizes)
        tf = transforms.Compose(
            [
                aug,
                padresize,
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        ds = BUSDataset(data["images"], data["labels"], tf, border_ratio, border_type)
        subdatasets.append(ds)
        w = 1.0 / max(np.sqrt(source_sizes.get(sname, 500)), 1.0)
        concat_weights.extend([w] * len(ds))
    return ConcatDataset(subdatasets), np.array(concat_weights, dtype=np.float32)


def build_eval_transform(model_name):
    target = 299 if "inception" in model_name.lower() else 224
    padresize = PadSquareAndResize(target_size=target, small_threshold=150, fill=0)
    return transforms.Compose(
        [
            padresize,
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )


# ==== MODELS (All unfrozen for fair benchmarking) ====
def get_model(name):
    """
    Get pretrained model with final classification layer modified for binary classification.
    All models fully unfrozen for fair comparison (including VGG16).

    Benchmark Models:
    - Classical CNNs: VGG16
    - Residual/Modern ConvNets: ResNet-50, DenseNet-121, RegNetY-8GF, ConvNeXt-Tiny
    - Efficient CNNs: MobileNetV3-Small, EfficientNet-B0
    - Transformers and Hybrids: DeiT-Tiny-Distilled, Swin-Tiny, MaxViT-Tiny
    """
    if name == "vgg16":
        m = vgg16(weights="IMAGENET1K_V1")
        # FIXED: Fully unfrozen for fair benchmarking (no feature freezing)
        m.classifier[6] = nn.Linear(m.classifier[6].in_features, 2)

    elif name == "resnet50":
        m = resnet50(weights="IMAGENET1K_V2")
        m.fc = nn.Linear(m.fc.in_features, 2)

    elif name == "densenet121":
        m = densenet121(weights="IMAGENET1K_V1")
        m.classifier = nn.Linear(m.classifier.in_features, 2)

    elif name == "regnety_008":
        m = timm.create_model("regnety_008", pretrained=True)
        m.head.fc = nn.Linear(m.head.fc.in_features, 2)

    elif name == "convnext_tiny":
        m = convnext_tiny(weights="IMAGENET1K_V1")
        m.classifier[2] = nn.Linear(m.classifier[2].in_features, 2)

    elif name == "mobilenet_v3_small":
        m = mobilenet_v3_small(weights="IMAGENET1K_V1")
        m.classifier[3] = nn.Linear(m.classifier[3].in_features, 2)

    elif name == "efficientnet_b0":
        m = efficientnet_b0(weights="IMAGENET1K_V1")
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, 2)

    elif name == "deit_tiny_distilled_patch16_224":
        m = timm.create_model("deit_tiny_distilled_patch16_224", pretrained=True)
        m.head = nn.Linear(m.head.in_features, 2)
        m.head_dist = nn.Linear(m.head_dist.in_features, 2)

    elif name == "swin_t":
        m = swin_t(weights="IMAGENET1K_V1")
        m.head = nn.Linear(m.head.in_features, 2)

    elif name == "maxvit_t":
        m = maxvit_t(weights="IMAGENET1K_V1")
        m.classifier[5] = nn.Linear(m.classifier[5].in_features, 2)

    else:
        raise ValueError(f"Unknown model: {name}")

    return m


# ==== THRESHOLD & METRICS ====
def _metrics_from_confusion(tp, tn, fp, fn):
    acc = (tp + tn) / max(1, (tp + tn + fp + fn))
    sens = tp / max(1, (tp + fn))
    spec = tn / max(1, (tn + fp))
    prec = tp / max(1, (tp + fp))
    f1 = (2 * tp) / max(1, (2 * tp + fp + fn))
    bal = 0.5 * (sens + spec)
    npv = tn / max(1, (tn + fn))
    return acc, sens, spec, prec, f1, bal, npv


def find_threshold(
    y_true, y_prob, strategy="youden", beta=1.0, cost_fp=1.0, cost_fn=1.0
):
    thr_grid = np.linspace(0.0, 1.0, 501)
    best_thr, best_score, best_tuple = 0.5, -np.inf, None
    for th in thr_grid:
        y_pred = (y_prob >= th).astype(int)
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        if cm.shape != (2, 2):
            continue
        tn, fp, fn, tp = cm.ravel()
        acc, sens, spec, prec, f1, bal, _ = _metrics_from_confusion(tp, tn, fp, fn)
        if strategy in ("youden", "balanced_accuracy"):
            score = sens + spec - 1.0
        elif strategy == "f1":
            score = f1
        elif strategy == "f_beta":
            beta2 = beta * beta
            denom = beta2 * prec + sens
            score = (1 + beta2) * prec * sens / (denom if denom > 0 else 1e-12)
        elif strategy == "cost":
            total = tp + tn + fp + fn
            score = -(cost_fp * fp + cost_fn * fn) / max(1, total)
        elif strategy == "fixed":
            return float(FIXED_THRESHOLD), None
        else:
            score = sens + spec - 1.0
        tiebreak = (bal, spec, -abs(th - 0.5))
        candidate = (score, tiebreak)
        if candidate > (
            best_score,
            best_tuple if best_tuple is not None else (-np.inf, -np.inf, np.inf),
        ):
            best_score, best_thr, best_tuple = score, th, tiebreak
    return float(best_thr), float(best_score)


def compute_operating_points(y_true, y_prob, target_sens=0.90, target_spec=0.90):
    fpr, tpr, thr = roc_curve(y_true, y_prob)
    out = {}
    j_scores = tpr - fpr
    j_idx = int(np.argmax(j_scores))
    out["youden_j"] = float(j_scores[j_idx])
    out["youden_threshold"] = float(thr[j_idx]) if j_idx < len(thr) else 0.5
    out["youden_sensitivity"] = float(tpr[j_idx])
    out["youden_specificity"] = float(1 - fpr[j_idx])

    idx = np.where(tpr >= target_sens)[0]
    if idx.size:
        i = int(idx[np.argmin(fpr[idx])])
        out["spec_at_target_sens"] = float(1 - fpr[i])
        out["thr_at_target_sens"] = (
            float(thr[i]) if i < len(thr) else out["youden_threshold"]
        )
    else:
        out["spec_at_target_sens"] = None
        out["thr_at_target_sens"] = None

    idx = np.where((1 - fpr) >= target_spec)[0]
    if idx.size:
        i = int(idx[np.argmax(tpr[idx])])
        out["sens_at_target_spec"] = float(tpr[i])
        out["thr_at_target_spec"] = (
            float(thr[i]) if i < len(thr) else out["youden_threshold"]
        )
    else:
        out["sens_at_target_spec"] = None
        out["thr_at_target_spec"] = None

    return out


def clopper_pearson_ci(k, n, alpha=0.05):
    if n <= 0:
        return (0.0, 1.0)
    lo = beta_dist.ppf(alpha / 2, k, n - k + 1) if k > 0 else 0.0
    hi = beta_dist.ppf(1 - alpha / 2, k + 1, n - k) if k < n else 1.0
    return float(lo), float(hi)


def bootstrap_auc_ci(y_true, y_prob, alpha=0.95, n_boot=2000, seed=42):
    rng = np.random.RandomState(seed)
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    pos = np.where(y_true == 1)[0]
    neg = np.where(y_true == 0)[0]
    n_pos, n_neg = len(pos), len(neg)
    if n_pos == 0 or n_neg == 0:
        return (0.5, 0.5)
    aucs = []
    for _ in range(n_boot):
        idx = np.concatenate(
            [rng.choice(pos, n_pos, True), rng.choice(neg, n_neg, True)]
        )
        aucs.append(roc_auc_score(y_true[idx], y_prob[idx]))
    lo = float(np.percentile(aucs, (1 - alpha) / 2 * 100))
    hi = float(np.percentile(aucs, (1 + alpha) / 2 * 100))
    return lo, hi


def save_curves(y_true, y_prob, out_prefix):
    os.makedirs(os.path.dirname(out_prefix), exist_ok=True)
    if len(set(y_true)) > 1:
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        plt.figure()
        plt.plot(fpr, tpr, label="ROC")
        plt.plot([0, 1], [0, 1], linestyle="--")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_prefix + "_roc.png")
        plt.close()

        prec, rec, _ = precision_recall_curve(y_true, y_prob)
        plt.figure()
        plt.plot(rec, prec, label="PR")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title("Precision-Recall Curve")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_prefix + "_pr.png")
        plt.close()


# ==== CALIBRATION FUNCTIONS ====
def compute_ece(y_true, y_prob, n_bins=15):
    """Compute Expected Calibration Error (ECE) with M=15 bins."""
    bins = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.digitize(y_prob, bins[:-1]) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)

    ece = 0.0
    for i in range(n_bins):
        mask = bin_indices == i
        if mask.sum() > 0:
            bin_accuracy = y_true[mask].mean()
            bin_confidence = y_prob[mask].mean()
            bin_weight = mask.sum() / len(y_true)
            ece += bin_weight * abs(bin_accuracy - bin_confidence)

    return float(ece)


def plot_reliability_diagram(
    y_true, y_prob, n_bins=15, save_path=None, title="Reliability Diagram"
):
    """Plot reliability diagram for calibration assessment."""
    bins = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.digitize(y_prob, bins[:-1]) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)

    bin_accs = []
    bin_confs = []
    bin_counts = []

    for i in range(n_bins):
        mask = bin_indices == i
        if mask.sum() > 0:
            bin_accs.append(y_true[mask].mean())
            bin_confs.append(y_prob[mask].mean())
            bin_counts.append(mask.sum())

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.plot([0, 1], [0, 1], "k--", linewidth=2, label="Perfect Calibration")

    if bin_confs:
        ax.scatter(
            bin_confs,
            bin_accs,
            s=np.array(bin_counts) * 2,
            alpha=0.6,
            c="blue",
            edgecolors="black",
            linewidth=1.5,
            label="Model Calibration",
        )

    ax.set_xlabel("Mean Predicted Probability", fontsize=12)
    ax.set_ylabel("Observed Fraction Positive", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


class TemperatureScaling(nn.Module):
    """Temperature scaling for post-hoc calibration."""

    def __init__(self):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1))

    def forward(self, logits):
        return logits / self.temperature

    def fit(self, val_loader, model, device, max_iter=100):
        """
        Tune temperature on validation set to minimize NLL.
        Returns optimal temperature value.
        """
        model.eval()
        logits_list = []
        labels_list = []

        # Collect on CPU first to save GPU memory
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device, non_blocking=True)
                outputs = model(inputs)
                logits_list.append(outputs.cpu())
                labels_list.append(labels)

        # Concatenate on CPU
        logits = torch.cat(logits_list)
        labels = torch.cat(labels_list)

        # Move to device for optimization
        logits = logits.to(device)
        labels = labels.to(device)

        # Optimize temperature
        nll_criterion = nn.CrossEntropyLoss()
        optimizer = optim.LBFGS([self.temperature], lr=0.01, max_iter=max_iter)

        def eval_nll():
            optimizer.zero_grad()
            loss = nll_criterion(self.forward(logits), labels)
            loss.backward()
            return loss

        optimizer.step(eval_nll)

        return self.temperature.item()


# ==== EVALUATION (TTA removed - calibration-focused) ====
def evaluate_comprehensive(
    model,
    loader,
    decision_threshold=0.5,
    compute_opt_threshold=True,
    threshold_strategy=THRESHOLD_STRATEGY,
    f_beta=F_BETA,
    cost_fp=COST_FP,
    cost_fn=COST_FN,
    curves_prefix=None,
    skip_bootstrap=False,
    skip_curves=False,
    temperature_scaler=None,
):
    """
    Comprehensive evaluation with optional temperature scaling.
    TTA removed - prioritizing calibration for clinical deployment.
    Returns: (metrics_dict, probabilities, labels)
    """
    model.eval()
    probs_all, preds_all, labels_all = [], [], []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)

            outputs = model(x)

            # Apply temperature scaling if provided
            if temperature_scaler is not None:
                outputs = temperature_scaler(outputs)

            probs = torch.softmax(outputs, dim=1)[:, 1]
            preds = (probs >= decision_threshold).long()

            probs_all.extend(probs.cpu().numpy())
            preds_all.extend(preds.cpu().numpy())
            labels_all.extend(y.numpy())

    y_true = np.array(labels_all)
    y_prob = np.array(probs_all)
    y_pred = np.array(preds_all)

    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel() if cm.shape == (2, 2) else (0, 0, 0, 0)
    total = tp + tn + fp + fn

    acc, sens, spec, prec, f1, bal, npv = _metrics_from_confusion(tp, tn, fp, fn)

    if len(set(y_true)) > 1:
        auc = roc_auc_score(y_true, y_prob)
        if skip_bootstrap:
            auc_lo, auc_hi = (float("nan"), float("nan"))
        else:
            auc_lo, auc_hi = bootstrap_auc_ci(
                y_true, y_prob, alpha=0.95, n_boot=2000, seed=SEED
            )
        auprc = average_precision_score(y_true, y_prob)
    else:
        auc = auprc = 0.5
        auc_lo = auc_hi = 0.5

    ece = compute_ece(y_true, y_prob, n_bins=15)

    metrics = {
        "threshold": float(decision_threshold),
        "prevalence": float(np.mean(y_true)),
        "accuracy": acc,
        "sensitivity": sens,
        "specificity": spec,
        "precision": prec,
        "npv": npv,
        "f1": f1,
        "balanced_accuracy": bal,
        "auc": auc,
        "auc_ci": (auc_lo, auc_hi),
        "auprc": auprc,
        "mcc": matthews_corrcoef(y_true, y_pred) if len(set(y_true)) > 1 else 0.0,
        "cohen_kappa": cohen_kappa_score(y_true, y_pred),
        "brier": brier_score_loss(y_true, y_prob),
        "ece": ece,
        "confusion_matrix": cm.tolist(),
        "support": {
            "tp": int(tp),
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "total": int(total),
        },
    }

    if len(set(y_true)) > 1:
        ops = compute_operating_points(
            y_true, y_prob, target_sens=0.90, target_spec=0.90
        )
        metrics.update(ops)

    if compute_opt_threshold and len(set(y_true)) > 1:
        opt_thr, _ = find_threshold(
            y_true,
            y_prob,
            strategy=threshold_strategy,
            beta=f_beta,
            cost_fp=cost_fp,
            cost_fn=cost_fn,
        )
        metrics["opt_threshold"] = float(opt_thr)

        y_pred_op = (y_prob >= opt_thr).astype(int)
        cm_op = confusion_matrix(y_true, y_pred_op, labels=[0, 1])
        if cm_op.shape == (2, 2):
            tn_op, fp_op, fn_op, tp_op = cm_op.ravel()
            sens_ci = clopper_pearson_ci(tp_op, tp_op + fn_op, alpha=0.05)
            spec_ci = clopper_pearson_ci(tn_op, tn_op + fp_op, alpha=0.05)
            ppv_ci = (
                clopper_pearson_ci(tp_op, tp_op + fp_op, alpha=0.05)
                if (tp_op + fp_op) > 0
                else (0.0, 1.0)
            )
            npv_ci = (
                clopper_pearson_ci(tn_op, tn_op + fn_op, alpha=0.05)
                if (tn_op + fn_op) > 0
                else (0.0, 1.0)
            )
            metrics["sens_ci_at_opt"] = sens_ci
            metrics["spec_ci_at_opt"] = spec_ci
            metrics["ppv_ci_at_opt"] = ppv_ci
            metrics["npv_ci_at_opt"] = npv_ci

    sens_ci = (
        clopper_pearson_ci(tp, tp + fn, alpha=0.05) if (tp + fn) > 0 else (0.0, 1.0)
    )
    spec_ci = (
        clopper_pearson_ci(tn, tn + fp, alpha=0.05) if (tn + fp) > 0 else (0.0, 1.0)
    )
    ppv_ci = (
        clopper_pearson_ci(tp, tp + fp, alpha=0.05) if (tp + fp) > 0 else (0.0, 1.0)
    )
    npv_ci = (
        clopper_pearson_ci(tn, tn + fn, alpha=0.05) if (tn + fn) > 0 else (0.0, 1.0)
    )
    metrics["sens_ci"] = sens_ci
    metrics["spec_ci"] = spec_ci
    metrics["ppv_ci"] = ppv_ci
    metrics["npv_ci"] = npv_ci

    if (
        SAVE_CURVES
        and curves_prefix is not None
        and len(set(y_true)) > 1
        and not skip_curves
    ):
        save_curves(y_true, y_prob, curves_prefix)

    return metrics, y_prob, y_true


# ==== TRAINING (Complete checkpoint restoration) ====
def _labels_from_dataset(d):
    if isinstance(d, ConcatDataset):
        lab = []
        for sub in d.datasets:
            lab.extend(sub.labels)
        return lab
    return list(getattr(d, "labels", []))


def train_model(
    model,
    train_loader,
    val_loader,
    model_name,
    exp_name,
    fold,
    checkpoint_dir="checkpoints",
):
    model = model.to(device)

    tl = _labels_from_dataset(train_loader.dataset)
    counts = np.bincount(np.array(tl, dtype=np.int64), minlength=2).astype(np.float32)
    counts[counts == 0] = 1.0
    w0, w1 = 1.0 / counts[0], 1.0 / counts[1]
    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor([w0, w1], dtype=torch.float, device=device)
    )

    optim_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.Adam(optim_params, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3
    )

    scaler = GradScaler()

    os.makedirs(checkpoint_dir, exist_ok=True)
    ckpt_path = f"{checkpoint_dir}/{SCRIPT_VERSION}_{model_name}_{exp_name}_f{fold}.pth"
    best_path = ckpt_path.replace(".pth", "_best.pth")

    start_epoch = 0
    history = {"train_loss": [], "val_loss": [], "val_acc": [], "epoch_time_sec": []}

    # Complete checkpoint restoration
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location="cpu")
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])

        if "scheduler" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler"])
        if "scaler" in ckpt:
            scaler.load_state_dict(ckpt["scaler"])

        start_epoch = ckpt["epoch"] + 1
        history = ckpt["history"]
        best_loss = ckpt.get("best_loss", float("inf"))
        patience = ckpt.get("patience", 0)

        print(
            f"✓ Resumed from epoch {start_epoch} (best_loss={best_loss:.4f}, patience={patience})"
        )
    else:
        best_loss = float("inf")
        patience = 0

    best_weights = {k: v.clone().detach().cpu() for k, v in model.state_dict().items()}
    train_t0 = perf_counter()

    for epoch in range(start_epoch, NUM_EPOCHS):
        ep_t0 = perf_counter()

        model.train()
        tr_loss = 0.0
        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            with autocast():
                loss = criterion(model(x), y)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            tr_loss += loss.item()

        model.eval()
        va_loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)

                with autocast():
                    out = model(x)
                    va_loss += criterion(out, y).item()

                correct += (out.argmax(1) == y).sum().item()
                total += y.size(0)

        history["train_loss"].append(tr_loss / max(1, len(train_loader)))
        history["val_loss"].append(va_loss / max(1, len(val_loader)))
        history["val_acc"].append(correct / max(1, total))
        history["epoch_time_sec"].append(perf_counter() - ep_t0)

        scheduler.step(history["val_loss"][-1])

        if history["val_loss"][-1] < best_loss:
            best_loss = history["val_loss"][-1]
            best_weights = {
                k: v.clone().detach().cpu() for k, v in model.state_dict().items()
            }
            patience = 0

            torch.save(best_weights, best_path)
            # Save complete state
            torch.save(
                {
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "scaler": scaler.state_dict(),
                    "history": history,
                    "best_loss": best_loss,
                    "patience": patience,
                },
                ckpt_path,
            )
        else:
            patience += 1
            if patience >= PATIENCE:
                break

    history["total_time_sec"] = perf_counter() - train_t0

    if os.path.exists(best_path):
        model.load_state_dict(torch.load(best_path, map_location="cpu"))
    else:
        model.load_state_dict({k: v.to(device) for k, v in best_weights.items()})

    try:
        os.remove(ckpt_path)
    except OSError:
        pass

    return model, history


# ==== EXPERIMENTS ====
def run_experiments():
    print(f"\n=== {SCRIPT_VERSION} Pipeline with Calibration ===")
    train_images, train_labels, train_sources = load_data_with_source(INTERNAL_DATASETS)
    test_images, test_labels, _ = load_data_with_source(EXTERNAL_DATASETS)
    if not train_images:
        print("❌ No training data")
        return

    prev_results, prev_index = load_previous_results()
    results = list(prev_results)

    # Build patient-level groups for StratifiedGroupKFold
    bus_bra_map, bus_uclm_map = load_patient_id_maps()
    train_groups = build_patient_groups(
        train_images, train_sources, bus_bra_map, bus_uclm_map
    )

    sgkf = StratifiedGroupKFold(n_splits=NUM_FOLDS, shuffle=True, random_state=SEED)

    for exp in EXPERIMENTS:
        print(
            f"\n-- Experiment: {exp['name']} (ratio={exp['ratio']}, type={exp['type']}) --"
        )
        for model_name in MODEL_NAMES:
            print(f"  Model: {model_name}")
            fold_rows = []
            for fold, (tr_idx, va_idx) in enumerate(
                sgkf.split(train_images, train_labels, groups=train_groups), 1
            ):
                clear_memory()
                key = (model_name, exp["name"], fold)
                mark = done_marker_path(model_name, exp["name"], fold)

                if (
                    SKIP_COMPLETED
                    and key in prev_index
                    and fold_is_done(model_name, exp["name"], fold)
                ):
                    print(f"    Fold {fold}/{NUM_FOLDS} - SKIP (completed)")
                    results.append(prev_index[key])
                    fold_rows.append(prev_index[key])
                    continue

                print(f"    Fold {fold}/{NUM_FOLDS}")
                X_train = [train_images[i] for i in tr_idx]
                y_train = [train_labels[i] for i in tr_idx]
                S_train = [train_sources[i] for i in tr_idx]
                X_val = [train_images[i] for i in va_idx]
                y_val = [train_labels[i] for i in va_idx]

                train_dataset, concat_weights = build_source_aware_train_dataset(
                    X_train,
                    y_train,
                    S_train,
                    border_ratio=exp["ratio"],
                    border_type=exp["type"],
                    model_name=model_name,
                    source_sizes=SOURCE_SIZES,
                )
                val_tf = build_eval_transform(model_name)
                val_ds = BUSDataset(X_val, y_val, val_tf, exp["ratio"], exp["type"])
                # FIXED: Test set uses SAME preprocessing as val
                test_ds = BUSDataset(
                    test_images, test_labels, val_tf, exp["ratio"], exp["type"]
                )

                num_workers = min(8, os.cpu_count() or 1)
                if USE_SOURCE_SAMPLER:
                    sampler = WeightedRandomSampler(
                        torch.tensor(concat_weights, dtype=torch.float),
                        num_samples=len(concat_weights),
                        replacement=True,
                    )
                    train_loader = DataLoader(
                        train_dataset,
                        BATCH_SIZE,
                        sampler=sampler,
                        num_workers=num_workers,
                        pin_memory=True,
                        persistent_workers=(num_workers > 0),
                    )
                else:
                    train_loader = DataLoader(
                        train_dataset,
                        BATCH_SIZE,
                        shuffle=True,
                        num_workers=num_workers,
                        pin_memory=True,
                        persistent_workers=(num_workers > 0),
                    )

                val_loader = DataLoader(
                    val_ds,
                    BATCH_SIZE,
                    num_workers=num_workers,
                    pin_memory=True,
                    persistent_workers=(num_workers > 0),
                )
                test_loader = DataLoader(
                    test_ds,
                    BATCH_SIZE,
                    num_workers=num_workers,
                    pin_memory=True,
                    persistent_workers=(num_workers > 0),
                )

                # Train model
                model = get_model(model_name)
                model, history = train_model(
                    model, train_loader, val_loader, model_name, exp["name"], fold
                )

                # Evaluate BEFORE calibration
                print(f"      → Evaluating uncalibrated...")
                curves_dir = f"results/curves/{exp['name']}_{model_name}_f{fold}"
                val_metrics_uncal, val_probs, val_labels = evaluate_comprehensive(
                    model,
                    val_loader,
                    decision_threshold=0.5,
                    compute_opt_threshold=True,
                    threshold_strategy=THRESHOLD_STRATEGY,
                    f_beta=F_BETA,
                    cost_fp=COST_FP,
                    cost_fn=COST_FN,
                    curves_prefix=curves_dir + "_val_uncalibrated",
                    skip_bootstrap=True,
                    skip_curves=False,
                )

                # Fit temperature scaling on validation
                print(f"      → Fitting temperature scaling...")
                temp_scaler = TemperatureScaling().to(device)
                optimal_temp = temp_scaler.fit(val_loader, model, device, max_iter=100)
                print(f"      → Optimal temperature: {optimal_temp:.4f}")

                # Evaluate AFTER calibration
                print(f"      → Evaluating calibrated...")
                val_metrics_cal, _, _ = evaluate_comprehensive(
                    model,
                    val_loader,
                    decision_threshold=0.5,
                    compute_opt_threshold=True,
                    threshold_strategy=THRESHOLD_STRATEGY,
                    f_beta=F_BETA,
                    cost_fp=COST_FP,
                    cost_fn=COST_FN,
                    curves_prefix=curves_dir + "_val_calibrated",
                    skip_bootstrap=True,
                    skip_curves=False,
                    temperature_scaler=temp_scaler,
                )

                # Efficient reliability diagram generation
                rel_dir = f"results/reliability/{exp['name']}_{model_name}_f{fold}"
                os.makedirs(rel_dir, exist_ok=True)

                # Plot uncalibrated
                plot_reliability_diagram(
                    val_labels,
                    val_probs,
                    n_bins=15,
                    save_path=f"{rel_dir}/reliability_uncalibrated.png",
                    title=f"Uncalibrated - {model_name} Fold {fold}",
                )

                # Get calibrated probabilities efficiently
                model.eval()
                logits_list = []
                with torch.no_grad():
                    for x, _ in val_loader:
                        x = x.to(device, non_blocking=True)
                        logits = model(x)
                        logits_list.append(logits.cpu())

                logits_tensor = torch.cat(logits_list).to(device)
                with torch.no_grad():
                    scaled_logits = temp_scaler(logits_tensor)
                    val_probs_cal = (
                        torch.softmax(scaled_logits, dim=1)[:, 1].cpu().numpy()
                    )

                plot_reliability_diagram(
                    val_labels,
                    val_probs_cal,
                    n_bins=15,
                    save_path=f"{rel_dir}/reliability_calibrated.png",
                    title=f"Calibrated (T={optimal_temp:.3f}) - {model_name} Fold {fold}",
                )

                # Use optimal threshold from CALIBRATED validation for test
                op_thr = val_metrics_cal.get("opt_threshold", 0.5)

                # Evaluate on test with calibrated model
                test_metrics_cal, test_probs, test_labels = evaluate_comprehensive(
                    model,
                    test_loader,
                    decision_threshold=op_thr,
                    compute_opt_threshold=False,
                    threshold_strategy=THRESHOLD_STRATEGY,
                    curves_prefix=curves_dir + "_test_calibrated",
                    skip_bootstrap=False,
                    skip_curves=False,
                    temperature_scaler=temp_scaler,
                )

                # Save enhanced results
                row = {
                    "model": model_name,
                    "fold": fold,
                    "experiment": exp["name"],
                    "border_ratio": exp["ratio"],
                    "border_type": exp["type"],
                    "history": history,
                    "train_time_sec": history.get("total_time_sec"),
                    "avg_epoch_sec": (
                        float(np.mean(history["epoch_time_sec"]))
                        if history.get("epoch_time_sec")
                        else None
                    ),
                    # Calibration info
                    "temperature": optimal_temp,
                    "val_ece_uncalibrated": val_metrics_uncal["ece"],
                    "val_ece_calibrated": val_metrics_cal["ece"],
                    "ece_improvement": val_metrics_uncal["ece"]
                    - val_metrics_cal["ece"],
                    # Validation metrics (calibrated)
                    "val_metrics": val_metrics_cal,
                    "val_metrics_uncalibrated": val_metrics_uncal,
                    # Test metrics (calibrated)
                    "test_metrics": test_metrics_cal,
                    "threshold_strategy": THRESHOLD_STRATEGY,
                    # Threshold portability
                    "val_operating_points": {
                        "youden_threshold": val_metrics_cal.get("youden_threshold"),
                        "sens_90spec_threshold": val_metrics_cal.get(
                            "thr_at_target_sens"
                        ),
                        "spec_90sens_threshold": val_metrics_cal.get(
                            "thr_at_target_spec"
                        ),
                    },
                }
                results.append(row)
                fold_rows.append(row)

                mark.write_text("done")

                # Memory cleanup
                model = model.cpu()
                del model
                clear_memory()

                print(
                    f"      → Val AUC {val_metrics_cal['auc']:.3f} | "
                    f"Test BalAcc {test_metrics_cal['balanced_accuracy']:.3f} | "
                    f"ECE: {val_metrics_uncal['ece']:.4f}→{val_metrics_cal['ece']:.4f}"
                )

            if fold_rows:
                ba = [r["test_metrics"]["balanced_accuracy"] for r in fold_rows]
                ece_improv = [r["ece_improvement"] for r in fold_rows]
                print(
                    f"    Mean: Test BalAcc={np.mean(ba):.3f}±{np.std(ba):.3f}, "
                    f"ECE Improvement={np.mean(ece_improv):.4f}±{np.std(ece_improv):.4f}"
                )

    save_results(results)
    print(f"\n✅ Done! Results saved with version tag: {SCRIPT_VERSION}")


def save_results(results):
    """Save results to versioned JSON and CSV files."""
    os.makedirs("results", exist_ok=True)

    def _py(v):
        if isinstance(v, (np.floating, np.integer)):
            return float(v)
        if isinstance(v, np.ndarray):
            return v.tolist()
        return v

    out = []
    for r in results:
        out.append(
            {
                "model": r["model"],
                "fold": r["fold"],
                "experiment": r["experiment"],
                "border_ratio": r["border_ratio"],
                "border_type": r["border_type"],
                "train_time_sec": _py(r.get("train_time_sec")),
                "avg_epoch_sec": _py(r.get("avg_epoch_sec")),
                "threshold_strategy": r.get("threshold_strategy"),
                "temperature": r.get("temperature"),
                "val_ece_uncalibrated": r.get("val_ece_uncalibrated"),
                "val_ece_calibrated": r.get("val_ece_calibrated"),
                "ece_improvement": r.get("ece_improvement"),
                "history": r.get("history", {}),
                "val_metrics": {k: _py(v) for k, v in r["val_metrics"].items()},
                "val_metrics_uncalibrated": {
                    k: _py(v) for k, v in r.get("val_metrics_uncalibrated", {}).items()
                },
                "test_metrics": {k: _py(v) for k, v in r["test_metrics"].items()},
                "val_operating_points": r.get("val_operating_points"),
            }
        )

    # Save with version tag
    with open(f"results/raw_results_{SCRIPT_VERSION}.json", "w") as f:
        json.dump(out, f, indent=2)

    rows = []
    for r in results:
        v, t = r["val_metrics"], r["test_metrics"]
        row = {
            "Model": r["model"],
            "Fold": r["fold"],
            "Experiment": r["experiment"],
            "Border_Type": r["border_type"],
            "Border_Ratio": r["border_ratio"],
            "Train_Time_sec": r.get("train_time_sec"),
            "Avg_Epoch_sec": r.get("avg_epoch_sec"),
            "Thr_Strategy": r.get("threshold_strategy"),
            "Temperature": r.get("temperature"),
            "Val_ECE_Uncal": r.get("val_ece_uncalibrated"),
            "Val_ECE_Cal": r.get("val_ece_calibrated"),
            "ECE_Improvement": r.get("ece_improvement"),
            "Val_AUC": v["auc"],
            "Val_AUROC_CI_L": v["auc_ci"][0],
            "Val_AUROC_CI_U": v["auc_ci"][1],
            "Val_AUPRC": v["auprc"],
            "Val_BalAcc": v["balanced_accuracy"],
            "Val_YoudenJ": v.get("youden_j"),
            "Val_YoudenThr": v.get("youden_threshold"),
            "Val_Sens@Youden": v.get("youden_sensitivity"),
            "Val_Spec@Youden": v.get("youden_specificity"),
            "Val_OP_Thr": v.get("opt_threshold"),
            "Val_Sens_CI_L": v.get("sens_ci_at_opt", (None, None))[0],
            "Val_Sens_CI_U": v.get("sens_ci_at_opt", (None, None))[1],
            "Val_Spec_CI_L": v.get("spec_ci_at_opt", (None, None))[0],
            "Val_Spec_CI_U": v.get("spec_ci_at_opt", (None, None))[1],
            "Val_PPV_CI_L": v.get("ppv_ci_at_opt", (None, None))[0],
            "Val_PPV_CI_U": v.get("ppv_ci_at_opt", (None, None))[1],
            "Val_NPV_CI_L": v.get("npv_ci_at_opt", (None, None))[0],
            "Val_NPV_CI_U": v.get("npv_ci_at_opt", (None, None))[1],
            "Val_Spec@90Sens": v.get("spec_at_target_sens"),
            "Val_Sens@90Spec": v.get("sens_at_target_spec"),
            "Test_Thr": t["threshold"],
            "Test_AUC": t["auc"],
            "Test_AUROC_CI_L": t["auc_ci"][0],
            "Test_AUROC_CI_U": t["auc_ci"][1],
            "Test_AUPRC": t["auprc"],
            "Test_Acc": t["accuracy"],
            "Test_BalAcc": t["balanced_accuracy"],
            "Test_F1": t["f1"],
            "Test_Sens": t["sensitivity"],
            "Test_Spec": t["specificity"],
            "Test_ECE": t.get("ece"),
            "Test_Sens_CI_L": t.get("sens_ci", (None, None))[0],
            "Test_Sens_CI_U": t.get("sens_ci", (None, None))[1],
            "Test_Spec_CI_L": t.get("spec_ci", (None, None))[0],
            "Test_Spec_CI_U": t.get("spec_ci", (None, None))[1],
            "Test_PPV_CI_L": t.get("ppv_ci", (None, None))[0],
            "Test_PPV_CI_U": t.get("ppv_ci", (None, None))[1],
            "Test_NPV_CI_L": t.get("npv_ci", (None, None))[0],
            "Test_NPV_CI_U": t.get("npv_ci", (None, None))[1],
            "Test_MCC": t["mcc"],
            "Test_Kappa": t["cohen_kappa"],
            "Test_Brier": t["brier"],
        }
        rows.append(row)
    pd.DataFrame(rows).to_csv(f"results/summary_{SCRIPT_VERSION}.csv", index=False)

    df = pd.DataFrame(rows)
    (
        df.groupby(["Experiment", "Model"])
        .agg(
            {
                "Test_BalAcc": ["mean", "std"],
                "Test_Sens": ["mean", "std"],
                "Test_Spec": ["mean", "std"],
                "Test_AUC": ["mean", "std"],
                "Test_AUPRC": ["mean", "std"],
                "ECE_Improvement": ["mean", "std"],
            }
        )
        .round(3)
    ).to_csv(f"results/aggregate_results_{SCRIPT_VERSION}.csv")

    print(f"📊 Saved: results/raw_results_{SCRIPT_VERSION}.json")
    print(f"📊 Saved: results/summary_{SCRIPT_VERSION}.csv")
    print(f"📊 Saved: results/aggregate_results_{SCRIPT_VERSION}.csv")
    if SAVE_CURVES:
        print("🖼️ Curves + reliability diagrams saved")


def validate_datasets():
    """Validate that all datasets exist and report counts."""
    print("\n🔍 Dataset Validation")
    print(f"Expected structure: [dataset]/[benign|malignant]/images/")
    for title, dsets in [
        ("Internal", INTERNAL_DATASETS),
        ("External", EXTERNAL_DATASETS),
    ]:
        print(f"\n{title}:")
        for name, path in dsets.items():
            if not os.path.exists(path):
                print(f"  ❌ {name}: not found at {path}")
                continue
            total = sum(
                len(os.listdir(os.path.join(path, c, "images")))
                for c in ["benign", "malignant"]
                if os.path.exists(os.path.join(path, c, "images"))
            )
            print(f"  ✅ {name}: {total} images")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--validate":
        validate_datasets()
    else:
        print(f"🎯 Device: {device}")
        if torch.cuda.is_available():
            print(f"🎮 GPU: {torch.cuda.get_device_name()}")
        print(
            f"📋 Calibration-Focused Pipeline (Temperature Scaling, ECE, Reliability Diagrams)"
        )
        print(f"⚖️ Threshold strategy: {THRESHOLD_STRATEGY}")
        print(f"🔬 TTA: Disabled (prioritizing calibration for clinical deployment)")
        validate_datasets()
        run_experiments()
