"""
Benchmark Figures Generation Script
====================================
Generates publication-ready figures for breast ultrasound classification benchmark.

Font: DejaVu Sans (matplotlib default sans-serif)
Output: PNG (300 DPI) and PDF formats

Main figures:
1. External AUROC dot-whisker (mean +/- SD across folds)
2. Generalization scatter (validation vs external)
3. Calibration improvement (paired-dot)
4. Operating points (sensitivity vs specificity; paired-dot)
5. Threshold stability (validation Youden thresholds)
6. ROI ablation heatmap (AUROC) + delta heatmap vs baseline
7. ROI strategy summary (per-architecture dots + mean +/- SD)
8. Efficiency vs performance (epoch time vs external AUROC)

Supplementary (optional when external bootstrap summary is available):
S1. External AUROC with bootstrap 95% CI
S2. Ensemble uplift (ensemble AUROC - mean single-fold AUROC)
"""

import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Rectangle
from matplotlib.backends.backend_pdf import PdfPages

# =============================================================================
# CONFIGURATION
# =============================================================================

plt.rcParams.update({
    'font.family': 'sans-serif',  # Uses DejaVu Sans by default
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 11,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'savefig.dpi': 300,
    'axes.axisbelow': True,
    'axes.spines.top': False,
    'axes.spines.right': False,
})

ERR_LINEWIDTH = 1.6
ERR_CAPSIZE = 5
ERR_CAPSIZE_SMALL = 4
ERR_CAPTHICK = 1.6
SCATTER_SIZE_TOP = 90
SCATTER_SIZE_MID = 65
SCATTER_SIZE_BASE = 45
SCATTER_ALPHA = 0.9
EDGEWIDTH_TOP = 1.0

ARCH_NAMES = {
    'swin_t': 'Swin-T',
    'convnext_tiny': 'ConvNeXt-T',
    'deit_tiny_distilled_patch16_224': 'DeiT-T',
    'vgg16': 'VGG-16',
    'regnety_008': 'RegNetY',
    'efficientnet_b0': 'EffNet-B0',
    'densenet121': 'DenseNet',
    'maxvit_t': 'MaxViT-T',
    'mobilenet_v3_small': 'MobNetV3',
    'resnet50': 'ResNet-50',
}

ARCH_ORDER = [
    'Swin-T',
    'ConvNeXt-T',
    'DeiT-T',
    'VGG-16',
    'RegNetY',
    'EffNet-B0',
    'DenseNet',
    'MaxViT-T',
    'MobNetV3',
    'ResNet-50',
]
ARCH_NUM = {name: idx + 1 for idx, name in enumerate(ARCH_ORDER)}

CLASSICAL_CNN_MODELS = {
    'vgg16',
    'resnet50',
    'densenet121',
    'efficientnet_b0',
    'mobilenet_v3_small',
    'regnety_008',
}

DEFAULT_BASELINE_PATH = Path('results/raw_results_v2.0_final_artifacts_5pct.json')
DEFAULT_ROI_PATH = Path('results/raw_results_v2.0_final_roi_ablation_5pct.json')
DEFAULT_EXTERNAL_BOOTSTRAP_PATH = Path(
    'results/external_bootstrap_artifacts_5pct/external_auc_bootstrap_summary.csv'
)
DEFAULT_OUT_DIR = Path('results/benchmark_figures_v5')

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def flatten_runs(raw_list):
    """Convert raw JSON results to DataFrame."""
    rows = []
    for r in raw_list:
        row = {
            'Model': r.get('model'),
            'Fold': r.get('fold'),
            'Experiment': r.get('experiment'),
            'Avg_Epoch_sec': r.get('avg_epoch_sec')
        }
        valm = r.get('val_metrics', {}) or {}
        valm_uncal = r.get('val_metrics_uncalibrated', {}) or {}
        testm = r.get('test_metrics', {}) or {}
        val_ops = r.get('val_operating_points', {}) or {}
        row['Val_AUC'] = valm.get('auc')
        row['Test_AUC'] = testm.get('auc')
        row['Test_Threshold'] = testm.get('threshold')
        row['Sensitivity'] = testm.get('sensitivity')
        row['Specificity'] = testm.get('specificity')
        row['Test_Youden_Threshold'] = testm.get('youden_threshold')
        row['Test_Youden_Sensitivity'] = testm.get('youden_sensitivity')
        row['Test_Youden_Specificity'] = testm.get('youden_specificity')
        row['Val_Youden_Threshold'] = val_ops.get(
            'youden_threshold', valm.get('youden_threshold')
        )
        row['ECE_pre'] = r.get('val_ece_uncalibrated', valm_uncal.get('ece'))
        row['ECE_post'] = r.get('val_ece_calibrated', valm.get('ece'))
        row['Confusion_Matrix'] = testm.get('confusion_matrix')
        rows.append(row)
    return pd.DataFrame(rows)

def add_arch_names(df):
    """Map model names to display names."""
    df = df.copy()
    df['Architecture'] = df['Model'].map(ARCH_NAMES).fillna(df['Model'])
    return df

def get_gradient_color(rank, n):
    """Get viridis gradient color based on rank."""
    t = (rank - 1) / max(n - 1, 1)
    cmap = plt.get_cmap('viridis')
    return mcolors.to_hex(cmap(t))

def get_gradient_colors(n):
    """Get list of gradient colors for n items."""
    return [get_gradient_color(i+1, n) for i in range(n)]

def load_external_bootstrap_summary(path):
    """Load external bootstrap summary CSV if available."""
    if path is None:
        return None
    if str(path).strip().lower() in {'', 'none', 'null'}:
        return None
    p = Path(path)
    if not p.exists():
        print(f'WARNING: External bootstrap summary not found: {p}')
        return None
    df = pd.read_csv(p)
    if 'architecture' not in df.columns:
        print(f'WARNING: Missing "architecture" column in {p}')
        return None
    df = df.rename(columns={'architecture': 'Model'})
    df = add_arch_names(df)
    return df

def build_external_auc_table(df_base, ext_summary=None):
    """Return external AUROC table with consistent columns."""
    if ext_summary is not None:
        s = ext_summary.copy()
        s = s.rename(columns={
            'fold_auc_mean': 'AUC_mean',
            'fold_auc_sd': 'AUC_sd',
            'fold_auc_boot_ci95_l': 'AUC_ci_l',
            'fold_auc_boot_ci95_u': 'AUC_ci_u',
            'ensemble_auc': 'Ensemble_AUC',
            'ensemble_boot_ci95_l': 'Ensemble_CI_l',
            'ensemble_boot_ci95_u': 'Ensemble_CI_u',
        })
        cols = ['Model', 'Architecture', 'AUC_mean', 'AUC_sd', 'AUC_ci_l', 'AUC_ci_u',
                'Ensemble_AUC', 'Ensemble_CI_l', 'Ensemble_CI_u']
        for c in cols:
            if c not in s.columns:
                s[c] = np.nan
        return s[cols]

    s = df_base.groupby(['Model', 'Architecture'], as_index=False).agg(
        AUC_mean=('Test_AUC', 'mean'),
        AUC_sd=('Test_AUC', 'std'),
    )
    s['AUC_ci_l'] = np.nan
    s['AUC_ci_u'] = np.nan
    s['Ensemble_AUC'] = np.nan
    s['Ensemble_CI_l'] = np.nan
    s['Ensemble_CI_u'] = np.nan
    return s

def build_arch_number_legend():
    """Return multi-line legend text for architecture numbering."""
    parts = [f"{ARCH_NUM[name]}={name}" for name in ARCH_ORDER if name in ARCH_NUM]
    lines = []
    line = []
    for i, item in enumerate(parts, 1):
        line.append(item)
        if i % 4 == 0:
            lines.append(", ".join(line))
            line = []
    if line:
        lines.append(", ".join(line))
    return "\n".join(lines)

def save_fig(fig, out_path_base):
    """Save figure to PNG and PDF with a small pad to avoid cut-off labels."""
    fig.savefig(out_path_base.with_suffix('.png'), dpi=300, bbox_inches='tight', pad_inches=0.25, facecolor='white')
    fig.savefig(out_path_base.with_suffix('.pdf'), bbox_inches='tight', pad_inches=0.25, facecolor='white')

# =============================================================================
# LOAD DATA
# =============================================================================

def load_data(baseline_path, roi_path):
    """Load and preprocess data from JSON files."""
    baseline_path = Path(baseline_path)
    roi_path = Path(roi_path)

    if not baseline_path.exists():
        raise FileNotFoundError(f'Baseline results not found: {baseline_path}')
    if not roi_path.exists():
        raise FileNotFoundError(f'ROI results not found: {roi_path}')

    with open(baseline_path, 'r') as f:
        df = flatten_runs(json.load(f))
    df = add_arch_names(df)
    df_base = df[df['Experiment'] == 'baseline'].copy()
    
    with open(roi_path, 'r') as f:
        df_roi = flatten_runs(json.load(f))
    df_roi = add_arch_names(df_roi)
    
    return df_base, df_roi

# =============================================================================
# FIGURE 1: EXTERNAL AUROC
# =============================================================================

def fig1_external_auroc(df_base, out_dir, ext_summary=None):
    """Dot-whisker plot of external AUROC by architecture (mean +/- SD across folds)."""
    print('Fig 1: External AUROC...')

    s = build_external_auc_table(df_base, ext_summary)
    s = s.dropna(subset=['AUC_mean'])
    if s.empty:
        print('  Skipped fig1_external_auroc (no AUROC data)')
        return
    s = s.sort_values('AUC_mean', ascending=False).reset_index(drop=True)
    s['Rank'] = range(1, len(s) + 1)

    n = len(s)
    base_color = '#4E79A7'

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(n)

    for i, row in s.iterrows():
        is_top = row['Rank'] == 1
        ax.errorbar(
            i,
            row['AUC_mean'],
            yerr=row['AUC_sd'],
            fmt='none',
            ecolor=base_color,
            elinewidth=ERR_LINEWIDTH,
            capsize=ERR_CAPSIZE,
            capthick=ERR_CAPTHICK,
            zorder=2,
        )
        ax.scatter(
            [i],
            [row['AUC_mean']],
            s=45 if is_top else 30,
            color=base_color,
            edgecolors='black' if is_top else 'none',
            linewidth=EDGEWIDTH_TOP if is_top else 0.0,
            zorder=3,
        )
        y_pos = row['AUC_mean'] + (row['AUC_sd'] if pd.notna(row['AUC_sd']) else 0) + 0.01
        ax.text(
            i,
            y_pos,
            f"{row['AUC_mean']:.3f}",
            ha='center',
            va='bottom',
            fontsize=9,
            fontweight='bold' if row['Rank'] == 1 else 'normal',
        )

    ax.set_xticks(x)
    ax.set_xticklabels(s['Architecture'], rotation=45, ha='right', fontsize=10)
    n_folds = None
    if 'Fold' in df_base.columns:
        n_folds = int(df_base['Fold'].nunique())
    ylab = 'External AUROC (mean +/- SD across folds'
    if n_folds:
        ylab += f'; n={n_folds}'
    ylab += ')'
    ax.set_ylabel(ylab, fontsize=11)
    ax.set_xlabel('Architecture', fontsize=11)
    ax.set_title('External Validation Performance by Architecture', fontsize=13)
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, axis='y', alpha=0.2)

    plt.tight_layout()
    save_fig(fig, out_dir / 'fig1_external_auroc')
    plt.close(fig)
    print('  OK fig1_external_auroc')

def figS1_external_bootstrap_ci(df_base, out_dir, ext_summary=None):
    """Dot-whisker plot of external AUROC with bootstrap 95% CI."""
    print('Fig S1: External AUROC bootstrap CI...')
    if ext_summary is None:
        print('  Skipped Fig S1 (no external bootstrap summary)')
        return

    s = build_external_auc_table(df_base, ext_summary)
    if s['AUC_ci_l'].isna().all() or s['AUC_ci_u'].isna().all():
        print('  Skipped Fig S1 (no CI columns)')
        return

    s = s.sort_values('AUC_mean', ascending=False).reset_index(drop=True)
    s['Rank'] = range(1, len(s) + 1)
    n = len(s)
    base_color = '#4E79A7'
    n_external = None
    if 'n_external' in ext_summary.columns:
        vals = ext_summary['n_external'].dropna().values
        if len(vals):
            n_external = int(vals[0])

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(n)

    for i, row in s.iterrows():
        if pd.isna(row['AUC_ci_l']) or pd.isna(row['AUC_ci_u']):
            continue
        is_top = row['Rank'] == 1
        yerr = [[row['AUC_mean'] - row['AUC_ci_l']], [row['AUC_ci_u'] - row['AUC_mean']]]
        ax.errorbar(
            i,
            row['AUC_mean'],
            yerr=yerr,
            fmt='none',
            ecolor=base_color,
            elinewidth=ERR_LINEWIDTH,
            capsize=ERR_CAPSIZE,
            capthick=ERR_CAPTHICK,
            zorder=2,
        )
        ax.scatter(
            [i],
            [row['AUC_mean']],
            s=45 if is_top else 30,
            color=base_color,
            edgecolors='black' if is_top else 'none',
            linewidth=EDGEWIDTH_TOP if is_top else 0.0,
            zorder=3,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(s['Architecture'], rotation=45, ha='right', fontsize=10)
    ylab = 'External AUROC (bootstrap 95% CI; image-level'
    if n_external:
        ylab += f'; n={n_external}'
    ylab += ')'
    ax.set_ylabel(ylab, fontsize=11)
    ax.set_xlabel('Architecture', fontsize=11)
    ax.set_title('External Validation Performance (Bootstrap CI)', fontsize=13)
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, axis='y', alpha=0.2)

    plt.tight_layout()
    save_fig(fig, out_dir / 'figS1_external_auroc_bootstrap_ci')
    plt.close(fig)
    print('  OK figS1_external_auroc_bootstrap_ci')

def figS2_ensemble_uplift(df_base, out_dir, ext_summary=None):
    """Dot plot of ensemble uplift (ensemble AUROC - mean single-fold AUROC)."""
    print('Fig S2: Ensemble uplift...')
    if ext_summary is None:
        print('  Skipped Fig S2 (no external bootstrap summary)')
        return

    s = build_external_auc_table(df_base, ext_summary)
    s = s.dropna(subset=['Ensemble_AUC'])
    if s.empty:
        print('  Skipped Fig S2 (no ensemble AUROC values)')
        return

    s['Uplift'] = s['Ensemble_AUC'] - s['AUC_mean']
    s = s.sort_values('Uplift', ascending=False).reset_index(drop=True)

    n = len(s)
    base_color = '#4E79A7'

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(n)

    ax.axhline(0.0, color='#888888', linewidth=1.2, zorder=1)
    is_top = s['Uplift'] == s['Uplift'].max()
    edgecolors = ['black' if t else 'none' for t in is_top]
    ax.scatter(x, s['Uplift'], s=45, c=base_color, edgecolors=edgecolors, linewidth=0.8, zorder=3)

    ax.set_xticks(x)
    ax.set_xticklabels(s['Architecture'], rotation=45, ha='right', fontsize=10)
    ax.set_ylabel('Ensemble uplift (AUROC)', fontsize=11)
    ax.set_xlabel('Architecture', fontsize=11)
    ax.set_title('Ensemble Uplift vs Mean Single-Fold AUROC', fontsize=13)
    ax.grid(True, axis='y', alpha=0.2)

    max_abs = max(abs(s['Uplift'].min()), abs(s['Uplift'].max()))
    max_abs = max(max_abs, 0.005)
    ax.set_ylim(-max_abs - 0.005, max_abs + 0.005)

    plt.tight_layout()
    save_fig(fig, out_dir / 'figS2_ensemble_uplift')
    plt.close(fig)
    print('  OK figS2_ensemble_uplift')

# =============================================================================
# FIGURE 2: GENERALIZATION GAP
# =============================================================================

def fig2_generalization(df_base, out_dir, ext_summary=None):
    """Two-panel scatter: full-scale + zoomed generalization gap."""
    print('Fig 2: Generalization...')

    val = df_base.groupby('Architecture', as_index=False).agg(
        Val_AUC=('Val_AUC', 'mean'),
    )
    ext = build_external_auc_table(df_base, ext_summary)[['Architecture', 'AUC_mean']]
    s = pd.merge(val, ext, on='Architecture', how='inner').rename(
        columns={'AUC_mean': 'External_AUC'}
    )
    s = s.dropna(subset=['Val_AUC', 'External_AUC'])
    if s.empty:
        print('  Skipped fig2_generalization_gap (no merged data)')
        return
    s = s.sort_values('External_AUC', ascending=False).reset_index(drop=True)
    s['Rank'] = range(1, len(s) + 1)
    s['Gap'] = s['External_AUC'] - s['Val_AUC']

    fig, (ax_full, ax_zoom) = plt.subplots(1, 2, figsize=(12, 6))

    # Full-scale panel
    lims_full = [0.0, 1.0]
    ax_full.plot(lims_full, lims_full, '-', color='#888888', linewidth=1.5, zorder=1)
    ax_full.fill_between(lims_full, lims_full, [lims_full[1], lims_full[1]], alpha=0.06, color='#A6CEE3', zorder=0)
    ax_full.fill_between(lims_full, [lims_full[0], lims_full[0]], lims_full, alpha=0.06, color='#FDBF6F', zorder=0)

    base_color = '#4E79A7'
    for _, row in s.iterrows():
        is_top = row['Rank'] == 1
        size = SCATTER_SIZE_TOP if row['Rank'] == 1 else SCATTER_SIZE_MID if row['Rank'] <= 3 else SCATTER_SIZE_BASE
        ax_full.scatter(
            row['Val_AUC'],
            row['External_AUC'],
            s=size,
            c=base_color,
            edgecolors='black' if is_top else 'none',
            linewidth=EDGEWIDTH_TOP if is_top else 0.0,
            alpha=SCATTER_ALPHA,
            zorder=5,
        )

    ax_full.set_xlabel('Internal AUROC (validation)', fontsize=11)
    ax_full.set_ylabel('External AUROC', fontsize=11)
    ax_full.set_title('(a) Full scale (0-1)', fontsize=12)
    ax_full.set_xlim(lims_full)
    ax_full.set_ylim(lims_full)
    ax_full.set_aspect('equal')
    ax_full.set_axisbelow(True)
    ax_full.grid(True, alpha=0.2)

    # Zoomed panel
    all_vals = np.concatenate([s['Val_AUC'].values, s['External_AUC'].values])
    zmin = max(0.0, np.floor((all_vals.min() - 0.02) * 100) / 100)
    zmax = min(1.0, np.ceil((all_vals.max() + 0.02) * 100) / 100)

    ax_zoom.plot([zmin, zmax], [zmin, zmax], '-', color='#888888', linewidth=1.5, zorder=1)
    ax_zoom.fill_between([zmin, zmax], [zmin, zmax], [zmax, zmax], alpha=0.06, color='#A6CEE3', zorder=0)
    ax_zoom.fill_between([zmin, zmax], [zmin, zmin], [zmin, zmax], alpha=0.06, color='#FDBF6F', zorder=0)

    for _, row in s.iterrows():
        is_top = row['Rank'] == 1
        size = SCATTER_SIZE_TOP if row['Rank'] == 1 else SCATTER_SIZE_MID if row['Rank'] <= 3 else SCATTER_SIZE_BASE
        ax_zoom.scatter(
            row['Val_AUC'],
            row['External_AUC'],
            s=size,
            c=base_color,
            edgecolors='black' if is_top else 'none',
            linewidth=EDGEWIDTH_TOP if is_top else 0.0,
            alpha=SCATTER_ALPHA,
            zorder=5,
        )
        num = ARCH_NUM.get(row['Architecture'])
        if num is not None:
            ax_zoom.text(
                row['Val_AUC'],
                row['External_AUC'],
                str(num),
                ha='center',
                va='center',
                fontsize=8,
                bbox=dict(facecolor='white', alpha=0.75, edgecolor='none', boxstyle='round,pad=0.15'),
                zorder=6,
            )

    ax_zoom.set_xlabel('Internal AUROC (validation)', fontsize=11)
    ax_zoom.set_ylabel('External AUROC', fontsize=11)
    ax_zoom.set_title('(b) Zoomed view', fontsize=12)
    ax_zoom.set_xlim([zmin, zmax])
    ax_zoom.set_ylim([zmin, zmax])
    ax_zoom.set_aspect('equal')
    ax_zoom.set_axisbelow(True)
    ax_zoom.grid(True, alpha=0.2)

    # Number legend (bottom-right of zoomed panel)
    legend_text = build_arch_number_legend()
    ax_zoom.text(
        0.98,
        0.02,
        legend_text,
        transform=ax_zoom.transAxes,
        ha='right',
        va='bottom',
        fontsize=7,
        bbox=dict(facecolor='white', alpha=0.85, edgecolor='none', boxstyle='round,pad=0.2'),
        zorder=10,
    )

    # Zoom window rectangle on full panel
    zoom_rect = Rectangle((zmin, zmin), zmax - zmin, zmax - zmin,
                          linewidth=1.5, edgecolor='#444444', facecolor='none', linestyle='--')
    ax_full.add_patch(zoom_rect)

    plt.tight_layout()
    save_fig(fig, out_dir / 'fig2_generalization_gap')
    plt.close(fig)
    print('  OK fig2_generalization_gap')

# =============================================================================
# FIGURE 3: CALIBRATION
# =============================================================================

def fig3_calibration(df_base, out_dir):
    """Paired-dot chart of ECE before/after calibration."""
    print('Fig 3: Calibration...')

    s = df_base.groupby('Architecture', as_index=False).agg(
        ECE_pre=('ECE_pre', 'mean'),
        ECE_pre_std=('ECE_pre', 'std'),
        ECE_post=('ECE_post', 'mean'),
        ECE_post_std=('ECE_post', 'std'),
    )
    s = s.dropna(subset=['ECE_pre', 'ECE_post'])
    if s.empty:
        print('  Skipped fig3_calibration (no ECE data)')
        return
    s['Improvement'] = s['ECE_pre'] - s['ECE_post']
    s = s.sort_values('Improvement', ascending=False).reset_index(drop=True)

    n = len(s)
    x = np.arange(n)
    offset = 0.18
    pre_color = '#5B8BD6'
    post_color = '#2E7D32'

    fig, ax = plt.subplots(figsize=(12, 5))

    for i, row in s.iterrows():
        ax.plot(
            [i - offset, i + offset],
            [row['ECE_pre'], row['ECE_post']],
            color='#999999',
            linewidth=1.5,
            zorder=1,
        )

    ax.errorbar(
        x - offset,
        s['ECE_pre'],
        yerr=s['ECE_pre_std'],
        fmt='o',
        color=pre_color,
        ecolor=pre_color,
        capsize=ERR_CAPSIZE_SMALL,
        elinewidth=ERR_LINEWIDTH,
        capthick=ERR_CAPTHICK,
        markersize=5,
        markeredgecolor=pre_color,
        markeredgewidth=0.0,
        label='Before calibration',
        zorder=3,
    )
    ax.errorbar(
        x + offset,
        s['ECE_post'],
        yerr=s['ECE_post_std'],
        fmt='o',
        color=post_color,
        ecolor=post_color,
        capsize=ERR_CAPSIZE_SMALL,
        elinewidth=ERR_LINEWIDTH,
        capthick=ERR_CAPTHICK,
        markersize=5,
        markeredgecolor=post_color,
        markeredgewidth=0.0,
        label='After calibration',
        zorder=3,
    )

    ax.set_xticks(x)
    ax.set_xticklabels(s['Architecture'], rotation=45, ha='right', fontsize=10)
    ax.set_ylabel('Expected Calibration Error (ECE)', fontsize=11)
    ax.set_xlabel('Architecture (sorted by improvement)', fontsize=11)
    ax.set_title('Calibration Improvement by Architecture', fontsize=13)
    ax.legend(loc='upper right', fontsize=9)

    y_max = max(s['ECE_pre'].max(), s['ECE_post'].max()) + 0.02
    ax.set_ylim(0.0, y_max)
    ax.grid(True, axis='y', alpha=0.2)

    plt.tight_layout()
    save_fig(fig, out_dir / 'fig3_calibration')
    plt.close(fig)
    print('  OK fig3_calibration')

# =============================================================================
# FIGURE 4: OPERATING POINTS
# =============================================================================

def fig4_operating_points(df_base, out_dir):
    """Paired-dot chart of sensitivity and specificity at the operating point."""
    print('Fig 4: Operating Points...')

    s = df_base.groupby('Architecture', as_index=False).agg(
        Sens_mean=('Sensitivity', 'mean'),
        Sens_std=('Sensitivity', 'std'),
        Spec_mean=('Specificity', 'mean'),
        Spec_std=('Specificity', 'std'),
        AUC_mean=('Test_AUC', 'mean'),
    )
    s = s.dropna(subset=['Sens_mean', 'Spec_mean', 'AUC_mean'])
    if s.empty:
        print('  Skipped fig4_operating_points (no operating point data)')
        return
    s = s.sort_values('AUC_mean', ascending=False).reset_index(drop=True)

    n = len(s)
    x = np.arange(n)
    offset = 0.18
    sens_color = '#1F77B4'
    spec_color = '#FF7F0E'

    fig, ax = plt.subplots(figsize=(12, 5))

    for i, row in s.iterrows():
        ax.plot(
            [i - offset, i + offset],
            [row['Sens_mean'], row['Spec_mean']],
            color='#999999',
            linewidth=1.5,
            zorder=1,
        )

    ax.errorbar(
        x - offset,
        s['Sens_mean'],
        yerr=s['Sens_std'],
        fmt='o',
        color=sens_color,
        ecolor=sens_color,
        capsize=ERR_CAPSIZE_SMALL,
        elinewidth=ERR_LINEWIDTH,
        capthick=ERR_CAPTHICK,
        markersize=5,
        markeredgecolor=sens_color,
        markeredgewidth=0.0,
        label='Sensitivity',
        zorder=3,
    )
    ax.errorbar(
        x + offset,
        s['Spec_mean'],
        yerr=s['Spec_std'],
        fmt='o',
        color=spec_color,
        ecolor=spec_color,
        capsize=ERR_CAPSIZE_SMALL,
        elinewidth=ERR_LINEWIDTH,
        capthick=ERR_CAPTHICK,
        markersize=5,
        markeredgecolor=spec_color,
        markeredgewidth=0.0,
        label='Specificity',
        zorder=3,
    )

    ax.set_xticks(x)
    ax.set_xticklabels(s['Architecture'], rotation=45, ha='right', fontsize=10)
    ax.set_ylabel('Metric Value', fontsize=11)
    ax.set_xlabel('Architecture (sorted by AUROC)', fontsize=11)
    ax.set_title('Operating Point Characteristics (validation threshold applied to external)', fontsize=13)
    ax.legend(loc='lower right', fontsize=9)

    ax.set_ylim(0.0, 1.0)
    ax.grid(True, axis='y', alpha=0.2)

    plt.tight_layout()
    save_fig(fig, out_dir / 'fig4_operating_points')
    plt.close(fig)
    print('  OK fig4_operating_points')

# =============================================================================
# FIGURE 5: THRESHOLD STABILITY
# =============================================================================

def fig5_threshold_stability(df_base, out_dir):
    """Box plots with individual points showing validation threshold variation across folds."""
    print('Fig 5: Threshold Stability...')

    cv_data = df_base.groupby('Architecture')['Val_Youden_Threshold'].agg(['mean', 'std', 'count'])
    cv_data = cv_data[cv_data['count'] > 0]
    if cv_data.empty:
        print('  Skipped fig5_threshold_stability (no validation thresholds)')
        return
    cv_data['CV'] = cv_data['std'] / cv_data['mean']
    cv_data = cv_data.sort_values('CV', ascending=True)
    arch_order = cv_data.index.tolist()
    cv_data['Rank'] = range(1, len(cv_data) + 1)

    n = len(arch_order)
    cmap = plt.get_cmap('cividis')
    color_vals = np.linspace(0.2, 0.85, n)
    colors = [mcolors.to_hex(cmap(v)) for v in color_vals]

    fig, ax = plt.subplots(figsize=(12, 6))

    # Get data for each architecture
    data_by_arch = []
    for arch in arch_order:
        vals = df_base[df_base['Architecture'] == arch]['Val_Youden_Threshold'].dropna().values
        data_by_arch.append(vals)

    # Draw box plots manually for full control
    box_width = 0.4
    cap_width = 0.15

    for i, (arch, vals, color) in enumerate(zip(arch_order, data_by_arch, colors)):
        q1 = np.percentile(vals, 25)
        q2 = np.percentile(vals, 50)
        q3 = np.percentile(vals, 75)
        v_min = vals.min()
        v_max = vals.max()

        # Box (IQR)
        box = plt.Rectangle((i - box_width/2, q1), box_width, q3 - q1,
                             facecolor=color, edgecolor=color, alpha=0.6, linewidth=2, zorder=3)
        ax.add_patch(box)

        # Median line
        ax.hlines(q2, i - box_width/2, i + box_width/2, colors='white', linewidth=2.5, zorder=4)

        # Whiskers
        ax.plot([i, i], [v_min, q1], color=color, linewidth=2.5, solid_capstyle='butt', zorder=2)
        ax.plot([i, i], [q3, v_max], color=color, linewidth=2.5, solid_capstyle='butt', zorder=2)

        # Caps
        ax.hlines(v_min, i - cap_width, i + cap_width, colors=color, linewidth=2.5, zorder=2)
        ax.hlines(v_max, i - cap_width, i + cap_width, colors=color, linewidth=2.5, zorder=2)

        # Individual points (perfectly aligned)
        ax.scatter(np.full(len(vals), i), vals, c=color, s=30,
                   alpha=0.9, edgecolors='none', linewidth=0.0, zorder=5)

    # CV annotations
    y_max = max(max(v) for v in data_by_arch) + 0.08
    y_text = min(0.98, y_max)
    for i, arch in enumerate(arch_order):
        cv = cv_data.loc[arch, 'CV']
        rank = cv_data.loc[arch, 'Rank']
        label = f'#{rank} CV={cv:.2f}' if rank <= 3 else f'CV={cv:.2f}'
        ax.text(i, y_text, label, ha='center', va='bottom', fontsize=9,
                fontweight='bold' if rank == 1 else 'normal', color='black')

    ax.set_xticks(range(n))
    ax.set_xticklabels(arch_order, rotation=45, ha='right', fontsize=10)
    ax.set_ylabel('Validation Youden Threshold', fontsize=11)
    ax.set_xlabel('Architecture (sorted by stability, most stable -> least)', fontsize=11)
    ax.set_title('Threshold Stability Across Folds\n(lower CV = more deployment-stable)', fontsize=13)
    ax.grid(True, axis='y', alpha=0.2)
    ax.set_xlim(-0.5, n - 0.5)
    ax.set_ylim(0.0, 1.0)

    plt.tight_layout()
    save_fig(fig, out_dir / 'fig5_threshold_stability')
    plt.close(fig)
    print('  OK fig5_threshold_stability')

# =============================================================================
# FIGURE 6: ROI HEATMAP
# =============================================================================

def fig6_roi_heatmap(df_roi, out_dir):
    """Heatmap of ROI ablation results."""
    print('Fig 6: ROI Heatmap...')

    exp_order = ['no_mask', 'baseline', 'uniform_5', 'uniform_7_5', 'uniform_10',
                 'uniform_15', 'uniform_20', 'bottom_5', 'bottom_10', 'bottom_15']
    exp_names = {
        'no_mask': 'No mask', 'baseline': '0%',
        'uniform_5': 'Unif+5%', 'uniform_7_5': 'Unif+7.5%',
        'uniform_10': 'Unif+10%', 'uniform_15': 'Unif+15%', 'uniform_20': 'Unif+20%',
        'bottom_5': 'Bot+5%', 'bottom_10': 'Bot+10%', 'bottom_15': 'Bot+15%'
    }

    df_r = df_roi.copy()
    df_r['Experiment'] = pd.Categorical(df_r['Experiment'], categories=exp_order, ordered=True)
    pivot = df_r.groupby(['Architecture', 'Experiment'], observed=False, as_index=False)['Test_AUC'].mean()
    pivot = pivot.pivot(index='Architecture', columns='Experiment', values='Test_AUC')

    if 'baseline' not in pivot.columns:
        print('  Skipped fig6_roi_heatmap (no baseline column)')
        return

    baseline_order = pivot['baseline'].sort_values(ascending=False).index
    pivot = pivot.loc[baseline_order]
    pivot = pivot[[c for c in exp_order if c in pivot.columns]]
    if pivot.empty:
        print('  Skipped fig6_roi_heatmap (no ROI data)')
        return

    fig, ax = plt.subplots(figsize=(14, 6))

    vmin = float(np.nanmin(pivot.values))
    vmax = float(np.nanmax(pivot.values))
    vmin = max(0.0, vmin - 0.005)
    vmax = min(1.0, vmax + 0.005)

    im = ax.imshow(pivot.values, aspect='auto', cmap='viridis', vmin=vmin, vmax=vmax)

    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            val = pivot.iloc[i, j]
            if np.isnan(val):
                continue
            weight = 'bold' if val == pivot.iloc[i, :].max() else 'normal'
            if vmax > vmin:
                norm = (val - vmin) / (vmax - vmin)
            else:
                norm = 0.5
            text_color = 'white' if norm < 0.35 else 'black'
            ax.text(j, i, f'{val:.3f}', ha='center', va='center', fontsize=9,
                    color=text_color, fontweight=weight)

    ax.set_xticks(np.arange(pivot.shape[1]))
    ax.set_xticklabels([exp_names.get(c, c) for c in pivot.columns], fontsize=9)
    ax.set_yticks(np.arange(pivot.shape[0]))
    ax.set_yticklabels(pivot.index, fontsize=10)
    ax.set_xlabel('ROI Strategy', fontsize=11)
    ax.set_ylabel('Architecture', fontsize=11)
    ax.set_title('ROI Ablation: External AUROC (bold = best for each architecture)', fontsize=13, pad=10)

    n_rows = pivot.shape[0]

    # Uniform expansion box (columns 2-6, NOT including 0%)
    uniform_rect = Rectangle((1.5, -0.5), 5, n_rows, linewidth=2.5,
                              edgecolor='#1F77B4', facecolor='none', linestyle='--')
    ax.add_patch(uniform_rect)
    ax.text(4, n_rows + 0.3, 'Uniform expansion', ha='center', va='top',
            fontsize=10, color='#1F77B4', fontweight='bold')

    # Bottom expansion box (columns 7-9)
    bottom_rect = Rectangle((6.5, -0.5), 3, n_rows, linewidth=2.5,
                             edgecolor='#FF7F0E', facecolor='none', linestyle='--')
    ax.add_patch(bottom_rect)
    ax.text(8, n_rows + 0.3, 'Bottom expansion', ha='center', va='top',
            fontsize=10, color='#FF7F0E', fontweight='bold')

    cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label('External AUROC', fontsize=11)

    plt.tight_layout()
    save_fig(fig, out_dir / 'fig6_roi_heatmap')
    plt.close(fig)
    print('  OK fig6_roi_heatmap')

    # Delta heatmap vs baseline (0%) if available
    if 'baseline' not in pivot.columns:
        print('  Skipped delta heatmap (no baseline column)')
        return

    delta = pivot.sub(pivot['baseline'], axis=0)
    vmax_delta = float(np.nanmax(np.abs(delta.values)))
    if not np.isfinite(vmax_delta) or vmax_delta == 0:
        vmax_delta = 0.01

    fig, ax = plt.subplots(figsize=(14, 6))
    im = ax.imshow(delta.values, aspect='auto', cmap='coolwarm', vmin=-vmax_delta, vmax=vmax_delta)

    for i in range(delta.shape[0]):
        for j in range(delta.shape[1]):
            val = delta.iloc[i, j]
            if np.isnan(val):
                continue
            color = 'white' if abs(val) > vmax_delta * 0.6 else 'black'
            ax.text(j, i, f'{val:+.3f}', ha='center', va='center', fontsize=9, color=color)

    ax.set_xticks(np.arange(delta.shape[1]))
    ax.set_xticklabels([exp_names.get(c, c) for c in delta.columns], fontsize=9)
    ax.set_yticks(np.arange(delta.shape[0]))
    ax.set_yticklabels(delta.index, fontsize=10)
    ax.set_xlabel('ROI Strategy', fontsize=11)
    ax.set_ylabel('Architecture', fontsize=11)
    ax.set_title('ROI Ablation: Delta AUROC vs baseline (0%)', fontsize=13, pad=10)

    # Uniform expansion box (columns 2-6, NOT including 0%)
    uniform_rect = Rectangle((1.5, -0.5), 5, delta.shape[0], linewidth=2.5,
                              edgecolor='#1F77B4', facecolor='none', linestyle='--')
    ax.add_patch(uniform_rect)
    ax.text(4, delta.shape[0] + 0.3, 'Uniform expansion', ha='center', va='top',
            fontsize=10, color='#1F77B4', fontweight='bold')

    # Bottom expansion box (columns 7-9)
    bottom_rect = Rectangle((6.5, -0.5), 3, delta.shape[0], linewidth=2.5,
                             edgecolor='#FF7F0E', facecolor='none', linestyle='--')
    ax.add_patch(bottom_rect)
    ax.text(8, delta.shape[0] + 0.3, 'Bottom expansion', ha='center', va='top',
            fontsize=10, color='#FF7F0E', fontweight='bold')

    cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label('Delta AUROC vs baseline', fontsize=11)

    plt.tight_layout()
    save_fig(fig, out_dir / 'fig6_roi_heatmap_delta')
    plt.close(fig)
    print('  OK fig6_roi_heatmap_delta')

# =============================================================================
# FIGURE 7: ROI BAR CHART
# =============================================================================

def fig7_roi_bar(df_roi, out_dir):
    """Dot plot of ROI strategy performance across architectures."""
    print('Fig 7: ROI Strategy Summary...')

    exp_order = ['no_mask', 'baseline', 'uniform_5', 'uniform_7_5', 'uniform_10',
                 'uniform_15', 'uniform_20', 'bottom_5', 'bottom_10', 'bottom_15']
    exp_names = {
        'no_mask': 'No mask', 'baseline': '0%',
        'uniform_5': 'Unif+5%', 'uniform_7_5': 'Unif+7.5%',
        'uniform_10': 'Unif+10%', 'uniform_15': 'Unif+15%', 'uniform_20': 'Unif+20%',
        'bottom_5': 'Bot+5%', 'bottom_10': 'Bot+10%', 'bottom_15': 'Bot+15%'
    }

    df_r = df_roi.copy()
    arch_means = df_r.groupby(['Architecture', 'Experiment'], as_index=False).agg(
        AUC=('Test_AUC', 'mean')
    )
    summary = arch_means.groupby('Experiment', as_index=False).agg(
        AUC_mean=('AUC', 'mean'),
        AUC_std=('AUC', 'std'),
        N=('AUC', 'count'),
    )
    summary['Experiment'] = pd.Categorical(summary['Experiment'], categories=exp_order, ordered=True)
    summary = summary.sort_values('Experiment').dropna(subset=['AUC_mean'])
    if summary.empty:
        print('  Skipped fig7_roi_bar (no ROI summary data)')
        return

    fig, ax = plt.subplots(figsize=(12, 5))

    x = np.arange(len(summary))
    dot_color = '#5B5B5B'

    for i, exp in enumerate(summary['Experiment']):
        vals = arch_means[arch_means['Experiment'] == exp]['AUC'].values
        vals = np.sort(vals)
        if len(vals) == 0:
            continue
        jitter = np.linspace(-0.18, 0.18, len(vals))
        ax.scatter(
            np.full(len(vals), i) + jitter,
            vals,
            s=28,
            color=dot_color,
            alpha=0.6,
            edgecolors='none',
            linewidth=0.0,
            zorder=2,
        )

    ax.errorbar(
        x,
        summary['AUC_mean'],
        yerr=summary['AUC_std'],
        fmt='o',
        color='black',
        ecolor='black',
        capsize=ERR_CAPSIZE,
        elinewidth=ERR_LINEWIDTH,
        capthick=ERR_CAPTHICK,
        markersize=5,
        markeredgecolor='black',
        markeredgewidth=0.6,
        zorder=4,
    )

    ax.set_xticks(x)
    ax.set_xticklabels([exp_names.get(str(e), str(e)) for e in summary['Experiment']],
                       rotation=45, ha='right', fontsize=10)
    ax.set_ylabel('External AUROC', fontsize=11)
    ax.set_xlabel('ROI Strategy', fontsize=11)
    ax.set_title('ROI Strategy Performance\n(dots = architectures, black = mean +/- SD)', fontsize=13)
    ax.grid(True, axis='y', alpha=0.2)
    ax.set_ylim(0.0, 1.0)

    # Uniform expansion shading (NOT including 0%)
    ax.axvspan(1.5, 6.5, alpha=0.08, color='#1F77B4')
    ax.text(4, 0.99, 'Uniform expansion', ha='center', va='top',
            fontsize=9, color='#1F77B4', fontweight='bold')

    # Bottom expansion shading
    ax.axvspan(6.5, 9.5, alpha=0.08, color='#FF7F0E')
    ax.text(8, 0.99, 'Bottom expansion', ha='center', va='top',
            fontsize=9, color='#FF7F0E', fontweight='bold')

    plt.tight_layout()
    save_fig(fig, out_dir / 'fig7_roi_bar')
    plt.close(fig)
    print('  OK fig7_roi_bar')

# =============================================================================
# FIGURE 8: EFFICIENCY VS PERFORMANCE
# =============================================================================

def fig8_efficiency(df_base, out_dir, ext_summary=None):
    """Scatter plot of epoch time vs external AUROC."""
    print('Fig 8: Efficiency...')

    s_time = df_base.groupby(['Model', 'Architecture'], as_index=False).agg(
        Epoch_sec=('Avg_Epoch_sec', 'mean')
    )
    s_auc = build_external_auc_table(df_base, ext_summary)
    s = pd.merge(s_time, s_auc, on=['Model', 'Architecture'], how='inner')
    s = s.dropna(subset=['AUC_mean', 'Epoch_sec'])
    if s.empty:
        print('  Skipped fig8_efficiency_vs_auroc (no data)')
        return
    s = s.sort_values('AUC_mean', ascending=False).reset_index(drop=True)
    s['Rank'] = range(1, len(s) + 1)

    n = len(s)
    base_color = '#4E79A7'

    fig, ax = plt.subplots(figsize=(10, 8))

    for i, row in s.iterrows():
        is_top = row['Rank'] == 1
        size = SCATTER_SIZE_TOP if row['Rank'] == 1 else SCATTER_SIZE_MID if row['Rank'] <= 3 else SCATTER_SIZE_BASE
        ax.scatter(
            row['Epoch_sec'],
            row['AUC_mean'],
            s=size,
            c=base_color,
            edgecolors='black' if is_top else 'none',
            linewidth=EDGEWIDTH_TOP if is_top else 0.0,
            alpha=SCATTER_ALPHA,
            zorder=5,
        )
        if pd.notna(row['AUC_sd']):
            ax.errorbar(
                row['Epoch_sec'],
                row['AUC_mean'],
                yerr=row['AUC_sd'],
                fmt='none',
                ecolor=base_color,
                elinewidth=ERR_LINEWIDTH,
                capsize=ERR_CAPSIZE,
                capthick=ERR_CAPTHICK,
                alpha=SCATTER_ALPHA,
                zorder=4,
            )
        num = ARCH_NUM.get(row['Architecture'])
        if num is not None:
            ax.annotate(
                str(num),
                (row['Epoch_sec'], row['AUC_mean']),
                textcoords='offset points',
                xytext=(6, 6),
                fontsize=8,
                bbox=dict(facecolor='white', alpha=0.75, edgecolor='none', boxstyle='round,pad=0.15'),
                zorder=6,
            )

    ax.set_xlabel('Mean Epoch Time (seconds)', fontsize=11)
    n_folds = None
    if 'Fold' in df_base.columns:
        n_folds = int(df_base['Fold'].nunique())
    ylab = 'External AUROC (mean +/- SD across folds'
    if n_folds:
        ylab += f'; n={n_folds}'
    ylab += ')'
    ax.set_ylabel(ylab, fontsize=11)
    ax.set_title('Efficiency vs Performance Trade-off', fontsize=13)
    ax.set_axisbelow(True)
    ax.grid(True, alpha=0.2)

    x_min = s['Epoch_sec'].min() - 1.5
    x_max = s['Epoch_sec'].max() + 1.5
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(0.0, 1.0)

    plt.tight_layout()
    save_fig(fig, out_dir / 'fig8_efficiency_vs_auroc')
    plt.close(fig)
    print('  OK fig8_efficiency_vs_auroc')

# =============================================================================
# MAIN
# =============================================================================

def main():
    ap = argparse.ArgumentParser(description='Generate benchmark figures.')
    ap.add_argument('--baseline', default=str(DEFAULT_BASELINE_PATH), help='Path to baseline JSON results.')
    ap.add_argument('--roi', default=str(DEFAULT_ROI_PATH), help='Path to ROI ablation JSON results.')
    ap.add_argument('--external_bootstrap', default=str(DEFAULT_EXTERNAL_BOOTSTRAP_PATH),
                    help='Path to external bootstrap summary CSV (optional).')
    ap.add_argument('--out_dir', default=str(DEFAULT_OUT_DIR), help='Output directory for figures.')
    ap.add_argument('--skip_supplementary', action='store_true', help='Skip supplementary figures (S1/S2).')
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print('Loading data...')
    df_base, df_roi = load_data(args.baseline, args.roi)
    ext_summary = load_external_bootstrap_summary(args.external_bootstrap)
    print(f'  Loaded {len(df_base)} baseline runs, {len(df_roi)} ROI runs')

    print('\nGenerating figures...')
    fig1_external_auroc(df_base, out_dir, ext_summary)
    fig2_generalization(df_base, out_dir, ext_summary)
    fig3_calibration(df_base, out_dir)
    fig4_operating_points(df_base, out_dir)
    fig5_threshold_stability(df_base, out_dir)
    fig6_roi_heatmap(df_roi, out_dir)
    fig7_roi_bar(df_roi, out_dir)
    fig8_efficiency(df_base, out_dir, ext_summary)

    if not args.skip_supplementary:
        figS1_external_bootstrap_ci(df_base, out_dir, ext_summary)
        figS2_ensemble_uplift(df_base, out_dir, ext_summary)

    print(f'\nAll figures saved to {out_dir}')
    print('Formats: PNG (300 DPI) and PDF')

if __name__ == '__main__':
    main()
