# Proposed Changes to Report v4 Final

## Executive Summary

This document contains a comprehensive review of "Deep Learning for Breast Ultrasound Classification: A Rigorous Benchmarking Framework" (Report v4 Final). All proposed changes have been cross-referenced against the source data files. The review identifies **1 major conceptual error**, **2 moderate errors**, and **several minor numerical corrections**.

---

## MAJOR ERROR: Section 4.2 - Generalisation Gap Misinterpretation

### Location
Section 4.2 (Generalisation: Internal Versus External Performance), page ~23

### Current Text
> "Swin-T demonstrated the smallest absolute gap (internal: 0.923, external: 0.907, Δ = 0.016), whilst ResNet-50 exhibited a larger differential (internal: 0.891, external: 0.852, Δ = 0.039)."

### Issue
The values labelled as "internal" (0.923 for Swin-T, 0.891 for ResNet-50) are **not** internal validation metrics. These values correspond to `ensemble_auc` from the bootstrap summary file, which represents an **ensemble of all 5 fold models making predictions on the external test set**. Both metrics are therefore external, not internal.

The true internal validation AUC values (from `Val_AUC` column, averaged across folds) are:
- Swin-T: **0.870** (not 0.923)
- ResNet-50: **0.843** (not 0.891)

### Corrected Text
Option A - Reframe as ensemble vs fold-averaged external comparison:
> "When comparing ensemble predictions to fold-averaged predictions on external data, Swin-T demonstrated consistent performance (ensemble: 0.924, fold-mean: 0.907), whilst ResNet-50 showed greater variability between aggregation methods (ensemble: 0.888, fold-mean: 0.852)."

Option B - Use actual internal validation values:
> "All architectures exhibited performance degradation on external data relative to internal validation. Swin-T showed a moderate gap (internal: 0.870, external: 0.907—though notably, external performance exceeded internal in this case, likely due to favourable class distribution). ResNet-50 exhibited similar patterns (internal: 0.843, external: 0.852)."

### Impact
This correction affects the core narrative of Section 4.2 and Figure 4.2. The finding that "all architectures exhibited some degree of performance degradation on external data" is actually **contradicted** by the data—most architectures performed better on external data than internal validation. This may reflect the external test set's characteristics rather than true generalisation behaviour.

### Recommendation
Rewrite Section 4.2 to accurately characterise the internal-external relationship. Consider whether the internal validation metrics are computed differently (e.g., per-fold subsets vs aggregated), and clarify the methodology.

---

## MODERATE ERROR 1: Confidence Intervals in Table 4.1

### Location
Table 4.1 (External test performance across ten architectures), page ~21-22

### Issue
The 95% bootstrap confidence intervals reported in Table 4.1 do not precisely match the values in `external_auc_bootstrap_summary.csv`.

### Current Values vs Actual Values

| Model | Report CI | Actual CI | Discrepancy |
|-------|-----------|-----------|-------------|
| Swin-T | [0.885, 0.927] | [0.887, 0.924] | Lower bound -0.002, Upper +0.003 |
| ConvNeXt-T | [0.871, 0.916] | [0.874, 0.913] | Lower bound -0.003, Upper +0.003 |
| DeiT-T | [0.869, 0.915] | [0.873, 0.911] | Lower bound -0.004, Upper +0.004 |
| VGG-16 | [0.851, 0.898] | [0.854, 0.895] | Lower bound -0.003, Upper +0.003 |
| RegNetY-008 | [0.848, 0.898] | [0.854, 0.894] | Lower bound -0.006, Upper +0.004 |
| EfficientNet-B0 | [0.848, 0.897] | [0.852, 0.892] | Lower bound -0.004, Upper +0.005 |
| DenseNet-121 | [0.847, 0.897] | [0.851, 0.893] | Lower bound -0.004, Upper +0.004 |
| MaxViT-T | [0.845, 0.894] | [0.849, 0.890] | Lower bound -0.004, Upper +0.004 |
| MobileNetV3-S | [0.839, 0.888] | [0.843, 0.885] | Lower bound -0.004, Upper +0.003 |
| ResNet-50 | [0.826, 0.877] | [0.830, 0.872] | Lower bound -0.004, Upper +0.005 |

### Corrected Table 4.1 (AUROC with CI column only)

| Model | AUROC [95% CI] |
|-------|----------------|
| Swin-T | **0.907** [0.887, 0.924] |
| ConvNeXt-T | 0.894 [0.874, 0.913] |
| DeiT-T | 0.894 [0.873, 0.911] |
| VGG-16 | 0.875 [0.854, 0.895] |
| RegNetY-008 | 0.875 [0.854, 0.894] |
| EfficientNet-B0 | 0.873 [0.852, 0.892] |
| DenseNet-121 | 0.873 [0.851, 0.893] |
| MaxViT-T | 0.870 [0.849, 0.890] |
| MobileNetV3-S | 0.865 [0.843, 0.885] |
| ResNet-50 | 0.852 [0.830, 0.872] |

---

## MODERATE ERROR 2: Table 4.2 - Threshold Values

### Location
Table 4.2 (Error profile summary across ten architectures), page ~25

### Issue
Several Youden threshold values in Table 4.2 do not match the computed means from the source data.

### Current Values vs Actual Values

| Model | Report Threshold | Actual Threshold | Discrepancy |
|-------|------------------|------------------|-------------|
| Swin-T | 0.304 ± 0.030 | 0.306 ± 0.032 | Minor |
| ConvNeXt-T | 0.466 ± 0.117 | 0.468 ± 0.121 | Minor |
| DeiT-T | **0.462 ± 0.155** | **0.497 ± 0.080** | **Significant** |
| VGG-16 | **0.488 ± 0.074** | **0.427 ± 0.082** | **Significant** |
| RegNetY-008 | **0.340 ± 0.072** | **0.420 ± 0.092** | **Significant** |
| EfficientNet-B0 | **0.454 ± 0.092** | **0.486 ± 0.079** | **Moderate** |
| DenseNet-121 | 0.376 ± 0.075 | 0.365 ± 0.112 | Minor |
| MaxViT-T | **0.376 ± 0.082** | **0.467 ± 0.105** | **Significant** |
| MobileNetV3-S | **0.402 ± 0.079** | **0.475 ± 0.099** | **Significant** |
| ResNet-50 | 0.466 ± 0.148 | 0.462 ± 0.157 | Minor |

### Note
The confusion matrix values (TN, FP, FN, TP) in Table 4.2 are **correct** and match the source data. Only the threshold values require correction.

### Corrected Table 4.2

| Model | Threshold | TN | FP | FN | TP |
|-------|-----------|-----|-----|-----|-----|
| Swin-T | 0.306 ± 0.032 | 441 | 122 | 45 | 289 |
| ConvNeXt-T | 0.468 ± 0.121 | 457 | 106 | 65 | 269 |
| DeiT-T | 0.497 ± 0.080 | 426 | 137 | 50 | 284 |
| VGG-16 | 0.427 ± 0.082 | 444 | 119 | 70 | 264 |
| RegNetY-008 | 0.420 ± 0.092 | 390 | 173 | 43 | 291 |
| EfficientNet-B0 | 0.486 ± 0.079 | 438 | 125 | 69 | 265 |
| DenseNet-121 | 0.365 ± 0.112 | 394 | 169 | 49 | 285 |
| MaxViT-T | 0.467 ± 0.105 | 420 | 143 | 56 | 278 |
| MobileNetV3-S | 0.475 ± 0.099 | 422 | 141 | 63 | 271 |
| ResNet-50 | 0.462 ± 0.157 | 441 | 122 | 83 | 251 |

---

## MINOR ERROR 1: Abstract - Balanced Accuracy Improvement

### Location
Abstract, page ~2

### Current Text
> "Region-of-interest preprocessing with uniform border expansion appeared to provide modest gains over strict lesion cropping, improving aggregate balanced accuracy by 1.4%"

### Actual Value
From ROI ablation data:
- Baseline (0% expansion): 0.803 balanced accuracy
- Best (uniform-7.5%): 0.815 balanced accuracy
- Improvement: **1.2%** (not 1.4%)

### Corrected Text
> "Region-of-interest preprocessing with uniform border expansion appeared to provide modest gains over strict lesion cropping, improving aggregate balanced accuracy by 1.2%"

---

## MINOR ERROR 2: Section 4.3 - False Negative Percentage

### Location
Section 4.3 (Confusion Matrices and Error Profiles), page ~24

### Current Text
> "RegNetY-008 produced the fewest false negatives (FN = 43, representing 12.1% of malignant cases)"

### Actual Calculation
FN = 43, Total malignant cases = 334
Percentage = 43/334 = **12.9%** (not 12.1%)

### Corrected Text
> "RegNetY-008 produced the fewest false negatives (FN = 43, representing 12.9% of malignant cases)"

---

## MINOR ERROR 3: Section 4.3 - Difference in Missed Malignancies

### Location
Section 4.3, page ~24

### Current Text
> "...this differential of approximately 38 additional missed malignancies per external test cohort..."

### Actual Calculation
ResNet-50 FN = 83, RegNetY-008 FN = 43
Difference = **40** (not ~38)

### Corrected Text
> "...this differential of approximately 40 additional missed malignancies per external test cohort..."

---

## VERIFIED CORRECT ITEMS

The following reported values were verified as accurate:

1. **Table 4.1 - Core Metrics**: All AUROC, balanced accuracy, sensitivity, specificity, and ECE mean ± SD values match the source data.

2. **Table 4.2 - Confusion Matrix Values**: TN, FP, FN, TP counts are all correct.

3. **Table 4.3 - Training Efficiency Metrics**: All training time values match exactly.

4. **Table 4.8 - Artifact Filtering Sensitivity**: All AUROC values across thresholds match exactly.

5. **ROI Ablation Results**: The ranking of strategies and claims about optimal expansion (uniform-7.5%, bottom-5%, bottom-10% as top performers) are accurate.

6. **Section 5.3 - Clinical Scenario Calculation**: The calculation of 67 vs 125 missed cancers (difference of 58) in a 10,000-patient population is correct.

7. **Calibration ECE Values**: Pre- and post-temperature scaling ECE values are accurate.

---

## APPENDIX RECOMMENDATIONS

Based on the review, the following items should be included in the Appendix:

### A. Sensitivity Analysis for BUS_UC Dataset
The report mentions this in Section 3.1 but should include:
- Validation metrics recomputed after excluding BUS_UC samples
- Cross-validation results with BUS_UC restricted to training folds only
- Comparison tables showing impact on main findings

### B. Multi-Lesion Image Analysis
Per Section 3.1.2, should include:
- List of affected images (16 from BUS-UCLM, 17 from BUSI)
- Statistical impact assessment on reported metrics
- Confirmation of negligible effect on conclusions

### C. Full ROI Ablation Heatmap Data
Supplement Figure 4.6 with:
- Complete numerical table of AUROC values for all 50 architecture × ROI combinations
- Standard deviations across folds for each cell
- Statistical significance indicators

### D. Bootstrap Distribution Visualisations
Supplement the confidence intervals with:
- Histograms of bootstrap AUROC distributions for each architecture
- Ensemble vs fold-mean comparison plots

### E. Per-Dataset External Validation Breakdown
Disaggregate external results by:
- BUSI (n=665) performance per architecture
- QAMEBI (n=232) performance per architecture
- Discussion of dataset-specific patterns

### F. Generative AI Usage Documentation
As required by the declaration, document:
- Specific prompts used for learning assistance
- Code refinement examples
- Research summarisation outputs

### G. Reproducibility Information
Include:
- Full hyperparameter tables
- Random seeds used
- Software versions (PyTorch, timm, etc.)
- Hardware specifications

---

## FIGURE REVIEW

Figures located in `benchmark_figures_v5/` were reviewed:

| Figure | File | Status |
|--------|------|--------|
| Fig 4.1 - External AUROC Bar Chart | fig1_external_auroc.png | Should be regenerated to reflect CI corrections |
| Fig 4.2 - Generalisation Gap | fig2_generalization_gap.png | **Requires major revision** (see Major Error above) |
| Fig 4.3 - Calibration | fig3_calibration.png | Appears consistent with data |
| Fig 4.4 - Operating Points | fig4_operating_points.png | Threshold values may need update |
| Fig 4.5 - Threshold Stability | fig5_threshold_stability.png | Should be verified against corrected thresholds |
| Fig 4.6 - ROI Heatmap | fig6_roi_heatmap.png | Appears consistent with data |
| Fig 4.7 - ROI Bar Chart | fig7_roi_bar.png | Appears consistent with data |
| Fig 4.8 - Efficiency vs AUROC | fig8_efficiency_vs_auroc.png | Appears consistent with data |

---

## SUMMARY OF REQUIRED CHANGES

| Priority | Section | Change Required |
|----------|---------|-----------------|
| **Critical** | 4.2 | Rewrite generalisation gap analysis with correct internal values |
| **Critical** | Fig 4.2 | Regenerate or remove generalisation scatter plot |
| **High** | Table 4.1 | Update confidence intervals |
| **High** | Table 4.2 | Update threshold values |
| **Medium** | Abstract | Change 1.4% to 1.2% |
| **Low** | Section 4.3 | Change 12.1% to 12.9% |
| **Low** | Section 4.3 | Change ~38 to ~40 |

---

*Document generated: 2026-02-02*
*Cross-referenced against source files in `C:\Users\ahraz\Downloads\workspace\results\`*
