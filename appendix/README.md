# Appendix Assets

This folder contains figures and tables generated for the dissertation appendix.

## Figures
- fig_A1_bus_uc_sensitivity_auc.png: Validation AUROC with/without BUS_UC.
- fig_B1_multi_lesion_impact_busi.png: BUSI multi-lesion impact on AUROC.
- fig_C1_roi_ablation_table.png: ROI ablation AUROC table (mean +/- SD, significance).
- fig_D1_bootstrap_histograms.png: Bootstrap AUROC histograms by architecture.
- fig_D2_ensemble_vs_foldmean.png: Ensemble vs fold-mean AUROC with 95% CI.
- fig_E1_external_by_dataset.png: External AUROC by dataset (BUSI vs QAMEBI).

## Tables
- table_A1_bus_uc_sensitivity_metrics.csv
- table_B1_multi_lesion_list.csv
- table_B2_multi_lesion_impact_external.csv
- table_B3_multi_lesion_impact_uclm_internal.csv
- table_C1_roi_ablation_auc.csv
- table_D1_bootstrap_summary.csv
- table_E1_external_by_dataset.csv

## Documentation
- generative_ai_usage.md (placeholder prompts; replace with actual usage)
- reproducibility.md, hardware_specs.txt, hyperparameters.csv, software_versions.csv

## Regeneration
Run:
```
python appendix/build_appendix_assets.py
```
