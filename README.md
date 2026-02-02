# Deep Learning for Breast Ultrasound Classification: A Rigorous Benchmarking Framework

This repository contains the code and experimental framework for my undergraduate dissertation at the University of Buckingham, supervised by Prof Hongbo Du and Dr Naseer Al-Jawad.

## Abstract

Breast cancer remains the most commonly diagnosed malignancy amongst women worldwide. Ultrasound imaging serves as a critical adjunct modality to mammography, particularly for patients with dense breast tissue. This project presents a rigorous benchmarking framework addressing methodological gaps in existing research through systematic evaluation of **ten deep learning architectures** spanning classic CNNs, modern efficient designs, and vision transformers.

### Key Findings
- **Swin Transformer** achieved the best external validation performance (AUC = 0.907, balanced accuracy = 0.825)
- ROI preprocessing with uniform border expansion provided modest gains over strict lesion cropping (+1.4% balanced accuracy)
- Temperature scaling universally improved calibration, reducing expected calibration error by 0.5-3.0%
- Results derived from **333 experimental runs**

## Architectures Benchmarked

| Architecture | Type |
|-------------|------|
| VGG16 | Classic CNN |
| ResNet50 | Classic CNN |
| DenseNet121 | Classic CNN |
| ConvNeXt-Tiny | Modern CNN |
| MobileNetV3-Small | Efficient CNN |
| EfficientNet-B0 | Efficient CNN |
| Swin-T | Vision Transformer |
| MaxViT-T | Vision Transformer |
| DeiT-Tiny | Vision Transformer |
| ViT-Tiny | Vision Transformer |

## Repository Structure

```
.
├── breast_us_benchmark_final_v3.py    # Main benchmarking script
├── external_auc_bootstrap.py          # External validation with bootstrap CIs
├── make_all_figures_final.py          # Generate all figures for the report
├── dataset_config.json                # Dataset path configuration
├── runners/                           # Individual model runner scripts
│   ├── artifact_resnet50_runner.py
│   ├── artifact_convnext_runner.py
│   └── ...
├── artifact_detection/                # Artifact detection pipeline
│   ├── yolo_ui_predict_to_masks.py
│   ├── dash_unet_predict_batched.py
│   └── ...
├── results/                           # Experimental results and metrics
├── appendix/                          # Supplementary materials
└── breastdataset_NORMALIZED/          # Processed dataset (see Data section)
```

## Data

The framework uses five publicly available breast ultrasound datasets pooled for internal training/validation:
- BUSI (Al-Dhabyani et al., 2020)
- BUS-BRA (Gomez-Flores et al., 2023)
- Dataset B (Yap et al., 2018)
- BUS-UC (Gómez-Flores & Ruiz-Ortega, 2020)
- UDIAT (Yap et al., 2018)

Two additional datasets reserved for external validation:
- OASBUD
- STU

The `breastdataset_NORMALIZED/` folder contains the processed and standardised versions of these datasets.

## Model Checkpoints

**Model checkpoints (~22GB) are not included in this repository due to size constraints.**

If you would like access to the trained model weights for reproducibility or further research, please [open a GitHub Issue](../../issues/new) with:
- Your name and affiliation
- Intended use case
- Which specific checkpoints you need

I will respond with download instructions.

## Requirements

```bash
pip install numpy>=1.24.0,<2.0.0 pandas scikit-learn opencv-python timm scipy matplotlib pillow torch torchvision
```

The main script will also attempt to install dependencies automatically.

## Usage

### Running the Benchmark

```bash
# Run full benchmark (10 models)
python breast_us_benchmark_final_v3.py --mode benchmark

# Run ROI ablation study
python breast_us_benchmark_final_v3.py --mode ablation

# Run both
python breast_us_benchmark_final_v3.py --mode both
```

### External Validation

```bash
python external_auc_bootstrap.py
```

### Generate Figures

```bash
python make_all_figures_final.py
```

## Configuration

Edit `dataset_config.json` to configure dataset paths for your environment.

## Methodology Highlights

- **Patient-level data splitting** using StratifiedGroupKFold where patient IDs are available
- **Artifact-aware preprocessing** with configurable contamination thresholds
- **Temperature scaling** for probability calibration
- **Bootstrap confidence intervals** for all metrics
- **External validation** on geographically distinct datasets

## Citation

If you use this code in your research, please cite:

```
@thesis{raza2026breast,
  author = {Raza, Ahmad Hasan},
  title = {Deep Learning for Breast Ultrasound Classification: A Rigorous Benchmarking Framework},
  school = {University of Buckingham},
  year = {2026},
  type = {Bachelor's Thesis}
}
```

## License

This project is for academic research purposes. Please contact the author for commercial use inquiries.

## Author

Ahmad Hasan Raza
University of Buckingham
Student ID: 2311531
