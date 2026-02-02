#!/usr/bin/env python3
import argparse
from pathlib import Path

import pandas as pd


def print_quantiles(name: str, series: pd.Series, quantiles=None):
    if quantiles is None:
        quantiles = [0.0, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1.0]
    qvals = series.quantile(quantiles)
    print(f"\n{name} quantiles:")
    for q, v in qvals.items():
        print(f"  q{q:4.2f}: {v:.6f}")


def main():
    parser = argparse.ArgumentParser(
        description="Summarise overlap metrics and simulate drop thresholds (overall + by class), with CSV export."
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("/workspace/ui_dash_vs_lesion_overlap.csv"),
        help="Overlap CSV path.",
    )
    parser.add_argument(
        "--thresholds",
        type=float,
        nargs="*",
        default=[0.0, 0.01, 0.02, 0.05, 0.10, 0.15, 0.20],
        help="Candidate thresholds on frac_lesion_covered.",
    )
    parser.add_argument(
        "--out-overall",
        type=Path,
        default=Path("/workspace/overlap_threshold_sweep_overall.csv"),
        help="Output CSV for overall threshold sweep.",
    )
    parser.add_argument(
        "--out-by-class",
        type=Path,
        default=Path("/workspace/overlap_threshold_sweep_by_class.csv"),
        help="Output CSV for per-class threshold sweep.",
    )

    args = parser.parse_args()
    thresholds = sorted(set(args.thresholds))

    df = pd.read_csv(args.csv)

    # Only consider rows with a lesion mask (exclude BUSI normals with no lesion)
    df = df[df["lesion_path"].notna() & (df["lesion_path"] != "")]
    n_total = len(df)
    print(f"[INFO] Loaded {n_total} pairs with lesion masks.")

    # Basic global stats (printed only)
    for col in ["iou", "frac_lesion_covered", "frac_union_inside_lesion"]:
        if col in df.columns:
            print(f"\n=== Global stats for {col} ===")
            print(f"  mean:   {df[col].mean():.6f}")
            print(f"  median: {df[col].median():.6f}")
            print(f"  std:    {df[col].std():.6f}")
            print_quantiles(col, df[col])

    # === Threshold sweep (overall) ===
    overall_rows = []
    print("\n=== Threshold sweep on frac_lesion_covered (overall) ===")
    for th in thresholds:
        dropped = (df["frac_lesion_covered"] > th)
        n_drop = int(dropped.sum())
        pct_drop = 100.0 * n_drop / n_total
        print(f"  th={th:5.3f}: drop {n_drop:4d}/{n_total} ({pct_drop:5.2f}%)")
        overall_rows.append(
            {
                "threshold": th,
                "n_total": n_total,
                "n_drop": n_drop,
                "pct_drop": pct_drop,
            }
        )

    df_overall = pd.DataFrame(overall_rows)
    args.out_overall.parent.mkdir(parents=True, exist_ok=True)
    df_overall.to_csv(args.out_overall, index=False)
    print(f"\n[OK] Overall threshold sweep CSV written to: {args.out_overall}")

    # === Threshold sweep by class (benign/malignant/normal) ===
    by_class_rows = []
    print("\n=== Threshold sweep by class (benign/malignant/normal) ===")
    for cls, gc in df.groupby("class"):
        n_cls = len(gc)
        print(f"\n[CLASS] {cls} (n={n_cls})")
        for th in thresholds:
            dropped = (gc["frac_lesion_covered"] > th)
            n_drop = int(dropped.sum())
            pct_drop = 100.0 * n_drop / n_cls if n_cls > 0 else 0.0
            print(f"  th={th:5.3f}: drop {n_drop:4d}/{n_cls} ({pct_drop:5.2f}%)")
            by_class_rows.append(
                {
                    "class": cls,
                    "threshold": th,
                    "n_total": n_cls,
                    "n_drop": n_drop,
                    "pct_drop": pct_drop,
                }
            )

    df_by_class = pd.DataFrame(by_class_rows)
    args.out_by_class.parent.mkdir(parents=True, exist_ok=True)
    df_by_class.to_csv(args.out_by_class, index=False)
    print(f"\n[OK] Per-class threshold sweep CSV written to: {args.out_by_class}")


if __name__ == "__main__":
    main()
