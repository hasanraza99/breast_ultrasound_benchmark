# save as make_bus_uclm_patient_ids.py

from pathlib import Path
import csv

# Directories to scan
benign_dir = Path("/workspace/breastdataset_NORMALIZED/breastdataset_NORMALIZED/internal/BUS-UCLM/benign/images")
malignant_dir = Path("/workspace/breastdataset_NORMALIZED/breastdataset_NORMALIZED/internal/BUS-UCLM/malignant/images")

out_csv = Path("bus_uclm_patient_ids.csv")

rows = []

for folder in [benign_dir, malignant_dir]:
    # adjust pattern if you have non-PNGs
    for img_path in sorted(folder.glob("*.png")):
        file_name = img_path.name              # e.g. "alwi_000.png"
        patient_id = file_name[:4]             # e.g. "alwi"
        rows.append((file_name, patient_id))

# write CSV
with out_csv.open("w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["file_name", "patient_id"])
    writer.writerows(rows)

print(f"Wrote {len(rows)} rows to {out_csv}")
