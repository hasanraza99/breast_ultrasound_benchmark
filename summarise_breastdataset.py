import os

ROOT_DIR = "/workspace/breastdataset_NORMALIZED"
OUT_FILE = "breastdataset_NORMALIZED_structure_sample.txt"

def main():
    with open(OUT_FILE, "w", encoding="utf-8") as f:
        for dirpath, dirnames, filenames in os.walk(ROOT_DIR):
            # Make path nicer (relative to root)
            rel_dir = os.path.relpath(dirpath, ROOT_DIR)
            if rel_dir == ".":
                rel_dir = "/"

            f.write(f"Folder: {rel_dir}\n")
            f.write(f"Full path: {dirpath}\n")

            if dirnames:
                f.write("Subfolders: " + ", ".join(sorted(dirnames)) + "\n")
            else:
                f.write("Subfolders: (none)\n")

            f.write(f"Total files: {len(filenames)}\n")

            if filenames:
                filenames_sorted = sorted(filenames)
                sample = filenames_sorted[:3]  # first 3 as a sample
                f.write("Sample files:\n")
                for name in sample:
                    f.write(f"  - {name}\n")
            else:
                f.write("Sample files: (no files)\n")

            f.write("\n")

    print(f"Summary written to: {OUT_FILE}")

if __name__ == "__main__":
    main()
