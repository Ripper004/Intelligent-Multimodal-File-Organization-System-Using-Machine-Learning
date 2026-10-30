import os
import csv
from pathlib import Path
from datetime import datetime

# Paths
INPUT_DIR = Path("data/raw_files")
OUTPUT_DIR = Path("data/metadata")
OUTPUT_FILE = OUTPUT_DIR / "file_index.csv"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def infer_file_type(extension):
    if extension in [".pdf", ".docx", ".txt", ".pptx"]:
        return "document"
    elif extension in [".jpg", ".jpeg", ".png", ".bmp"]:
        return "image"
    elif extension in [".mp4", ".mkv", ".avi", ".mov"]:
        return "video"
    else:
        return "other"

rows = []

for file in INPUT_DIR.iterdir():
    if file.is_file():
        stats = file.stat()
        rows.append([
            file.name,
            str(file.resolve()),
            file.suffix.lower(),
            infer_file_type(file.suffix.lower()),
            round(stats.st_size / 1024, 2),  # size in KB
            datetime.fromtimestamp(stats.st_ctime),
            datetime.fromtimestamp(stats.st_mtime)
        ])

with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow([
        "filename",
        "filepath",
        "extension",
        "file_type",
        "size_kb",
        "created_time",
        "modified_time"
    ])
    writer.writerows(rows)

print(f"[SUCCESS] Indexed {len(rows)} files into {OUTPUT_FILE}")
