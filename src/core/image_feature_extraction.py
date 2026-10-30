# step3_2_image_embeddings.py

import os
import torch
import clip
import pandas as pd
from PIL import Image
from tqdm import tqdm

# -----------------------------
# PATHS (PIPELINE-CORRECT)
# -----------------------------
INPUT_DIR = "data/input_files"
OUTPUT_CSV = "data/features/image_features.csv"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# -----------------------------
# IMAGE EXTENSIONS
# -----------------------------
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")

# -----------------------------
# LOAD CLIP MODEL
# -----------------------------
print("[INFO] Loading CLIP ViT-B/32...")
model, preprocess = clip.load("ViT-B/32", device=DEVICE)
model.eval()

records = []

print("[INFO] Scanning input_files for images...")

for file_name in tqdm(os.listdir(INPUT_DIR)):
    if not file_name.lower().endswith(IMAGE_EXTENSIONS):
        continue

    file_path = os.path.join(INPUT_DIR, file_name)

    try:
        image = Image.open(file_path).convert("RGB")
        image_tensor = preprocess(image).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            embedding = model.encode_image(image_tensor)
            embedding = embedding / embedding.norm(dim=-1, keepdim=True)

        embedding = embedding.cpu().numpy().flatten()

        record = {
            "file_name": file_name,
            "file_path": file_path
        }

        for i, val in enumerate(embedding):
            record[f"img_emb_{i}"] = val

        records.append(record)

    except Exception as e:
        print(f"[ERROR] Could not process {file_name}: {e}")

# -----------------------------
# SAVE FEATURES
# -----------------------------
df = pd.DataFrame(records)
df.to_csv(OUTPUT_CSV, index=False)

print(f"[SUCCESS] Image features saved to {OUTPUT_CSV}")
print(f"[INFO] Total images processed: {len(df)}")
