import os
import cv2
import torch
import clip
import pandas as pd
import numpy as np
from PIL import Image
from tqdm import tqdm

# -----------------------------
# Configuration
# -----------------------------
INPUT_DIR = "data/input_files"
OUTPUT_PATH = "data/features/video_features.csv"

VIDEO_EXTENSIONS = (".mp4", ".avi", ".mov", ".mkv", ".gif")
FRAME_INTERVAL = 15   # sample every N frames
MAX_FRAMES = 20       # cap frames per video

# -----------------------------
# Load CLIP
# -----------------------------
print("[INFO] Loading CLIP ViT-B/32...")
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# -----------------------------
# Helper: extract frames
# -----------------------------
def extract_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % FRAME_INTERVAL == 0:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(frame_rgb))

        frame_count += 1
        if len(frames) >= MAX_FRAMES:
            break

    cap.release()
    return frames

# -----------------------------
# Process videos & GIFs
# -----------------------------
video_embeddings = []

print("[INFO] Scanning input_files for videos and GIFs...")

for file in tqdm(os.listdir(INPUT_DIR)):
    if file.lower().endswith(VIDEO_EXTENSIONS):
        file_path = os.path.join(INPUT_DIR, file)

        frames = extract_frames(file_path)
        if len(frames) == 0:
            continue

        with torch.no_grad():
            frame_tensors = torch.stack(
                [preprocess(f) for f in frames]
            ).to(device)

            features = model.encode_image(frame_tensors)
            features = features / features.norm(dim=-1, keepdim=True)

            video_feature = features.mean(dim=0).cpu().numpy()

        video_embeddings.append([file] + video_feature.tolist())

# -----------------------------
# Save features
# -----------------------------
if video_embeddings:
    columns = ["file_name"] + [f"f{i}" for i in range(len(video_embeddings[0]) - 1)]
    df = pd.DataFrame(video_embeddings, columns=columns)

    os.makedirs("data/features", exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)

    print("[SUCCESS] Video & GIF features saved to data/features/video_features.csv")
    print(f"[INFO] Total videos/GIFs processed: {len(video_embeddings)}")
else:
    print("[WARNING] No video or GIF files found.")
