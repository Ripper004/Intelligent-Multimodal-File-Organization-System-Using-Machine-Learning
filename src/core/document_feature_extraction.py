import pandas as pd
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer
from PyPDF2 import PdfReader
from docx import Document

# ---------------- PATHS ----------------
METADATA_FILE = Path("data/metadata/file_index.csv")
OUTPUT_DIR = Path("data/features")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TEXT_FEATURES_FILE = OUTPUT_DIR / "text_features.npy"
TEXT_INDEX_FILE = OUTPUT_DIR / "text_features_index.csv"

# ---------------- LOAD MODEL ----------------
print("[INFO] Loading SBERT MiniLM model...")
model = SentenceTransformer("all-MiniLM-L6-v2")

# ---------------- TEXT EXTRACTION FUNCTIONS ----------------
def extract_text_from_pdf(file_path):
    try:
        reader = PdfReader(file_path)
        return " ".join(page.extract_text() or "" for page in reader.pages)
    except:
        return ""

def extract_text_from_docx(file_path):
    try:
        doc = Document(file_path)
        return " ".join(p.text for p in doc.paragraphs)
    except:
        return ""

def extract_text_from_txt(file_path):
    try:
        return Path(file_path).read_text(encoding="utf-8", errors="ignore")
    except:
        return ""

def clean_text(text):
    text = text.lower()
    text = " ".join(text.split())
    return text[:5000]   # safety cap for very large documents

# ---------------- PIPELINE ----------------
print("[INFO] Reading metadata...")
df = pd.read_csv(METADATA_FILE)

documents = df[df["file_type"] == "document"]

texts = []
filenames = []

print(f"[INFO] Processing {len(documents)} document files...")

for _, row in documents.iterrows():
    file_path = row["filepath"]
    extension = row["extension"]

    if extension == ".pdf":
        text = extract_text_from_pdf(file_path)
    elif extension == ".docx":
        text = extract_text_from_docx(file_path)
    elif extension == ".txt":
        text = extract_text_from_txt(file_path)
    else:
        continue

    text = clean_text(text)

    if text.strip():
        texts.append(text)
        filenames.append(row["filename"])

# ---------------- EMBEDDING ----------------
print("[INFO] Generating SBERT embeddings...")
embeddings = model.encode(texts, show_progress_bar=True)

# ---------------- SAVE OUTPUTS ----------------
np.save(TEXT_FEATURES_FILE, embeddings)

pd.DataFrame({
    "filename": filenames
}).to_csv(TEXT_INDEX_FILE, index=False)

print("[SUCCESS] Step 3.1 completed.")
print(f"Embeddings shape: {embeddings.shape}")
print(f"Saved to: {TEXT_FEATURES_FILE}")
