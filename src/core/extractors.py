import torch, clip, cv2, os, numpy as np, re
from PIL import Image
from sentence_transformers import SentenceTransformer
from PyPDF2 import PdfReader
from docx import Document

device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)
sbert_model = SentenceTransformer("all-MiniLM-L6-v2")

def clean_technical_noise(text):
    text = re.sub(r'\d{2}[A-Z]{3}\d{4}', '', text)
    text = re.sub(r'\S+@\S+', '', text)
    text = re.sub(r'(vegamovies|1080p|720p|x264|h264|bluray|web-dl|esub|hevc|brrip)', '', text, flags=re.I)
    return " ".join(text.split())

def extract_text_content(file_path, extension):
    text = ""
    try:
        if extension == ".pdf":
            reader = PdfReader(file_path)
            for page in reader.pages[:10]:
                text += page.extract_text() or ""
                if len(text) > 5000: break
        elif extension == ".docx":
            doc = Document(file_path)
            text = " ".join(p.text for p in doc.paragraphs[:50])
        elif extension == ".txt":
            with open(file_path, 'r', encoding="utf-8", errors="ignore") as f:
                text = f.read(5000)
        return clean_technical_noise(text)
    except Exception: return ""

def get_text_embedding(text, file_path):
    input_str = clean_technical_noise(f"{os.path.basename(file_path)} {text[:3000]}")
    return sbert_model.encode(input_str)

def get_image_embedding(file_path):
    try:
        image = clip_preprocess(Image.open(file_path)).unsqueeze(0).to(device)
        with torch.no_grad():
            return clip_model.encode_image(image).cpu().numpy().flatten()
    except Exception: return None

def get_video_embedding(video_path):
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total < 24: return None
    indices = np.linspace(0, total - 1, 12).astype(int)
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            frames.append(clip_preprocess(img))
    cap.release()
    if not frames: return None
    with torch.no_grad():
        tensors = torch.stack(frames).to(device)
        return clip_model.encode_image(tensors).cpu().numpy().mean(axis=0)