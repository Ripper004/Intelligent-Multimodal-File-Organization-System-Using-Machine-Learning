import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import normalize

# Importing your specific project functions
from core.extractors import (
    get_image_embedding, 
    get_video_embedding, 
    get_text_embedding, 
    extract_text_content
)

def generate_multimodal_graph(data_path):
    all_vectors = []
    labels = []
    
    print(f"🚀 Processing files in: {data_path}")

    for root, _, files in os.walk(data_path):
        for file in files:
            file_path = os.path.join(root, file)
            ext = os.path.splitext(file)[1].lower()
            vec = None
            category = ""

            # 1. Use your custom extractors to get embeddings
            if ext in [".jpg", ".jpeg", ".png", ".bmp"]:
                vec = get_image_embedding(file_path)
                category = "Image (CLIP)"
            elif ext in [".mp4", ".mkv", ".avi", ".mov"]:
                vec = get_video_embedding(file_path)
                category = "Video (CLIP)"
            elif ext in [".pdf", ".docx", ".txt"]:
                text = extract_text_content(file_path, ext)
                if text:
                    vec = get_text_embedding(text, file_path)
                    category = "Text (SBERT)"

            if vec is not None:
                # 2. DIMENSIONALITY ALIGNMENT (CRITICAL)
                # SBERT is 384-d, CLIP is 512-d. We pad SBERT to 512.
                aligned_vec = np.zeros(512)
                aligned_vec[:len(vec)] = vec
                
                all_vectors.append(aligned_vec)
                labels.append(category)

    if len(all_vectors) < 5:
        print("❌ Error: Not enough files found to create a cluster map.")
        return

    # 3. Prepare data for t-SNE
    X = np.array(all_vectors)
    X_norm = normalize(X) # Normalizing helps t-SNE find clusters faster

    print(f"📊 Running t-SNE on {len(X)} multimodal vectors...")
    # Using max_iter for modern sklearn compatibility
    tsne = TSNE(n_components=2, perplexity=min(30, len(X)-1), max_iter=1000, random_state=42)
    X_2d = tsne.fit_transform(X_norm)

    # 4. Plotting the results
    plt.figure(figsize=(12, 8))
    
    # Custom colors for your project categories
    color_map = {
        "Image (CLIP)": "#3498db",  # Blue
        "Video (CLIP)": "#e74c3c",  # Red
        "Text (SBERT)": "#2ecc71"   # Green
    }

    for cat in color_map:
        indices = [i for i, l in enumerate(labels) if l == cat]
        if indices:
            plt.scatter(X_2d[indices, 0], X_2d[indices, 1], 
                        label=cat, c=color_map[cat], 
                        alpha=0.7, edgecolors='white', s=100)

    plt.title("Multimodal Latent Space Visualization", fontsize=16, fontweight='bold')
    plt.xlabel("t-SNE Dimension 1", fontsize=12)
    plt.ylabel("t-SNE Dimension 2", fontsize=12)
    plt.legend(title="Modality", frameon=True, shadow=True)
    plt.grid(True, linestyle='--', alpha=0.3)

    # Save for PPT
    save_path = "multimodal_cluster_map.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ Success! Graph saved as: {save_path}")
    plt.show()

if __name__ == "__main__":
    # Point this to your actual raw files folder
    generate_multimodal_graph("data/raw_files")