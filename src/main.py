import os, pandas as pd, shutil, numpy as np
from scipy.spatial.distance import pdist
from datetime import datetime
from core.extractors import get_text_embedding, get_image_embedding, get_video_embedding, extract_text_content
from core.clustering import run_precision_clustering, run_superfolder_merging
from core.labeler import generate_leaf_cluster_name, generate_superfolder_name

def main():    
    ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    CSV_PATH = os.path.join(ROOT, "data", "metadata", "file_index.csv")
    OUTPUT_BASE = os.path.join(ROOT, "organized_output")
    REPORT_PATH = os.path.join(ROOT, "data", "metadata", "organization_report.txt")   
    
    MIN_CLUSTERS_TO_NEST = 6    
    SIMILARITY_THRESHOLD = 0.6   
    if os.path.exists(OUTPUT_BASE): 
        shutil.rmtree(OUTPUT_BASE)
    
    df = pd.read_csv(CSV_PATH)
    data_map = {"image": [], "document": [], "video": []}
    stats = {"image": 0, "document": 0, "video": 0}

   
    print("--- PHASE 1: Processing Files ---")
    for _, row in df.iterrows():
        path, f_type, ext = row['filepath'], row['file_type'], row['extension']
        target_type = f_type if f_type in ["image", "video"] else "document"
        
        try:
            if target_type == "image": 
                feat = get_image_embedding(path)
            elif target_type == "video": 
                feat = get_video_embedding(path)
            else: 
                feat = get_text_embedding(extract_text_content(path, ext), path)
            
            if feat is not None:
                data_map[target_type].append({"path": path, "vector": feat})
                stats[target_type] += 1
        except Exception as e:
            print(f"Error processing {path}: {e}")

    report = f"SMART ADAPTIVE REPORT - {datetime.now().strftime('%Y-%m-%d %H:%M')}\n"
    report += "="*60 + "\n"

   
    print("--- PHASE 2: Adaptive Hierarchy Generation ---")
    for f_type, items in data_map.items():
        if not items: continue
        
        vectors = np.array([i['vector'] for i in items])
        paths = [i['path'] for i in items]        
        
        leaf_labels = run_precision_clustering(vectors)
        leaf_data_map = {}
        
        for label in set(leaf_labels):
            idx_list = [i for i, l in enumerate(leaf_labels) if l == label]
            leaf_paths = [paths[i] for i in idx_list]
            leaf_data_map[label] = {
                "paths": leaf_paths,
                "name": generate_leaf_cluster_name(leaf_paths),
                "centroid": vectors[idx_list].mean(axis=0)
            }
        
        leaf_indices = sorted(leaf_data_map.keys())
        centroids = np.array([leaf_data_map[idx]["centroid"] for idx in leaf_indices])

        use_nested = False
        if len(leaf_indices) >= MIN_CLUSTERS_TO_NEST:            
            distances = pdist(centroids, metric='cosine')
            avg_similarity = 1 - np.mean(distances)
            
            if avg_similarity >= SIMILARITY_THRESHOLD:
                use_nested = True
            else:
                print(f"[DECISION] {f_type.upper()}: Diversity too high ({avg_similarity:.2f}). Keeping FLAT.")

       
        if not use_nested:
            report += f"\n>>> {f_type.upper()} (STRUCTURE: FLAT)\n"
            for l_idx in leaf_indices:
                leaf = leaf_data_map[l_idx]
                target_dir = os.path.join(OUTPUT_BASE, f_type, leaf['name'])
                os.makedirs(target_dir, exist_ok=True)
                
                for p in leaf['paths']:
                    shutil.copy(p, os.path.join(target_dir, os.path.basename(p)))
                report += f"  └─ {leaf['name']} ({len(leaf['paths'])} files)\n"
        else:          
            super_labels = run_superfolder_merging(centroids)
            hierarchy = {}
            for s_idx, s_label in enumerate(super_labels):
                leaf_idx = leaf_indices[s_idx]
                members = [leaf_data_map[leaf_indices[i]]["name"] 
                           for i, l in enumerate(super_labels) if l == s_label]
                super_name = generate_superfolder_name(members)
                
                if super_name not in hierarchy: 
                    hierarchy[super_name] = []
                hierarchy[super_name].append(leaf_data_map[leaf_idx])

            report += f"\n>>> {f_type.upper()} (STRUCTURE: NESTED)\n"
            for super_name, leaves in hierarchy.items():
                report += f"  ● SUPERFOLDER: {super_name}\n"
                for leaf in leaves:
                    report += f"    └─ {leaf['name']}\n"
                    target_dir = os.path.join(OUTPUT_BASE, f_type, super_name, leaf['name'])
                    os.makedirs(target_dir, exist_ok=True)
                    for p in leaf['paths']:
                        shutil.copy(p, os.path.join(target_dir, os.path.basename(p)))
 
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(report)
        
    print(f"\n[SUCCESS] Organized output created in 'organized_output'")
    print(f"[REPORT] Log saved to: {REPORT_PATH}")

if __name__ == "__main__":
    main()