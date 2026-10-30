import os
import pandas as pd

def smart_relabel_and_clean():
    # --- CONFIGURATION ---
    DATASET_ROOT = r'C:\Users\DELL\Desktop\FILE_ORGANIZER\audio\audio_dataset'
    MANIFEST_PATH = 'Manifests/final_smart_manifest2.csv'
    
    data = []
    categories = ['Music', 'Speech', 'Environmental_Sounds']
    
    print("🧹 Generating Strict Truth Manifest (Ignoring Filenames)...")
    
    for label in categories:
        cat_path = os.path.join(DATASET_ROOT, label)
        if not os.path.exists(cat_path):
            print(f"⚠️ Warning: Folder not found: {cat_path}")
            continue
        
        count = 0
        for root, _, files in os.walk(cat_path):
            for f in files:
                # We only want audio files for training
                if not f.lower().endswith(('.wav', '.mp3')): continue
                
                file_path = os.path.join(root, f)
                
                # THE "STRICT TRUTH" RULE:
                # We trust the folder you put the file in, 100%. 
                # If it's in the 'Speech' folder, the label is 'Speech'.
                # We no longer check if the name starts with '03-' or has UrbanSound IDs.
                current_label = label 
                
                data.append({'path': file_path, 'label': current_label})
                count += 1
        
        print(f"📂 Found {count} files in {label}")

    if not data:
        print("❌ Error: No audio files found in the dataset directory!")
        return

    df = pd.DataFrame(data)
    
    print("\n📊 Final Class Distribution (Folder-Based):")
    print(df['label'].value_counts())

    # --- BALANCED SAMPLING ---
    # We still balance the classes so the model doesn't get biased toward the largest folder.
    # Target 4000 per class for a robust foundation.
    balanced_df = df.groupby('label').apply(
        lambda x: x.sample(n=min(len(x), 4000), random_state=42)
    ).reset_index(drop=True)
    
    os.makedirs('Manifests', exist_ok=True)
    balanced_df.to_csv(MANIFEST_PATH, index=False)
    print(f"\n✅ Manifest ready: {len(balanced_df)} files mapped and balanced.")

if __name__ == "__main__":
    smart_relabel_and_clean()