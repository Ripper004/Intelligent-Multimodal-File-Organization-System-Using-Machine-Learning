import os
import librosa
import soundfile as sf
import numpy as np

# --- CONFIG ---
DATASET_ROOT = r'C:\Users\DELL\Desktop\FILE_ORGANIZER\audio\audio_dataset'
TARGET_SR = 16000

def reconstruct_without_ffmpeg():
    print("🏗️ Starting Phase 5.0: Python-Only Reconstruction...")
    
    count = 0
    errors = 0
    
    for root, _, files in os.walk(DATASET_ROOT):
        for f in files:
            if f.lower().endswith(('.mp3', '.wav')):
                file_path = os.path.join(root, f)
                target_path = os.path.splitext(file_path)[0] + "_clean.wav"
                
                try:
                    # 1. Load using librosa's built-in audioread (No FFmpeg needed)
                    # We load at 16k immediately to standardize
                    y, sr = librosa.load(file_path, sr=TARGET_SR, mono=True)
                    
                    # 2. Trim silence from the start and end
                    # This prevents the model from learning 'silence'
                    y_trimmed, _ = librosa.effects.trim(y, top_db=30)
                    
                    # 3. Save as a standard PCM WAV file
                    sf.write(target_path, y_trimmed, TARGET_SR, subtype='PCM_16')
                    
                    # 4. Cleanup
                    os.remove(file_path) # Remove original
                    os.rename(target_path, file_path.replace(os.path.splitext(f)[1], '.wav'))
                    
                    count += 1
                    if count % 50 == 0:
                        print(f"Processed {count} files...")
                        
                except Exception as e:
                    print(f"🗑️ Skipping/Deleting corrupted file: {f}")
                    try:
                        os.remove(file_path)
                    except:
                        pass
                    errors += 1

    print(f"✨ Phase 5.0 Complete! Reconstructed: {count} | Purged: {errors}")

if __name__ == "__main__":
    reconstruct_without_ffmpeg()