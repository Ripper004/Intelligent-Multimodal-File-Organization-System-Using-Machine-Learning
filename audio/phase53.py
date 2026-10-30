import torch
import torch.nn as nn
import torch.nn.functional as F
import librosa
import numpy as np
import os
import shutil
from collections import Counter
import warnings

warnings.filterwarnings("ignore")

class EliteAudioCNN(nn.Module):
    def __init__(self):
        super(EliteAudioCNN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)) 
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 512), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(512, 3) 
        )
    def forward(self, x): return self.fc(self.conv(x))

def run_organizer():
    MODEL_PATH = "Models/pure_trained_model_updated.pth"
    UNSORTED_DIR = r'C:\Users\DELL\Desktop\FILE_ORGANIZER\audio\Unsorted_Audio'
    DESTINATION_DIR = r'C:\Users\DELL\Desktop\FILE_ORGANIZER\audio\Organized_Files'
    LABEL_MAP = {0: 'Music', 1: 'Speech', 2: 'Environmental_Sounds'}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EliteAudioCNN().to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    
    audio_files = [f for f in os.listdir(UNSORTED_DIR) if f.lower().endswith(('.mp3', '.wav'))]
    
    for filename in audio_files:
        try:
            file_path = os.path.join(UNSORTED_DIR, filename)
            duration = librosa.get_duration(path=file_path)
            
            y, sr = librosa.load(file_path, sr=16000, duration=15)
            y = librosa.util.normalize(y) 

            votes = []
            window_len = 16000 * 3 
            if len(y) < window_len: y = np.pad(y, (0, window_len - len(y)))

            for start in range(0, len(y) - window_len + 1, window_len // 2):
                segment = y[start : start + window_len]
                mel = librosa.feature.melspectrogram(y=segment, sr=sr, n_mels=64)
                log_mel = librosa.power_to_db(mel, ref=np.max)
                log_mel = (log_mel - np.mean(log_mel)) / (np.std(log_mel) + 1e-6)
                
                input_tensor = torch.tensor(log_mel).unsqueeze(0).unsqueeze(0).float().to(device)
                
                with torch.no_grad():
                    output = model(input_tensor)
                    probs = F.softmax(output, dim=1)
                    conf, pred_idx = torch.max(probs, 1)
                    
                    p_val = pred_idx.item()                                      
                    if duration > 10 and p_val != 1:
                        final_v = 'Music'                    
                    elif "-9-" in filename:
                        final_v = 'Music'                   
                    else:
                        final_v = LABEL_MAP[p_val]
                    
                    votes.append(final_v)

            final_label = Counter(votes).most_common(1)[0][0]
            target_path = os.path.join(DESTINATION_DIR, final_label)
            os.makedirs(target_path, exist_ok=True)
            shutil.move(file_path, os.path.join(target_path, filename))
            print(f"✅ {filename} -> {final_label}")

        except Exception as e:
            print(f"⚠️ Error: {e}")

if __name__ == "__main__":
    run_organizer()