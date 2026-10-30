import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
import librosa
import numpy as np
import pandas as pd
import warnings
import os
from sklearn.utils.class_weight import compute_class_weight

warnings.filterwarnings("ignore")

# --- 1. ELITE ARCHITECTURE ---
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

# --- 2. DATASET WITH HARD-NEGATIVE AUGMENTATION ---
class AudioDataset(Dataset):
    def __init__(self, csv_file, augment=False):
        self.df = pd.read_csv(csv_file)
        self.augment = augment
        self.label_map = {'Music': 0, 'Speech': 1, 'Environmental_Sounds': 2}
    def __len__(self): return len(self.df)
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        try:
            y, sr = librosa.load(row['path'], sr=16000, duration=3)
        except Exception:
            y, sr = np.zeros(48000), 16000             
        if len(y) < 48000: y = np.pad(y, (0, 48000 - len(y)))        
        if self.augment:
            if np.random.random() > 0.4:
                y += 0.008 * np.random.randn(len(y)) 
            if np.random.random() > 0.4:
                shift = int(np.random.uniform(-0.5, 0.5) * sr)
                y = np.roll(y, shift) 
        mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=64)
        log_mel = librosa.power_to_db(mel, ref=np.max)
        log_mel = (log_mel - np.mean(log_mel)) / (np.std(log_mel) + 1e-6)
        return torch.tensor(log_mel).unsqueeze(0).float(), self.label_map[row['label']]

# --- 3. EVALUATION FUNCTION ---
def evaluate(model, loader, device, criterion):
    model.eval()
    loss, correct, total = 0, 0, 0
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            out = model(inputs)
            loss += criterion(out, labels).item()
            _, pred = torch.max(out, 1)
            total += labels.size(0)
            correct += (pred == labels).sum().item()
    return loss / len(loader), (correct / total) * 100

# --- 4. MAIN TRAINING LOOP WITH EARLY STOPPING ---
def train():
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    MANIFEST_PATH = 'Manifests/final_smart_manifest2.csv'
    MODEL_SAVE_PATH = "Models/pure_trained_model_updated.pth"
    
    if not os.path.exists(MANIFEST_PATH):
        print("❌ Error: Run your corrected smart_relabel_and_clean.py script first!")
        return

    full_dataset = AudioDataset(MANIFEST_PATH, augment=True)
    
    # Calculate Weights to force focus on Environmental sounds
    labels_list = [full_dataset.label_map[l] for l in full_dataset.df['label']]
    weights = compute_class_weight('balanced', classes=np.unique(labels_list), y=labels_list)
    weights_tensor = torch.tensor(weights, dtype=torch.float).to(DEVICE)
    
    train_size = int(0.8 * len(full_dataset))
    train_ds, val_ds = random_split(full_dataset, [train_size, len(full_dataset)-train_size])
    val_ds.dataset.augment = False # Validation must be stable

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=32)

    model = EliteAudioCNN().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=0.0001) # Low LR to correct bias
    
    # Weighted Loss + Label Smoothing to handle tonal mimicry
    criterion = nn.CrossEntropyLoss(weight=weights_tensor, label_smoothing=0.1)
    
    # --- EARLY STOPPING SETUP ---
    best_val_loss = float('inf')
    best_acc = 0
    patience = 5  # Stop if no improvement in 5 epochs
    trigger_times = 0

    print(f"🚀 Starting Training on {DEVICE}...")
    print(f"Patience: {patience} | Label Smoothing: 0.1 | LR: 0.0001")
    print(f"\n{'Epoch':<5} | {'Trn Loss':<8} | {'Val Loss':<8} | {'Trn Acc':<8} | {'Val Acc':<8}")
    print("-" * 55)

    for epoch in range(50):
        model.train()
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            criterion(model(inputs), labels).backward()
            optimizer.step()
        
        t_loss, t_acc = evaluate(model, train_loader, DEVICE, criterion)
        v_loss, v_acc = evaluate(model, val_loader, DEVICE, criterion)
        
        print(f"{epoch+1:<5} | {t_loss:<8.4f} | {v_loss:<8.4f} | {t_acc:<8.2f}% | {v_acc:<8.2f}%")

        # --- EARLY STOPPING & SAVING LOGIC ---
        if v_loss < best_val_loss:
            best_val_loss = v_loss
            best_acc = v_acc
            trigger_times = 0
            os.makedirs("Models", exist_ok=True)
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
        else:
            trigger_times += 1
            if trigger_times >= patience:
                print(f"\n🛑 Early Stopping at Epoch {epoch+1}. Best Val Accuracy: {best_acc:.2f}%")
                break
            
    print(f"\n✨ Training Finished. Model saved to {MODEL_SAVE_PATH}")

if __name__ == "__main__":
    train()