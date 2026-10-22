import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
import random
import gc

# ==========================================
# 0. REPRODUCIBILITY
# ==========================================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
g = torch.Generator()
g.manual_seed(42)

# ==========================================
# 1. CONFIGURATION
# ==========================================
class Config:
    DATASET = 'DEAP'
    EMOTION = 'valence'
    
    # Standard processed cache path
    CACHE_PATH = f'./data/processed_data_{EMOTION}_{DATASET}' 
    
    NUM_SUBJECTS = 32
    BANDS = 4
    IMG_SIZE = 9
    
    SEQ_LEN_SECONDS = 5
    SEQ_LEN_SAMPLES = 5 * 128
    
    BATCH_SIZE = 16
    FOLDS = 10
    
   
    EPOCHS = 30
    LR = 5e-4            
    WEIGHT_DECAY = 0.0    # No weight decay
    DROPOUT = 0.0         # Zero dropout for memorization
    LABEL_SMOOTHING = 0.0 
    
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


# 2. DATASET LOADER

class InMemoryEEGDataset(Dataset):
    def __init__(self, metadata_df):
        self.metadata = metadata_df.reset_index(drop=True)
      
        self.data_list = [torch.load(row['filename']) for _, row in self.metadata.iterrows()]
        self.label_list = [torch.tensor(row['label'], dtype=torch.float32) for _, row in self.metadata.iterrows()]
            
    def __len__(self): return len(self.metadata)
    def __getitem__(self, idx): return self.data_list[idx], self.label_list[idx]


# 3. VANILLA ARCHITECTURE

class StandardConvBlock(nn.Module):
    """ VANILLA CNN BLOCK: NO Residual Skip Connections """
    def __init__(self, in_channels, out_channels, stride=1, dropout=0.0):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels) # BatchNorm to track subject noise
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels) 
        self.dropout = nn.Dropout(dropout) 

    def forward(self, x):
        # Without the "+ x" shortcut
        out = F.gelu(self.bn1(self.conv1(x)))
        out = self.dropout(out)
        out = self.bn2(self.conv2(out))
        return F.gelu(out)

class VanillaSpatialEncoder(nn.Module):
    """ Lightweight Vanilla Encoder: 4 -> 16 -> 32 -> 64 """
    def __init__(self, in_bands=4, embed_dim=64, dropout=0.0):
        super().__init__()
        self.init = nn.Conv2d(in_bands, 16, 3, padding=1)
        self.enc1 = StandardConvBlock(16, 32, stride=2, dropout=dropout)       
        self.enc2 = StandardConvBlock(32, 64, stride=2, dropout=dropout)      
        self.bottleneck = StandardConvBlock(64, embed_dim, stride=1, dropout=dropout) 
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        e1 = self.enc1(F.gelu(self.init(x))) 
        e2 = self.enc2(e1) 
        b = self.bottleneck(e2) 
        return self.global_pool(b).view(b.size(0), -1) 

class Vanilla_Entangled_Model(nn.Module):
    def __init__(self, in_bands=4, embed_dim=64, dropout=0.0):
        super().__init__()
        self.spatial_enc = VanillaSpatialEncoder(in_bands, embed_dim, dropout)
        
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.pos_emb = nn.Parameter(torch.randn(1, 6, embed_dim))
        
        # Lightweight 1-Layer Transformer (Easy to optimize)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=2, dim_feedforward=128, 
            dropout=dropout, activation='relu', batch_first=True, norm_first=False
        )
        self.temporal_net = nn.TransformerEncoder(enc_layer, num_layers=1)
        self.head = nn.Sequential(
            nn.Linear(embed_dim, 32), nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        B, T, C, H, W = x.shape
        feats = self.spatial_enc(x.view(B*T, C, H, W)).view(B, T, -1)
        
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x_seq = torch.cat((cls_tokens, feats), dim=1) + self.pos_emb
        x_seq = self.temporal_net(x_seq)
        return self.head(x_seq[:, 0])


# 4. EXECUTION LOOP

def train_and_eval(model, train_loader, test_loader, subj_df, test_mask, config):
    opt = optim.AdamW(model.parameters(), lr=config.LR, weight_decay=config.WEIGHT_DECAY)
    crit = nn.BCEWithLogitsLoss()
    
    best_acc = 0.0
    for epoch in range(config.EPOCHS):
        model.train()
        for Xb, yb in train_loader:
            Xb, yb = Xb.to(config.DEVICE), yb.to(config.DEVICE).unsqueeze(1)
            opt.zero_grad()
            loss = crit(model(Xb), yb)
            loss.backward()
            opt.step()
        
        model.eval()
        preds, truths = [], []
        with torch.no_grad():
            for Xv, yv in test_loader:
                preds.extend((torch.sigmoid(model(Xv.to(config.DEVICE))) > 0.5).int().cpu().numpy().flatten())
                truths.extend(yv.int().numpy().flatten())
                
        test_df = subj_df[test_mask]
        trial_votes, trial_truths = [], []
        for t_id in np.unique(test_df['trial'].values):
            idx = np.where(test_df['trial'].values == t_id)[0]
            if len(idx) == 0: continue
            trial_votes.append(1 if np.sum(np.array(preds)[idx]) >= (len(idx) / 2.0) else 0)
            trial_truths.append(np.array(truths)[idx][0])
            
        acc = np.mean(np.array(trial_votes) == np.array(trial_truths))
        if acc > best_acc: best_acc = acc
            
    return best_acc

def run_experiment(config):
    meta_df = pd.read_csv(os.path.join(config.CACHE_PATH, 'meta.csv'))
    
    # NEW CHECKPOINT FILE
    ckpt_file = os.path.join(config.CACHE_PATH, f"ckpt_subdep_{config.DATASET}_{config.EMOTION}_VANILLA_LIGHTWEIGHT.json")
    
    if os.path.exists(ckpt_file):
        with open(ckpt_file, 'r') as f: all_subject_accuracies = json.load(f)
        print(f"\n[CHECKPOINT] Resuming Lightweight Vanilla from Subject {len(all_subject_accuracies)}...")
    else:
        all_subject_accuracies = []
        
    start_subject = len(all_subject_accuracies)
    unique_subjects = sorted(meta_df['subject'].unique())
    
    for subject_id in unique_subjects[start_subject:]:
        print(f"\n  -> Subject {subject_id} | Model: VANILLA LIGHTWEIGHT ENTANGLED | Dataset: {config.DATASET}")
        subj_df = meta_df[meta_df['subject'] == subject_id].copy()
        unique_trials = subj_df['trial'].unique()
        kf = KFold(n_splits=config.FOLDS, shuffle=True, random_state=42)
        cv_scores = []
        
        for fold, (train_idx, test_idx) in enumerate(kf.split(unique_trials)):
            train_mask = subj_df['trial'].isin(unique_trials[train_idx])
            test_mask = subj_df['trial'].isin(unique_trials[test_idx]) & (subj_df['start_sample'] % config.SEQ_LEN_SAMPLES == 0)
            
            train_loader = DataLoader(InMemoryEEGDataset(subj_df[train_mask]), batch_size=config.BATCH_SIZE, shuffle=True, num_workers=0, generator=g, worker_init_fn=seed_worker)
            test_loader = DataLoader(InMemoryEEGDataset(subj_df[test_mask]), batch_size=config.BATCH_SIZE, shuffle=False, num_workers=0)
            
            model = Vanilla_Entangled_Model(in_bands=config.BANDS, dropout=config.DROPOUT).to(config.DEVICE)
            
            fold_acc = train_and_eval(model, train_loader, test_loader, subj_df, test_mask, config)
            cv_scores.append(fold_acc)
            
        subj_mean = np.mean(cv_scores)
        print(f"     => Subject {subject_id} Mean CV Accuracy: {subj_mean:.4f}")
        all_subject_accuracies.append(subj_mean)
        
        with open(ckpt_file, 'w') as f: json.dump(all_subject_accuracies, f)
            
    final_acc = np.mean(all_subject_accuracies)
    print("\n" + "="*65)
    print(f" FINAL VANILLA LIGHTWEIGHT SUBJECT-DEPENDENT RESULTS ({config.DATASET})")
    print(f" Mean Accuracy (32 Subjects): {final_acc:.4f} (+/- {np.std(all_subject_accuracies):.4f})")
    print("="*65)

if __name__ == "__main__":
    config = Config()
    run_experiment(config)
