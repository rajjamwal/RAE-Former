import os
import shutil
import pickle
import numpy as np
import scipy.signal as signal
from scipy.interpolate import griddata
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import gc
import pandas as pd

# Force Clean Memory
gc.collect()
torch.cuda.empty_cache()


# 1. CONFIGURATION 

class Config:
    DEAP_PATH = './data/deap'
    CACHE_PATH = './data/processed_data'
    
    NUM_SUBJECTS = 32
    CHANNELS = 32
    SR = 128
    
    # --- TEMPORAL & SPATIAL ---
    SEQ_LEN_SECONDS = 5
    SEQ_LEN_SAMPLES = 5 * 128
    GEN_STRIDE = int(0.5 * 128)  
    IMG_SIZE = 9 
    BANDS = [(4, 8), (8, 13), (13, 30), (30, 45)]
    
    # --- HYPERPARAMETERS ---
    BATCH_SIZE = 64          # Scaled up for 31-subject pooled training
    FOLDS = 32               # Leave-One-Subject-Out (32 Subjects)
    
    # For Phase 1
    EPOCHS_STAGE1 = 20      
    LR_STAGE1 = 5e-4        
    
    # For Phase 2
    EPOCHS_STAGE2 = 30      
    LR_STAGE2 = 3e-4        
    
    # REGULARIZATION
    WEIGHT_DECAY = 1e-3      
    DROPOUT = 0.5            
    LABEL_SMOOTHING = 0.1   
    
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

if not os.path.exists(Config.CACHE_PATH):
    os.makedirs(Config.CACHE_PATH, exist_ok=True)

print(f"Running Universal LOSO (Pure Spatial + Transformer) Pipeline on: {Config.DEVICE}")


# 2. DATA PROCESSING (Trial-Level Z-Scoring)

DEAP_TO_GRID = {
    0: (0, 3), 1: (1, 3), 2: (2, 2), 3: (2, 0), 4: (3, 1), 5: (3, 3), 
    6: (4, 2), 7: (4, 0), 8: (5, 1), 9: (5, 3), 10: (6, 2), 11: (6, 0), 
    12: (7, 3), 13: (8, 3), 14: (8, 4), 15: (6, 4), 16: (0, 5), 17: (1, 5), 
    18: (2, 4), 19: (2, 6), 20: (2, 8), 21: (3, 7), 22: (3, 5), 23: (4, 4), 
    24: (4, 6), 25: (4, 8), 26: (5, 7), 27: (5, 5), 28: (6, 6), 29: (6, 8), 
    30: (7, 5), 31: (8, 5)
}

def butter_bandpass_filter(data, lowcut, highcut, fs, order=3):
    nyq = 0.5 * fs
    b, a = signal.butter(order, [lowcut/nyq, highcut/nyq], btype='band')
    return signal.filtfilt(b, a, data, axis=-1)

def compute_de(segment):
    var = np.var(segment, axis=-1)
    return 0.5 * np.log(2 * np.pi * np.e * np.clip(var, 1e-8, None))

def interpolate_topo(de_values, config):
    points = np.array(list(DEAP_TO_GRID.values()))
    values = np.array([de_values[i] for i in range(config.CHANNELS)])
    grid_x, grid_y = np.mgrid[0:config.IMG_SIZE, 0:config.IMG_SIZE]
    grid_z = griddata(points, values, (grid_x, grid_y), method='cubic')
    if np.isnan(grid_z).any():
        grid_nearest = griddata(points, values, (grid_x, grid_y), method='nearest')
        grid_z[np.isnan(grid_z)] = grid_nearest[np.isnan(grid_z)]
    return grid_z

def process_segment_topo_9x9(segment_data, config):
    frames = []
    step = config.SR
    for start in range(0, segment_data.shape[1], step):
        end = start + step
        if end > segment_data.shape[1]: break
        chunk = segment_data[:, start:end]
        grid = np.zeros((len(config.BANDS), config.IMG_SIZE, config.IMG_SIZE))
        for b_idx, (low, high) in enumerate(config.BANDS):
            filt = butter_bandpass_filter(chunk, low, high, config.SR)
            de_vals = compute_de(filt)
            grid[b_idx] = interpolate_topo(de_vals, config)
        frames.append(grid)
    return np.array(frames) 

def process_baseline_topo_9x9(baseline_data, config):
    frames = process_segment_topo_9x9(baseline_data, config)
    return np.mean(frames, axis=0) 

def generate_and_cache_data(config):
    subject_files = sorted([f for f in os.listdir(config.DEAP_PATH) if f.endswith('.dat')])[:config.NUM_SUBJECTS]
    metadata = [] 
    file_counter = 0
    
    print(f"Generating TOPO Data (Trial-Level Z-Scored) to Disk ({config.CACHE_PATH})...")
    
    for subj_idx, f in enumerate(subject_files):
        print(f"  -> Processing Subject {subj_idx}/{config.NUM_SUBJECTS - 1}")
        d = pickle.load(open(os.path.join(config.DEAP_PATH, f), 'rb'), encoding='latin1')
        eeg = d['data'][:, :config.CHANNELS, :] 
        
        labels = (d['labels'][:, 1] >= 5.0).astype(int)
        n_trials, _, n_samples = eeg.shape 
        
        for t in range(n_trials):
            trial_data = eeg[t] 
            lbl = labels[t]
            
            baseline_samples = 3 * config.SR
            baseline_data = trial_data[:, :baseline_samples]
            baseline_grid = process_baseline_topo_9x9(baseline_data, config)
            
            curr = baseline_samples
            trial_frames = []
            trial_starts = []
            
            while (curr + config.SEQ_LEN_SAMPLES) <= n_samples:
                seg = trial_data[:, curr : curr+config.SEQ_LEN_SAMPLES]
                frames = process_segment_topo_9x9(seg, config)
                
                if len(frames) == int(config.SEQ_LEN_SECONDS):
                    normalized_frames = frames - baseline_grid
                    trial_frames.append(normalized_frames)
                    trial_starts.append(curr)
                curr += config.GEN_STRIDE 
            
            if len(trial_frames) > 0:
                trial_frames_np = np.array(trial_frames) 
                mean_val = np.mean(trial_frames_np)
                std_val = np.std(trial_frames_np) + 1e-8
                z_scored_trial = (trial_frames_np - mean_val) / std_val
                
                for idx, z_frames in enumerate(z_scored_trial):
                    fname = f"seq_{file_counter}.pt"
                    fpath = os.path.join(config.CACHE_PATH, fname)
                    torch.save(torch.tensor(z_frames, dtype=torch.float32), fpath)
                    
                    metadata.append({
                        'filename': fpath,
                        'label': lbl,
                        'subject': subj_idx,
                        'trial': t, 
                        'start_sample': trial_starts[idx]
                    })
                    file_counter += 1
                
    meta_df = pd.DataFrame(metadata)
    meta_df.to_csv(os.path.join(config.CACHE_PATH, 'meta.csv'), index=False)
    return meta_df

class LazyEEGDataset(Dataset):
    def __init__(self, metadata_df):
        self.metadata = metadata_df.reset_index(drop=True)
    def __len__(self): return len(self.metadata)
    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        x = torch.load(row['filename']) 
        y = torch.tensor(row['label'], dtype=torch.long)
        return x, y
    
# ==========================================
# 3. ARCHITECTURE (Pure Autoencoder + Transformer)
# ==========================================

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, dropout=0.5):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.in1 = nn.InstanceNorm2d(out_channels, affine=True) 
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.in2 = nn.InstanceNorm2d(out_channels, affine=True) 
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.InstanceNorm2d(out_channels, affine=True)
            )
        self.dropout = nn.Dropout(dropout) 

    def forward(self, x):
        out = F.gelu(self.in1(self.conv1(x)))
        out = self.dropout(out)
        out = self.in2(self.conv2(out))
        out += self.shortcut(x)
        return F.gelu(out)

class Pure_SpatialAutoencoder(nn.Module):
    """ NO CLASSIFICATION HEAD. Pure Self-Supervised Spatial Learner. """
    def __init__(self, in_bands=4, latent_dim=256, dropout=0.5):
        super(Pure_SpatialAutoencoder, self).__init__()
        self.init = nn.Conv2d(in_bands, 32, 3, padding=1)
        self.enc1 = ResidualBlock(32, 64, stride=2, dropout=dropout)       
        self.enc2 = ResidualBlock(64, 128, stride=2, dropout=dropout)      
        self.bottleneck = ResidualBlock(128, latent_dim, stride=1, dropout=dropout) 
        
        self.dec1 = ResidualBlock(latent_dim, 64, dropout=dropout) 
        self.dec2 = ResidualBlock(64, 32, dropout=dropout)
        self.final_rec = nn.Conv2d(32, in_bands, 3, padding=1)
        
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        x0 = F.gelu(self.init(x))
        e1 = self.enc1(x0) 
        e2 = self.enc2(e1) 
        b = self.bottleneck(e2) 
        
        rec_feat = F.interpolate(b, size=(9, 9), mode='bilinear', align_corners=False)
        d1 = self.dec1(rec_feat) 
        d2 = self.dec2(d1)       
        rec = self.final_rec(d2) 
        
        latent_flat = self.global_pool(b).view(b.size(0), -1) 
        return rec, latent_flat

class EndToEnd_LOSOTransformer_9x9(nn.Module):
    def __init__(self, pretrained_unet=None, in_bands=4, seq_len=5, num_classes=1, embed_dim=256, dropout=0.5):
        super(EndToEnd_LOSOTransformer_9x9, self).__init__()
        
        self.spatial_enc = Pure_SpatialAutoencoder(in_bands, latent_dim=embed_dim, dropout=dropout)
        if pretrained_unet:
            self.spatial_enc.load_state_dict(pretrained_unet.state_dict())
            
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.pos_emb = nn.Parameter(torch.randn(1, seq_len + 1, embed_dim))
        
        # FULL CAPACITY TRANSFORMER (3 Layers, 8 Heads) for massive LOSO training sets
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=8, dim_feedforward=1024, 
            dropout=dropout, activation='gelu', batch_first=True, norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=3)
        
        self.head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, 128), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        B, T, C, H, W = x.shape
        x_flat = x.view(B*T, C, H, W)
        
        # Pure Spatial Extraction (Full 256 dims)
        _, feats = self.spatial_enc(x_flat) 
        feats = feats.view(B, T, -1) 
        
        # Transformer Sequence Logic
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x_seq = torch.cat((cls_tokens, feats), dim=1)
        x_seq = x_seq + self.pos_emb
        x_seq = self.transformer(x_seq)
        
        return self.head(x_seq[:, 0])
    

# ==========================================
# 4. EXECUTION LOOP (LOSO + Trial Voting)
# ==========================================

class BinaryLabelSmoothingLoss(nn.Module):
    def __init__(self, smoothing=0.1, pos_weight=None):
        super(BinaryLabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.pos_weight = pos_weight
    def forward(self, pred, target):
        with torch.no_grad():
            smooth_target = target * self.confidence + 0.5 * self.smoothing
        return F.binary_cross_entropy_with_logits(pred, smooth_target, pos_weight=self.pos_weight)

def get_loso_folds(metadata_df, config):
    folds = []
    unique_subjects = sorted(metadata_df['subject'].unique())
    
    print(f"Splitting into {len(unique_subjects)} LOSO Folds (Train on 31 humans, Test on 1)...")
    for subj in unique_subjects:
        is_test_subj = (metadata_df['subject'] == subj)
        is_strict = (metadata_df['start_sample'] % config.SEQ_LEN_SAMPLES == 0) 
        test_mask = is_test_subj & is_strict
        
        train_mask = (metadata_df['subject'] != subj)
        folds.append((metadata_df[train_mask], metadata_df[test_mask], subj))
        
    return folds

def run_loso_experiment(config):
    meta_df = pd.read_csv(os.path.join(config.CACHE_PATH, 'meta.csv'))
    folds = get_loso_folds(meta_df, config)
    all_loso_accuracies = []

    for k, (train_df, test_df, test_subj) in enumerate(folds):
        print(f"\n" + "="*55)
        print(f"  FOLD {k+1}/{config.FOLDS} | HELD-OUT TEST SUBJECT: {test_subj}")
        print("="*55)
        
        train_ds = LazyEEGDataset(train_df)
        test_ds = LazyEEGDataset(test_df)
        
        train_loader = DataLoader(train_ds, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=2)
        test_loader = DataLoader(test_ds, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=2)
        
        # --- PHASE 1: SELF-SUPERVISED AUTOENCODER ---
        print(f"  [Phase 1] Pure Spatial Autoencoder (MSE Training)...")
        unet = Pure_SpatialAutoencoder(in_bands=len(config.BANDS), dropout=config.DROPOUT).to(config.DEVICE)
        opt_unet = optim.AdamW(unet.parameters(), lr=config.LR_STAGE1, weight_decay=config.WEIGHT_DECAY)
        crit_rec = nn.MSELoss()
        
        for epoch in range(config.EPOCHS_STAGE1):
            unet.train()
            for Xb, _ in train_loader: 
                B, T, C, H, W = Xb.shape
                Xb_flat = Xb.view(B*T, C, H, W).to(config.DEVICE)
                
                opt_unet.zero_grad()
                recon, _ = unet(Xb_flat)
                loss = crit_rec(recon, Xb_flat) 
                loss.backward()
                opt_unet.step()
                
        # --- PHASE 2: FULL TRANSFORMER ---
        print(f"  [Phase 2] Full Transformer Fine-tuning (Soft-Unlocked Core)...")
        model = EndToEnd_LOSOTransformer_9x9(
            pretrained_unet=unet, in_bands=len(config.BANDS), 
            dropout=config.DROPOUT
        ).to(config.DEVICE)
        
        unet_params = []
        transformer_params = []
        for name, param in model.named_parameters():
            if param.requires_grad:
                if 'spatial_enc' in name:
                    unet_params.append(param)
                else:
                    transformer_params.append(param)
                    
        # Differential Optimizer 
        optimizer = optim.AdamW([
            {'params': unet_params, 'lr': config.LR_STAGE2 * 0.1}, 
            {'params': transformer_params, 'lr': config.LR_STAGE2}
        ], weight_decay=config.WEIGHT_DECAY)
        
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10)
        
        pos_count = train_df['label'].sum()
        neg_count = len(train_df) - pos_count
        weight_ratio = torch.tensor([neg_count / (pos_count + 1e-5)], dtype=torch.float32).to(config.DEVICE)
        crit_cls_weighted = BinaryLabelSmoothingLoss(smoothing=config.LABEL_SMOOTHING, pos_weight=weight_ratio)
        
        best_trial_acc = 0
        
        for epoch in range(config.EPOCHS_STAGE2):
            model.train()
            train_loss = 0
            for Xb, yb in train_loader:
                Xb = Xb.to(config.DEVICE)
                yb = yb.to(config.DEVICE).float().unsqueeze(1)
                
                optimizer.zero_grad()
                logits = model(Xb)
                
                loss = crit_cls_weighted(logits, yb)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            
            scheduler.step()
            
            # --- EVALUATION: MAJORITY VOTING ---
            model.eval()
            all_preds = []
            all_truths = []
            
            with torch.no_grad():
                for Xv, yv in test_loader:
                    Xv = Xv.to(config.DEVICE)
                    logits = model(Xv)
                    preds = (torch.sigmoid(logits) > 0.5).int().cpu().numpy().flatten()
                    truths = yv.int().cpu().numpy().flatten()
                    
                    all_preds.extend(preds)
                    all_truths.extend(truths)
                    
            all_preds = np.array(all_preds)
            all_truths = np.array(all_truths)
            segment_acc = np.mean(all_preds == all_truths)
            
            test_trial_ids = test_df['trial'].values
            unique_test_trials = np.unique(test_trial_ids)
            
            trial_votes = []
            trial_truths = []
            
            for t_id in unique_test_trials:
                idx = np.where(test_trial_ids == t_id)[0]
                seg_preds = all_preds[idx]
                seg_truths = all_truths[idx]
                
                vote = 1 if np.sum(seg_preds) >= (len(seg_preds) / 2.0) else 0
                trial_votes.append(vote)
                trial_truths.append(seg_truths[0]) 
            
            trial_acc = np.mean(np.array(trial_votes) == np.array(trial_truths))
            if trial_acc > best_trial_acc: best_trial_acc = trial_acc 
            
            if (epoch+1) % 5 == 0:
                lr = scheduler.get_last_lr()[0]
                print(f"    Ep {epoch+1:2d} | L: {train_loss/len(train_loader):.3f} | Seg Acc: {segment_acc:.3f} | Trial Acc: {trial_acc:.3f}")

        print(f"  >> Subject {test_subj} Best Trial-Wise Accuracy: {best_trial_acc:.3f}")
        all_loso_accuracies.append(best_trial_acc)
        
        del model, unet, optimizer, scheduler, train_loader, test_loader
        torch.cuda.empty_cache()

    print("\n" + "="*55)
    print(f"FINAL PUBLICATION RESULTS (LOSO Pure Autoencoder + Transformer)")
    print(f"Mean Trial Accuracy (32 Subjects): {np.mean(all_loso_accuracies):.4f} (+/- {np.std(all_loso_accuracies):.4f})")
    print("="*55)

if __name__ == "__main__":
    config = Config()
    csv_file_path = os.path.join(config.CACHE_PATH, 'meta.csv')
    
    if not os.path.exists(csv_file_path):
        print("Data files not found. Initiating file generation...")
        generate_and_cache_data(config)
    else:
        print("Existing data files found! Skipping generation...")
        
    run_loso_experiment(config)

