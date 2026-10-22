import os
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
from sklearn.model_selection import KFold
import random
import json

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) 
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

g = torch.Generator()
g.manual_seed(42)

gc.collect()
if torch.cuda.is_available(): torch.cuda.empty_cache()

class Config:
    DATASET = 'DEAP'
    EMOTION = 'valence' # Toggle to 'arousal' if needed
    
    DEAP_PATH = './data/dataset'
    CACHE_PATH = f'.data/processed_heavy_subdep_{EMOTION}_{DATASET}'
    
    NUM_SUBJECTS = 32
    CHANNELS = 32
    SR = 128
    
    # Temporal and Spatial settings
    SEQ_LEN_SECONDS = 5
    SEQ_LEN_SAMPLES = 5 * 128
    GEN_STRIDE = int(0.5 * 128)  
    IMG_SIZE = 9 
    BANDS = [(4, 8), (8, 13), (13, 30), (30, 45)]
    
    # Hyperparams for Subject-Dependent
    BATCH_SIZE = 64          
    FOLDS = 10               # 10-Fold CV over the 40 trials per subject
    
    # Phase 1: Spatial Autoencoder (MSE Training)
    EPOCHS_STAGE1 = 20      
    LR_STAGE1 = 5e-4        
    
    # Phase 2: Full Transformer
    EPOCHS_STAGE2 = 30      
    LR_STAGE2 = 3e-4        
    
    # Regularization
    WEIGHT_DECAY = 1e-3      
    DROPOUT = 0.5            
    LABEL_SMOOTHING = 0.1   
    
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

if not os.path.exists(Config.CACHE_PATH):
    os.makedirs(Config.CACHE_PATH, exist_ok=True)

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
    
    print(f"Generating TOPO Data to Disk ({config.CACHE_PATH})...")
    lbl_idx = 0 if config.EMOTION == 'valence' else 1
    
    for subj_idx, f in enumerate(subject_files):
        # We index 1-32 to match standard DEAP nomenclature for sanity checks
        actual_subject_id = subj_idx + 1
        print(f"  -> Processing Subject {actual_subject_id}/{config.NUM_SUBJECTS}")
        d = pickle.load(open(os.path.join(config.DEAP_PATH, f), 'rb'), encoding='latin1')
        eeg = d['data'][:, :config.CHANNELS, :] 
        
        labels = (d['labels'][:, lbl_idx] >= 5.0).astype(int)
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
                        'subject': actual_subject_id, # Using 1-32 indexing
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
        
        # FULL CAPACITY TRANSFORMER (3 Layers, 8 Heads)
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
        
        # Spatial Extraction (Full 256 dims)
        _, feats = self.spatial_enc(x_flat) 
        feats = feats.view(B, T, -1) 
        
        # Transformer Sequence Logic
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x_seq = torch.cat((cls_tokens, feats), dim=1)
        x_seq = x_seq + self.pos_emb
        x_seq = self.transformer(x_seq)
        
        return self.head(x_seq[:, 0])
    
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

def run_heavy_subdep_experiment(config):
    meta_df = pd.read_csv(os.path.join(config.CACHE_PATH, 'meta.csv'))
    
    ckpt_file = os.path.join(config.CACHE_PATH, f"ckpt_heavy_subdep_{config.DATASET}_{config.EMOTION}.json")
    if os.path.exists(ckpt_file):
        try:
            with open(ckpt_file, 'r') as f:
                all_subject_accuracies = json.load(f)
            print(f"\n[CHECKPOINT] Found existing progress. Resuming from Subject {len(all_subject_accuracies)}...")
        except json.JSONDecodeError:
            print("\n[WARNING] Checkpoint corrupted. Restarting.")
            all_subject_accuracies = []
    else:
        all_subject_accuracies = []

    start_subject = len(all_subject_accuracies)
    unique_subjects = sorted(meta_df['subject'].unique())
    
    if start_subject >= len(unique_subjects):
        print(f"[CHECKPOINT] Testing is 100% complete. Final Acc: {np.mean(all_subject_accuracies):.4f}")
        return

    # Loop over individual subjects
    for subj_idx in range(start_subject, len(unique_subjects)):
        target_subject = unique_subjects[subj_idx]
        print(f"\n" + "="*55)
        print(f"  SUBJECT DEPENDENT EVAL (Heavy Arch) | DEAP Subject: {target_subject}")
        print("="*55)
        
        subj_df = meta_df[meta_df['subject'] == target_subject].copy()
        unique_trials = subj_df['trial'].unique()
        
        # 10-Fold Cross Validation strictly over trials
        kf = KFold(n_splits=config.FOLDS, shuffle=True, random_state=42)
        cv_scores = []
        
        for fold, (train_idx, test_idx) in enumerate(kf.split(unique_trials)):
            print(f"  -> Fold {fold+1}/{config.FOLDS}")
            
            train_mask = subj_df['trial'].isin(unique_trials[train_idx])
            test_mask = subj_df['trial'].isin(unique_trials[test_idx]) & (subj_df['start_sample'] % config.SEQ_LEN_SAMPLES == 0)
            
            train_loader = DataLoader(LazyEEGDataset(subj_df[train_mask]), batch_size=config.BATCH_SIZE, shuffle=True, num_workers=2, generator=g, worker_init_fn=seed_worker)
            test_loader = DataLoader(LazyEEGDataset(subj_df[test_mask]), batch_size=config.BATCH_SIZE, shuffle=False, num_workers=2)
            
            # --- PHASE 1: SELF-SUPERVISED AUTOENCODER ---
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
                        
            optimizer = optim.AdamW([
                {'params': unet_params, 'lr': config.LR_STAGE2 * 0.1}, 
                {'params': transformer_params, 'lr': config.LR_STAGE2}
            ], weight_decay=config.WEIGHT_DECAY)
            
            scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10)
            
            pos_count = subj_df[train_mask]['label'].sum()
            neg_count = len(subj_df[train_mask]) - pos_count
            weight_ratio = torch.tensor([neg_count / (pos_count + 1e-5)], dtype=torch.float32).to(config.DEVICE)
            crit_cls_weighted = BinaryLabelSmoothingLoss(smoothing=config.LABEL_SMOOTHING, pos_weight=weight_ratio)
            
            best_trial_acc = 0
            
            for epoch in range(config.EPOCHS_STAGE2):
                model.train()
                for Xb, yb in train_loader:
                    Xb = Xb.to(config.DEVICE)
                    yb = yb.to(config.DEVICE).float().unsqueeze(1)
                    
                    optimizer.zero_grad()
                    logits = model(Xb)
                    loss = crit_cls_weighted(logits, yb)
                    loss.backward()
                    optimizer.step()
                
                scheduler.step()
                
                # EVALUATION
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
                
                test_df = subj_df[test_mask]
                test_trial_ids = test_df['trial'].values
                unique_test_trials = np.unique(test_trial_ids)
                
                trial_votes = []
                trial_truths = []
                
                for t_id in unique_test_trials:
                    idx = np.where(test_trial_ids == t_id)[0]
                    if len(idx) == 0: continue
                    seg_preds = all_preds[idx]
                    seg_truths = all_truths[idx]
                    
                    vote = 1 if np.sum(seg_preds) >= (len(seg_preds) / 2.0) else 0
                    trial_votes.append(vote)
                    trial_truths.append(seg_truths[0]) 
                
                trial_acc = np.mean(np.array(trial_votes) == np.array(trial_truths))
                if trial_acc > best_trial_acc: best_trial_acc = trial_acc 

            print(f"     Best Fold Acc: {best_trial_acc:.3f}")
            cv_scores.append(best_trial_acc)
            
            del model, unet, optimizer, scheduler, train_loader, test_loader
            torch.cuda.empty_cache()

        subj_mean = np.mean(cv_scores)
        print(f"  >> Subject {target_subject} Mean 10-Fold Accuracy: {subj_mean:.3f}")
        
        all_subject_accuracies.append(subj_mean)
        with open(ckpt_file, 'w') as f:
            json.dump(all_subject_accuracies, f)

    print("\n" + "="*55)
    print(f"FINAL DEAP HEAVY SUBJECT-DEPENDENT RESULTS ({config.EMOTION.upper()})")
    print(f"Mean Accuracy (32 Subjects): {np.mean(all_subject_accuracies):.4f} (+/- {np.std(all_subject_accuracies):.4f})")
    print("="*55)

if __name__ == "__main__":
    config = Config()
    csv_file_path = os.path.join(config.CACHE_PATH, 'meta.csv')
    
    if not os.path.exists(csv_file_path):
        print("Data files not found. Initiating generation...")
        generate_and_cache_data(config)
    else:
        print("Existing data files found! Skipping generation...")
        
    run_heavy_subdep_experiment(config)
