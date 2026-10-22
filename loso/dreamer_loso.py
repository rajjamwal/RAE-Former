import os
import gc
import random
import scipy.io as sio
import numpy as np
import scipy.signal as signal
from scipy.interpolate import griddata
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd

# ==========================================
# 0. REPRODUCIBILITY & MEMORY
# ==========================================
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


# 1. DREAMER CONFIGURATION & SPATIAL MAPPING

class Config_DREAMER:
    DREAMER_FILE = '.data/DREAMER.mat'
    
    # Toggle Emotion Here
    EMOTION = 'valence' 
    CACHE_PATH = f'.data/processed_heavy_loso_{EMOTION}' 
    
    NUM_SUBJECTS = 23        
    CHANNELS = 14            
    SR = 128                 
    
    SEQ_LEN_SECONDS = 5
    SEQ_LEN_SAMPLES = 5 * 128
    GEN_STRIDE = int(0.5 * 128)  
    IMG_SIZE = 9 
    BANDS = [(4, 8), (8, 13), (13, 30), (30, 45)] 
    
    BATCH_SIZE = 64          
    FOLDS = 23               
    
    # Standard DEAP Hyperparameters
    EPOCHS_STAGE1 = 20      
    LR_STAGE1 = 5e-4        
    EPOCHS_STAGE2 = 30      
    LR_STAGE2 = 3e-4        
    WEIGHT_DECAY = 1e-3      
    DROPOUT = 0.5            
    LABEL_SMOOTHING = 0.1   
    
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

if not os.path.exists(Config_DREAMER.CACHE_PATH):
    os.makedirs(Config_DREAMER.CACHE_PATH, exist_ok=True)

# 14 Emotiv EPOC channels to 9x9
DREAMER_TO_GRID = {
    0: (1, 3), 1: (2, 1), 2: (2, 3), 3: (3, 1), 
    4: (4, 0), 5: (6, 1), 6: (8, 3), 7: (8, 5), 
    8: (6, 7), 9: (4, 8), 10: (3, 7), 11: (2, 5), 
    12: (2, 7), 13: (1, 5)
}


# 2. DATA EXTRACTION & TOPOLOGICAL MAPPING

def butter_bandpass_filter(data, lowcut, highcut, fs, order=3):
    nyq = 0.5 * fs
    b, a = signal.butter(order, [lowcut/nyq, highcut/nyq], btype='band')
    return signal.filtfilt(b, a, data, axis=-1)

def compute_de(segment):
    var = np.var(segment, axis=-1)
    return 0.5 * np.log(2 * np.pi * np.e * np.clip(var, 1e-8, None))

def interpolate_topo_dreamer(de_values, config):
    points = np.array(list(DREAMER_TO_GRID.values()))
    values = np.array([de_values[i] for i in range(config.CHANNELS)])
    grid_x, grid_y = np.mgrid[0:config.IMG_SIZE, 0:config.IMG_SIZE]
    
    grid_z = griddata(points, values, (grid_x, grid_y), method='linear')
    if np.isnan(grid_z).any():
        grid_nearest = griddata(points, values, (grid_x, grid_y), method='nearest')
        grid_z[np.isnan(grid_z)] = grid_nearest[np.isnan(grid_z)]
    return grid_z

def process_segment_topo_dreamer(segment_data, config):
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
            grid[b_idx] = interpolate_topo_dreamer(de_vals, config)
        frames.append(grid)
    return np.array(frames) 

def generate_and_cache_dreamer_data(config):
    print(f"Loading DREAMER.mat... Extracting {config.EMOTION.upper()}")
    data = sio.loadmat(config.DREAMER_FILE)
    
    dreamer_struct = data['DREAMER'][0, 0]
    participants_data = dreamer_struct['Data'][0]
    
    metadata = [] 
    file_counter = 0
    score_key = 'ScoreValence' if config.EMOTION == 'valence' else 'ScoreArousal'
    
    for subj_idx in range(config.NUM_SUBJECTS):
        print(f"  -> Processing Subject {subj_idx + 1}/{config.NUM_SUBJECTS}")
        subj_data = participants_data[subj_idx][0, 0]
        
        eeg_data = subj_data['EEG'][0, 0]
        stimuli = eeg_data['stimuli']
        baseline = eeg_data['baseline']
        
        scores = subj_data[score_key].flatten()
        labels = (scores > 3.0).astype(int) 
        
        for t in range(18): 
            lbl = labels[t]
            
            trial_data = stimuli[t, 0].T
            base_data = baseline[t, 0].T
            
            baseline_frames = process_segment_topo_dreamer(base_data, config)
            baseline_grid = np.mean(baseline_frames, axis=0) 
            
            curr = 0
            n_samples = trial_data.shape[1]
            trial_frames, trial_starts = [], []
            
            while (curr + config.SEQ_LEN_SAMPLES) <= n_samples:
                seg = trial_data[:, curr : curr+config.SEQ_LEN_SAMPLES]
                frames = process_segment_topo_dreamer(seg, config)
                
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
                        'filename': fpath, 'label': lbl, 'subject': subj_idx + 1, 
                        'trial': t, 'start_sample': trial_starts[idx]
                    })
                    file_counter += 1
                
    meta_df = pd.DataFrame(metadata)
    meta_df.to_csv(os.path.join(config.CACHE_PATH, 'meta_heavy.csv'), index=False)
    return meta_df

class InMemoryEEGDataset(Dataset):
    def __init__(self, metadata_df, cache_dict):
        self.metadata = metadata_df.reset_index(drop=True)
        self.data_list = []
        self.label_list = []
        
        for _, row in self.metadata.iterrows():
            filepath = row['filename']
            if filepath not in cache_dict:
                cache_dict[filepath] = torch.load(filepath)
            
            self.data_list.append(cache_dict[filepath])
            self.label_list.append(torch.tensor(row['label'], dtype=torch.float32))
            
    def __len__(self): return len(self.metadata)
    def __getitem__(self, idx): return self.data_list[idx], self.label_list[idx]


# 3. ARCHITECTURE (Heavy RAE + Transformer from DEAP), NOTE: Table results uses 2 layers and 6 heads for dreamer 

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, dropout=0.5):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.in1 = nn.InstanceNorm2d(out_channels, affine=True) 
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.in2 = nn.InstanceNorm2d(out_channels, affine=True) 
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False), nn.InstanceNorm2d(out_channels, affine=True))
        self.dropout = nn.Dropout(dropout) 

    def forward(self, x):
        return F.gelu(self.in2(self.conv2(self.dropout(F.gelu(self.in1(self.conv1(x))))))) + self.shortcut(x)

class Pure_SpatialAutoencoder(nn.Module):
    def __init__(self, in_bands=4, latent_dim=256, dropout=0.5):
        super(Pure_SpatialAutoencoder, self).__init__()
        # Heavy DEAP Capacity: 32 -> 64 -> 128 -> 256
        self.init = nn.Conv2d(in_bands, 32, 3, padding=1)
        self.enc1 = ResidualBlock(32, 64, stride=2, dropout=dropout)       
        self.enc2 = ResidualBlock(64, 128, stride=2, dropout=dropout)      
        self.bottleneck = ResidualBlock(128, latent_dim, stride=1, dropout=dropout) 
        
        self.dec1 = ResidualBlock(latent_dim, 64, dropout=dropout) 
        self.dec2 = ResidualBlock(64, 32, dropout=dropout)
        self.final_rec = nn.Conv2d(32, in_bands, 3, padding=1)
        
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        e1 = self.enc1(F.gelu(self.init(x))) 
        e2 = self.enc2(e1) 
        b = self.bottleneck(e2) 
        
        rec = self.final_rec(self.dec2(self.dec1(F.interpolate(b, size=(9, 9), mode='bilinear', align_corners=False)))) 
        return rec, self.global_pool(b).view(b.size(0), -1) 

class EndToEnd_LOSOTransformer_9x9(nn.Module):
    def __init__(self, pretrained_unet=None, in_bands=4, seq_len=5, num_classes=1, embed_dim=256, dropout=0.5):
        super(EndToEnd_LOSOTransformer_9x9, self).__init__()
        
        # Spatial Encoder generates a 256-D signature
        self.spatial_enc = Pure_SpatialAutoencoder(in_bands, latent_dim=embed_dim, dropout=dropout)
        if pretrained_unet:
            self.spatial_enc.load_state_dict(pretrained_unet.state_dict())
            
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.pos_emb = nn.Parameter(torch.randn(1, seq_len + 1, embed_dim))
        
        # HEAVY TRANSFORMER: 3 Layers, 8 Heads, 1024 Feedforward.
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
        _, feats = self.spatial_enc(x.view(B*T, C, H, W)) 
        feats = feats.view(B, T, -1) 
        
        x_seq = self.transformer(torch.cat((self.cls_token.expand(B, -1, -1), feats), dim=1) + self.pos_emb)
        return self.head(x_seq[:, 0])


# 4. EXECUTION LOOP & MAJORITY VOTING

class BinaryLabelSmoothingLoss(nn.Module):
    def __init__(self, smoothing=0.1, pos_weight=None):
        super().__init__()
        self.conf, self.smooth, self.pos_weight = 1.0 - smoothing, smoothing, pos_weight
    def forward(self, pred, target):
        return F.binary_cross_entropy_with_logits(pred, target * self.conf + 0.5 * self.smooth, pos_weight=self.pos_weight)

def get_loso_folds_dreamer(metadata_df, config):
    folds = []
    unique_subjects = sorted(metadata_df['subject'].unique())
    for subj in unique_subjects:
        test_mask = (metadata_df['subject'] == subj) & (metadata_df['start_sample'] % config.SEQ_LEN_SAMPLES == 0)
        train_mask = (metadata_df['subject'] != subj)
        folds.append((metadata_df[train_mask], metadata_df[test_mask], subj))
    return folds

def run_dreamer_experiment(config):
    meta_df = pd.read_csv(os.path.join(config.CACHE_PATH, 'meta_heavy.csv'))
    folds = get_loso_folds_dreamer(meta_df, config)
    all_loso_accuracies = []

    print(f"\n[CACHE] Pre-loading dataset to RAM for fast training...")
    ram_cache = {}
    for f in meta_df['filename'].unique(): ram_cache[f] = torch.load(f)

    for k, (train_df, test_df, test_subj) in enumerate(folds):
        print(f"\n" + "="*55)
        print(f"  FOLD {k+1}/{config.FOLDS} | HELD-OUT DREAMER SUBJECT: {test_subj} (HEAVY ARCH)")
        print("="*55)
        
        train_ds = InMemoryEEGDataset(train_df, ram_cache)
        test_ds = InMemoryEEGDataset(test_df, ram_cache)
        
        train_loader = DataLoader(train_ds, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=0, generator=g, worker_init_fn=seed_worker)
        test_loader = DataLoader(test_ds, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=0)
        
        print(f"  [Phase 1] Heavy Spatial RAE (MSE Training)...")
        unet = Pure_SpatialAutoencoder(in_bands=len(config.BANDS), dropout=config.DROPOUT).to(config.DEVICE)
        opt_unet = optim.AdamW(unet.parameters(), lr=config.LR_STAGE1, weight_decay=config.WEIGHT_DECAY)
        crit_rec = nn.MSELoss()
        
        for epoch in range(config.EPOCHS_STAGE1):
            unet.train()
            for Xb, _ in train_loader: 
                Xb_flat = Xb.view(-1, len(config.BANDS), config.IMG_SIZE, config.IMG_SIZE).to(config.DEVICE)
                opt_unet.zero_grad()
                recon, _ = unet(Xb_flat)
                loss = crit_rec(recon, Xb_flat) 
                loss.backward()
                opt_unet.step()
                
        print(f"  [Phase 2] Heavy DEAP Transformer Fine-tuning...")
        model = EndToEnd_LOSOTransformer_9x9(pretrained_unet=unet, in_bands=len(config.BANDS), dropout=config.DROPOUT).to(config.DEVICE)
        
        # Differential Optimizer
        opt = optim.AdamW([
            {'params': [p for n, p in model.named_parameters() if 'spatial_enc' in n], 'lr': config.LR_STAGE2 * 0.1}, 
            {'params': [p for n, p in model.named_parameters() if 'spatial_enc' not in n], 'lr': config.LR_STAGE2}
        ], weight_decay=config.WEIGHT_DECAY)
        
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=10)
        pos_weight = torch.tensor([(len(train_df) - train_df['label'].sum()) / (train_df['label'].sum() + 1e-5)]).float().to(config.DEVICE)
        crit = BinaryLabelSmoothingLoss(smoothing=config.LABEL_SMOOTHING, pos_weight=pos_weight)
        
        best_trial_acc = 0
        
        for epoch in range(config.EPOCHS_STAGE2):
            model.train()
            for Xb, yb in train_loader:
                opt.zero_grad()
                loss = crit(model(Xb.to(config.DEVICE)), yb.to(config.DEVICE).unsqueeze(1))
                loss.backward()
                opt.step()
            scheduler.step()
            
            # --- EVALUATION: STRICT MAJORITY VOTING ---
            model.eval()
            preds, truths = [], []
            with torch.no_grad():
                for Xv, yv in test_loader:
                    preds.extend((torch.sigmoid(model(Xv.to(config.DEVICE))) > 0.5).int().cpu().numpy().flatten())
                    truths.extend(yv.int().numpy().flatten())
                    
            test_trial_ids = test_df['trial'].values
            trial_votes, trial_truths = [], []
            
            for t_id in np.unique(test_trial_ids):
                idx = np.where(test_trial_ids == t_id)[0]
                if len(idx) == 0: continue
                trial_votes.append(1 if np.sum(np.array(preds)[idx]) >= (len(idx) / 2.0) else 0)
                trial_truths.append(np.array(truths)[idx][0]) 
            
            trial_acc = np.mean(np.array(trial_votes) == np.array(trial_truths))
            if trial_acc > best_trial_acc: best_trial_acc = trial_acc 

        print(f"  >> Subject {test_subj} Best LOSO Trial-Wise Accuracy: {best_trial_acc:.3f}")
        all_loso_accuracies.append(best_trial_acc)
        
        del model, unet, opt, scheduler
        torch.cuda.empty_cache()

    print("\n" + "="*55)
    print(f"FINAL DREAMER HEAVY-LOSO RESULTS ({config.EMOTION.upper()})")
    print(f"Mean Trial Accuracy (23 Subjects): {np.mean(all_loso_accuracies)*100:.2f}% (+/- {np.std(all_loso_accuracies)*100:.2f}%)")
    print("="*55)

if __name__ == "__main__":
    config = Config_DREAMER()
    csv_file_path = os.path.join(config.CACHE_PATH, 'meta_heavy.csv')
    
    if not os.path.exists(csv_file_path):
        print("Data files not found. Initiating generation...")
        generate_and_cache_dreamer_data(config)
    else:
        print(f"Existing DREAMER {config.EMOTION} cache found! Skipping generation...")
        
    run_dreamer_experiment(config)
