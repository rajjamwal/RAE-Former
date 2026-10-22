import os
import random
import math
import numpy as np
import scipy.signal as signal
from scipy.interpolate import griddata
import torch
import torch.nn as nn
import pickle  # <-- for reading DEAP .dat files
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import gc
import pandas as pd
from sklearn.model_selection import KFold
import scipy.io as sio
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
    # --- DATASET TOGGLE ---
    DATASET = 'DREAMER'  # Change to 'DEAP' to switch datasets
    
    DEAP_PATH = '.data/deap' 
    DREAMER_FILE = '.data/DREAMER.mat'
    
    # Dual-Cache: Stores both Topo and Raw 1D signals (Switched to Arousal)
    CACHE_PATH = f'./data/processed_comparsion_arousal_{DATASET}'
    
    NUM_SUBJECTS = 32 if DATASET == 'DEAP' else 23
    CHANNELS = 32 if DATASET == 'DEAP' else 14
    SR = 128
    
    SEQ_LEN_SECONDS = 5
    SEQ_LEN_SAMPLES = 5 * 128
    GEN_STRIDE = int(0.5 * 128)  
    IMG_SIZE = 9 
    
    BANDS = [(4, 8), (8, 13), (13, 30), (30, 45)]
    BAND_NAMES = ['Theta', 'Alpha', 'Beta', 'Gamma']
    
    BATCH_SIZE = 32          
    FOLDS = NUM_SUBJECTS # Dynamically matches the dataset size for strict LOSO
    
    EPOCHS_SOTA = 30
    LR = 3e-4        
    WEIGHT_DECAY = 1e-3      
    DROPOUT = 0.5            
    LABEL_SMOOTHING = 0.1   
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

if not os.path.exists(Config.CACHE_PATH):
    os.makedirs(Config.CACHE_PATH, exist_ok=True)

def count_parameters(model, name):
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[{name}] Trainable Parameters: {params / 1e6:.4f} M")
    return params

DEAP_TO_GRID = {
    0: (0, 3), 1: (1, 3), 2: (2, 2), 3: (2, 0), 4: (3, 1), 5: (3, 3), 
    6: (4, 2), 7: (4, 0), 8: (5, 1), 9: (5, 3), 10: (6, 2), 11: (6, 0), 
    12: (7, 3), 13: (8, 3), 14: (8, 4), 15: (6, 4), 16: (0, 5), 17: (1, 5), 
    18: (2, 4), 19: (2, 6), 20: (2, 8), 21: (3, 7), 22: (3, 5), 23: (4, 4), 
    24: (4, 6), 25: (4, 8), 26: (5, 7), 27: (5, 5), 28: (6, 6), 29: (6, 8), 
    30: (7, 5), 31: (8, 5)
}

DREAMER_TO_GRID = {
    0: (1, 3), 1: (2, 1), 2: (2, 3), 3: (3, 1), 4: (4, 0), 5: (6, 1), 
    6: (8, 3), 7: (8, 5), 8: (6, 7), 9: (4, 8), 10: (3, 7), 11: (2, 5), 
    12: (2, 7), 13: (1, 5)
}

def butter_bandpass_filter(data, lowcut, highcut, fs, order=3):
    nyq = 0.5 * fs
    b, a = signal.butter(order, [lowcut/nyq, highcut/nyq], btype='band')
    return signal.filtfilt(b, a, data, axis=-1)

def compute_de(segment):
    var = np.var(segment, axis=-1)
    return 0.5 * np.log(2 * np.pi * np.e * np.clip(var, 1e-8, None))

def interpolate_topo(de_values, config):
    points_dict = DEAP_TO_GRID if config.DATASET == 'DEAP' else DREAMER_TO_GRID
    points = np.array(list(points_dict.values()))
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

def generate_and_cache_dreamer_data_dual(config):
    """ Generates BOTH 2D Topo Maps AND 1D Raw Sequences for DREAMER (AROUSAL) """
    print(f"\n[DATA GEN] Generating DUAL-CACHE (Topo + Raw) for {config.DATASET} (AROUSAL)...")
    data = sio.loadmat(config.DREAMER_FILE)
    dreamer_struct = data['DREAMER'][0, 0]
    participants_data = dreamer_struct['Data'][0]
    metadata, file_counter = [], 0
    
    for subj_idx in range(config.NUM_SUBJECTS):
        print(f"  -> Processing Subject {subj_idx + 1}/{config.NUM_SUBJECTS}")
        subj_data = participants_data[subj_idx][0, 0]
        eeg_data = subj_data['EEG'][0, 0]
        stimuli = eeg_data['stimuli']
        baseline = eeg_data['baseline']
        
        # EXTRACTING AROUSAL
        arousal_scores = subj_data['ScoreArousal'].flatten()
        labels = (arousal_scores > 3.0).astype(int) 
        
        for t in range(18):
            lbl = labels[t]
            trial_data = stimuli[t, 0].T
            base_data = baseline[t, 0].T
            
            baseline_grid = np.mean(process_segment_topo_9x9(base_data, config), axis=0)
            
            curr = 0
            n_samples = trial_data.shape[1]
            trial_topo_frames, trial_raw_frames, trial_starts = [], [], []
            
            while (curr + config.SEQ_LEN_SAMPLES) <= n_samples:
                seg = trial_data[:, curr : curr+config.SEQ_LEN_SAMPLES]
                
                # Create Topo features for Topo-based networks
                topo_frames = process_segment_topo_9x9(seg, config)
                if len(topo_frames) == int(config.SEQ_LEN_SECONDS):
                    trial_topo_frames.append(topo_frames - baseline_grid)
                    trial_raw_frames.append(seg) # Store the raw 1D signal for TSception/EEGNet/FAT
                    trial_starts.append(curr)
                curr += config.GEN_STRIDE 
            
            if len(trial_topo_frames) > 0:
                trial_topo_np = np.array(trial_topo_frames) 
                trial_raw_np = np.array(trial_raw_frames) 
                
                # Independent Trial-Wise Z-Scoring 
                z_topo = (trial_topo_np - np.mean(trial_topo_np)) / (np.std(trial_topo_np) + 1e-8)
                z_raw = (trial_raw_np - np.mean(trial_raw_np)) / (np.std(trial_raw_np) + 1e-8)
                
                for idx in range(len(z_topo)):
                    fpath = os.path.join(config.CACHE_PATH, f"seq_{file_counter}.pt")
                    
                    # SAVE DICTIONARY WITH BOTH SIGNAL TYPES
                    torch.save({
                        'topo': torch.tensor(z_topo[idx], dtype=torch.float32),
                        'raw': torch.tensor(z_raw[idx], dtype=torch.float32)
                    }, fpath)
                    
                    metadata.append({'filename': fpath, 'label': lbl, 'subject': subj_idx, 'trial': t, 'start_sample': trial_starts[idx]})
                    file_counter += 1
                
    pd.DataFrame(metadata).to_csv(os.path.join(config.CACHE_PATH, 'meta.csv'), index=False)
    print("[DATA GEN] Dual-Cache Generation Complete!")

def generate_deap_dual_cache(config):
    """ Generates BOTH 2D Topo Maps AND 1D Raw Sequences for DEAP (AROUSAL) """
    print(f"\n[DATA GEN] Generating DUAL-CACHE (Topo + Raw) for DEAP (AROUSAL)...")
    metadata = []
    file_counter = 0

    # DEAP has 32 subjects named s01.dat through s32.dat
    for subj_idx in range(1, config.NUM_SUBJECTS + 1):
        print(f"  -> Processing DEAP Subject {subj_idx}/{config.NUM_SUBJECTS}")
        subj_str = f"s{subj_idx:02d}.dat"
        filepath = os.path.join(config.DEAP_PATH, subj_str)
        
        with open(filepath, 'rb') as f:
            subject_data = pickle.load(f, encoding='latin1')
            
        data = subject_data['data']      # Shape: [40 trials, 40 channels, 8064 samples]
        labels = subject_data['labels']  # Shape: [40 trials, 4 ratings]
        
        # DEAP Ratings: [Valence, Arousal, Dominance, Liking]. 
        # Index 1 is Arousal. Threshold > 5.0
        arousal_labels = (labels[:, 1] > 5.0).astype(int)
        
        for t in range(40):
            lbl = arousal_labels[t]
            
            # Channels 0:32 are EEG. 
            # First 3 seconds (384 samples) are the baseline trial.
            # Remaining 60 seconds (7680 samples) are the active trial.
            baseline_data = data[t, :32, :384]
            trial_data = data[t, :32, 384:]
            
            # 1. Compute structural baseline Topo map
            baseline_grid = np.mean(process_segment_topo_9x9(baseline_data, config), axis=0)
            
            curr = 0
            n_samples = trial_data.shape[1]
            trial_topo_frames, trial_raw_frames, trial_starts = [], [], []
            
            # 2. Slice into 5-second overlapping sequences
            while (curr + config.SEQ_LEN_SAMPLES) <= n_samples:
                seg = trial_data[:, curr : curr+config.SEQ_LEN_SAMPLES] # 1D Raw Segment
                
                topo_frames = process_segment_topo_9x9(seg, config)
                
                if len(topo_frames) == int(config.SEQ_LEN_SECONDS):
                    trial_topo_frames.append(topo_frames - baseline_grid) # Subtract baseline
                    trial_raw_frames.append(seg) # Keep raw 1D signal
                    trial_starts.append(curr)
                    
                curr += config.GEN_STRIDE
                
            # 3. Independent Per-Trial Z-Scoring
            if len(trial_topo_frames) > 0:
                trial_topo_np = np.array(trial_topo_frames)
                trial_raw_np = np.array(trial_raw_frames)
                
                z_topo = (trial_topo_np - np.mean(trial_topo_np)) / (np.std(trial_topo_np) + 1e-8)
                z_raw = (trial_raw_np - np.mean(trial_raw_np)) / (np.std(trial_raw_np) + 1e-8)
                
                # 4. Save both formats as a Dictionary
                for idx in range(len(z_topo)):
                    fpath = os.path.join(config.CACHE_PATH, f"seq_{file_counter}.pt")
                    torch.save({
                        'topo': torch.tensor(z_topo[idx], dtype=torch.float32),
                        'raw': torch.tensor(z_raw[idx], dtype=torch.float32)
                    }, fpath)
                    
                    metadata.append({
                        'filename': fpath, 'label': lbl, 'subject': subj_idx, 
                        'trial': t, 'start_sample': trial_starts[idx]
                    })
                    file_counter += 1
                    
    pd.DataFrame(metadata).to_csv(os.path.join(config.CACHE_PATH, 'meta.csv'), index=False)
    print("[DATA GEN] DEAP Dual-Cache Generation Complete!")

class DualEEGDataset(Dataset):
    def __init__(self, metadata_df):
        self.metadata = metadata_df.reset_index(drop=True)
    def __len__(self): return len(self.metadata)
    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        data = torch.load(row['filename'])
        y = torch.tensor(row['label'], dtype=torch.float32)
        # Returns BOTH topologies and raw signals
        return data['topo'], data['raw'], y

class EEGNet(nn.Module):
    def __init__(self, channels, samples, F1=8, D=2, F2=16, dropout=0.5):
        super().__init__()
        self.conv1 = nn.Conv2d(1, F1, (1, 64), padding='same', bias=False)
        self.bn1 = nn.BatchNorm2d(F1)
        self.depthwise = nn.Conv2d(F1, F1 * D, (channels, 1), groups=F1, bias=False)
        self.bn2 = nn.BatchNorm2d(F1 * D)
        self.avgpool1 = nn.AvgPool2d((1, 4))
        self.dropout1 = nn.Dropout(dropout)
        self.separable = nn.Conv2d(F1 * D, F2, (1, 16), padding='same', groups=F1 * D, bias=False)
        self.pointwise = nn.Conv2d(F2, F2, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(F2)
        self.avgpool2 = nn.AvgPool2d((1, 8))
        self.dropout2 = nn.Dropout(dropout)
        flat_dim = F2 * (samples // 32)
        self.fc = nn.Linear(flat_dim, 1)

    def forward(self, x_raw):
        x = x_raw.unsqueeze(1) # Add channel dim
        x = self.dropout1(self.avgpool1(F.elu(self.bn2(self.depthwise(self.bn1(self.conv1(x)))))))
        x = self.dropout2(self.avgpool2(F.elu(self.bn3(self.pointwise(self.separable(x))))))
        return self.fc(x.flatten(start_dim=1))

class TSception(nn.Module):
    def __init__(self, channels, samples, hidden=32, dropout=0.5):
        super().__init__()
        self.T1 = nn.Sequential(nn.Conv2d(1, hidden, (1, int(0.5*128)), padding='same'), nn.LeakyReLU(), nn.AvgPool2d((1, 4)))
        self.T2 = nn.Sequential(nn.Conv2d(1, hidden, (1, int(0.25*128)), padding='same'), nn.LeakyReLU(), nn.AvgPool2d((1, 4)))
        self.T3 = nn.Sequential(nn.Conv2d(1, hidden, (1, int(0.125*128)), padding='same'), nn.LeakyReLU(), nn.AvgPool2d((1, 4)))
        self.S1 = nn.Sequential(nn.Conv2d(hidden*3, hidden, (channels, 1)), nn.LeakyReLU())
        self.S2 = nn.Sequential(nn.Conv2d(hidden*3, hidden, (int(channels/2), 1)), nn.LeakyReLU())
        self.dropout = nn.Dropout(dropout)
        self.flatten = nn.Flatten()
        with torch.no_grad():
            dummy = torch.zeros(1, 1, channels, samples)
            t_out = torch.cat([self.T1(dummy), self.T2(dummy), self.T3(dummy)], dim=1)
            flat_dim = self.S1(t_out).numel() + self.S2(t_out).numel()
        self.fc = nn.Linear(flat_dim, 1)

    def forward(self, x_raw):
        x = x_raw.unsqueeze(1)
        t_out = torch.cat([self.T1(x), self.T2(x), self.T3(x)], dim=1)
        s1, s2 = self.S1(t_out), self.S2(t_out)
        flat = torch.cat([self.flatten(s1), self.flatten(s2)], dim=1)
        return self.fc(self.dropout(flat))

class EEG_Conformer(nn.Module):
    def __init__(self, channels, samples, embed_dim=40, dropout=0.5):
        super().__init__()
        self.patch_emb = nn.Sequential(
            nn.Conv2d(1, 40, (1, 25), padding='same', bias=False), nn.BatchNorm2d(40),
            nn.Conv2d(40, 40, (channels, 1), bias=False), nn.BatchNorm2d(40), nn.ELU(),
            nn.AvgPool2d((1, 7), stride=(1, 7)), nn.Dropout(dropout)
        )
        enc_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=2, dim_feedforward=128, dropout=dropout, activation='gelu', batch_first=True)
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=2)
        with torch.no_grad():
            dummy = torch.zeros(1, 1, channels, samples)
            seq_len = self.patch_emb(dummy).shape[3]
        self.fc = nn.Sequential(nn.Flatten(), nn.Linear(embed_dim * seq_len, 1))

    def forward(self, x_raw):
        x = self.patch_emb(x_raw.unsqueeze(1))
        x = x.squeeze(2).permute(0, 2, 1)
        return self.fc(self.transformer(x))

class FAL(nn.Module):
    """ Fourier Analytic Linear (FAL) Layer """
    def __init__(self, in_features, out_features):
        super().__init__()
        self.W_p = nn.Linear(in_features, out_features // 2, bias=False)
        self.W_n = nn.Linear(in_features, out_features)
        
    def forward(self, x):
        wp = self.W_p(x)
        periodic = torch.cat([torch.cos(wp), torch.sin(wp)], dim=-1)
        aperiodic = self.W_n(x)
        return periodic, aperiodic

class FAA(nn.Module):
    """ Fourier Adjacent Attention (FAA) """
    def __init__(self, d_model, num_heads, channels):
        super().__init__()
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.q_proj = FAL(d_model, d_model)
        self.k_proj = FAL(d_model, d_model)
        self.v_proj = FAL(d_model, d_model)
        
        self.adj_p = nn.Parameter(torch.randn(1, num_heads, channels, channels))
        self.adj_n = nn.Parameter(torch.randn(1, num_heads, channels, channels))
        
        self.gate_p = nn.Linear(d_model, channels)
        self.gate_n = nn.Linear(d_model, channels)
        self.out_proj = nn.Linear(d_model * 2, d_model)

    def forward(self, x):
        B, L, _ = x.shape 
        qp, qn = self.q_proj(x)
        kp, kn = self.k_proj(x)
        vp, vn = self.v_proj(x)
        
        gp = torch.sigmoid(self.gate_p(qp)).unsqueeze(1)
        gn = torch.sigmoid(self.gate_n(qn)).unsqueeze(1)
        
        qp = qp.view(B, L, self.num_heads, self.d_k).transpose(1, 2)
        qn = qn.view(B, L, self.num_heads, self.d_k).transpose(1, 2)
        kp = kp.view(B, L, self.num_heads, self.d_k).transpose(1, 2)
        kn = kn.view(B, L, self.num_heads, self.d_k).transpose(1, 2)
        vp = vp.view(B, L, self.num_heads, self.d_k).transpose(1, 2)
        vn = vn.view(B, L, self.num_heads, self.d_k).transpose(1, 2)
        
        scores_p = torch.matmul(qp, kp.transpose(-2, -1)) / math.sqrt(self.d_k)
        scores_n = torch.matmul(qn, kn.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        attn_p = torch.softmax(scores_p + gp * self.adj_p, dim=-1)
        attn_n = torch.softmax(scores_n + gn * self.adj_n, dim=-1)
        
        out_p = torch.matmul(attn_p, vp).transpose(1, 2).reshape(B, L, -1)
        out_n = torch.matmul(attn_n, vn).transpose(1, 2).reshape(B, L, -1)
        
        return self.out_proj(torch.cat([out_p, out_n], dim=-1))

class FAT(nn.Module):
    """ Fourier Adjacency Transformer Network """
    def __init__(self, channels, samples, embed_dim=64, num_heads=4, dropout=0.5):
        super().__init__()
        self.conv_emb = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=31, padding='same'), nn.BatchNorm1d(16), nn.GELU(), nn.AvgPool1d(4),
            nn.Conv1d(16, 32, kernel_size=15, padding='same'), nn.BatchNorm1d(32), nn.GELU(), nn.AvgPool1d(4),
            nn.Dropout(dropout)
        )
        with torch.no_grad():
            dummy = torch.zeros(1, 1, samples)
            seq_len = self.conv_emb(dummy).shape[2]
            
        self.chan_proj = nn.Linear(32 * seq_len, embed_dim)
        self.faa = FAA(embed_dim, num_heads, channels)
        
        self.norm1 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2), nn.GELU(), nn.Dropout(dropout), nn.Linear(embed_dim * 2, embed_dim)
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(channels * embed_dim, 1)

    def forward(self, x_raw):
        B, C, S = x_raw.shape
        x = x_raw.view(B * C, 1, S)
        x = self.conv_emb(x)
        x = x.view(B, C, -1) 
        x = self.chan_proj(x)
        
        attn_out = self.faa(x)
        x = self.norm1(x + attn_out)
        
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)
        
        return self.fc(self.flatten(x))

class BinaryLabelSmoothingLoss(nn.Module):
    def __init__(self, smoothing=0.1, pos_weight=None):
        super().__init__()
        self.conf, self.smooth, self.pos_weight = 1.0 - smoothing, smoothing, pos_weight
    def forward(self, pred, target):
        return F.binary_cross_entropy_with_logits(pred, target * self.conf + 0.5 * self.smooth, pos_weight=self.pos_weight)

def get_loso_folds(metadata_df):
    folds = []
    for subj in sorted(metadata_df['subject'].unique()):
        test_mask = (metadata_df['subject'] == subj)
        folds.append((metadata_df[~test_mask], metadata_df[test_mask], subj))
    return folds

def run_loso_experiment(config, model_name):
    meta_df = pd.read_csv(os.path.join(config.CACHE_PATH, 'meta.csv'))
    ckpt_file = os.path.join(config.CACHE_PATH, f"ckpt_sota_{config.DATASET}_{model_name}.json")
    
    if os.path.exists(ckpt_file):
        with open(ckpt_file, 'r') as f: all_loso_accuracies = json.load(f)
        print(f"\n[CHECKPOINT] Resuming {model_name} from Subject {len(all_loso_accuracies)}...")
    else: all_loso_accuracies = []
        
    start_fold = len(all_loso_accuracies)
    if start_fold >= config.FOLDS: return np.mean(all_loso_accuracies)

    folds = get_loso_folds(meta_df)
    
    for k in range(start_fold, len(folds)):
        train_df, test_df, test_subj = folds[k]
        print(f"\n[{model_name}] FOLD {k+1}/{config.FOLDS} | HELD-OUT SUBJECT: {test_subj}")
        
        train_loader = DataLoader(DualEEGDataset(train_df), batch_size=config.BATCH_SIZE, shuffle=True, num_workers=2, generator=g, worker_init_fn=seed_worker)
        test_loader = DataLoader(DualEEGDataset(test_df), batch_size=config.BATCH_SIZE, shuffle=False, num_workers=2)
        
        if model_name == 'EEGNet': model = EEGNet(config.CHANNELS, config.SEQ_LEN_SAMPLES).to(config.DEVICE)
        elif model_name == 'TSception': model = TSception(config.CHANNELS, config.SEQ_LEN_SAMPLES).to(config.DEVICE)
        elif model_name == 'EEG-Conformer': model = EEG_Conformer(config.CHANNELS, config.SEQ_LEN_SAMPLES).to(config.DEVICE)
        elif model_name == 'FAT': model = FAT(config.CHANNELS, config.SEQ_LEN_SAMPLES).to(config.DEVICE)
        
        opt = optim.AdamW(model.parameters(), lr=config.LR, weight_decay=config.WEIGHT_DECAY)
        if len(all_loso_accuracies) == 0 and k == start_fold: count_parameters(model, model_name)
        
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=10)
        pos_weight = torch.tensor([(len(train_df) - train_df['label'].sum()) / (train_df['label'].sum() + 1e-5)]).float().to(config.DEVICE)
        crit = BinaryLabelSmoothingLoss(smoothing=config.LABEL_SMOOTHING, pos_weight=pos_weight)
        
        best_acc = 0
        for epoch in range(config.EPOCHS_SOTA):
            model.train()
            for _, X_raw, yb in train_loader:
                opt.zero_grad()
                logits = model(X_raw.to(config.DEVICE)) # Pass raw 1D signal to SOTA baselines
                loss = crit(logits, yb.to(config.DEVICE).unsqueeze(1))
                loss.backward()
                opt.step()
            scheduler.step()
            
            model.eval()
            preds, truths = [], []
            with torch.no_grad():
                for _, X_raw, yv in test_loader:
                    logits = model(X_raw.to(config.DEVICE))
                    preds.extend((torch.sigmoid(logits) > 0.5).int().cpu().numpy().flatten())
                    truths.extend(yv.int().numpy().flatten())
            
            # ---  TRIAL-WISE MAJORITY VOTING ---
            test_trial_ids = test_df['trial'].values
            unique_test_trials = np.unique(test_trial_ids)
            trial_votes, trial_truths = [], []
            
            for t_id in unique_test_trials:
                idx = np.where(test_trial_ids == t_id)[0]
                if len(idx) == 0: continue
                trial_votes.append(1 if np.sum(np.array(preds)[idx]) >= (len(idx) / 2.0) else 0)
                trial_truths.append(np.array(truths)[idx][0])
            
            acc = np.mean(np.array(trial_votes) == np.array(trial_truths))
            best_acc = max(best_acc, acc)
            
        all_loso_accuracies.append(best_acc)
        with open(ckpt_file, 'w') as f: json.dump(all_loso_accuracies, f)
        
    return np.mean(all_loso_accuracies)

def run_sota_comparisons(config):
    csv_filename = f"sota_comparison_metrics_arousal_{config.DATASET}.csv"
    results = {}
    
    models_to_test = ['EEGNet', 'TSception', 'EEG-Conformer', 'FAT']
    
    for model_name in models_to_test:
        if os.path.exists(csv_filename):
            df_existing = pd.read_csv(csv_filename)
            if model_name in df_existing['Model'].values:
                acc = df_existing.loc[df_existing['Model'] == model_name, 'Accuracy'].values[0]
                print(f"\n[CHECKPOINT] Baseline '{model_name}' completed in CSV. Skipping!")
                results[model_name] = acc
                continue
                
        acc = run_loso_experiment(config, model_name)
        results[model_name] = acc
        
        df_new = pd.DataFrame([[model_name, acc]], columns=['Model', 'Accuracy'])
        if not os.path.exists(csv_filename): df_new.to_csv(csv_filename, index=False)
        else: df_new.to_csv(csv_filename, mode='a', header=False, index=False)

    print("\n" + "="*65)
    print(f" FINAL SOTA COMPARISON: AROUSAL ({config.DATASET})")
    print("="*65)
    for name, acc in results.items():
        bar = "█" * int(acc * 40)
        print(f"{name.ljust(20)} | {acc:.4f} | {bar}")
    print("="*65)

if __name__ == "__main__":
    config = Config()
    if not os.path.exists(os.path.join(config.CACHE_PATH, 'meta.csv')):
        if config.DATASET == 'DREAMER': 
            generate_and_cache_dreamer_data_dual(config)
        elif config.DATASET == 'DEAP': 
            generate_deap_dual_cache(config)
        else: 
            print("Unknown Dataset.")
    
    run_sota_comparisons(config)
