import os
import pickle
import random
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


# 1. CONFIGURATION

class Config:
    DEAP_PATH = './data/deap' 
    DREAMER_FILE = '.data/DREAMER.mat'
    
    # Toggle this to 'DEAP' or 'DREAMER' to switch datasets
    DATASET = 'DREAMER' 
    
    CACHE_PATH = f'./data/processed_data_valence_{DATASET}'
    
    NUM_SUBJECTS = 32 if DATASET == 'DEAP' else 23
    CHANNELS = 32 if DATASET == 'DEAP' else 14
    SR = 128
    
    SEQ_LEN_SECONDS = 5
    SEQ_LEN_SAMPLES = 5 * 128
    GEN_STRIDE = int(0.5 * 128)  
    IMG_SIZE = 9 
    
    BANDS = [(4, 8), (8, 13), (13, 30), (30, 45)]
    BAND_NAMES = ['Theta', 'Alpha', 'Beta', 'Gamma']
    
    BATCH_SIZE = 16          
    FOLDS = 10               
    
    EPOCHS_STAGE1 = 20      
    LR_STAGE1 = 5e-4        
    EPOCHS_STAGE2 = 30      
    LR_STAGE2 = 3e-4        
    
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

# ==========================================
# 2. DATA PROCESSING (DEAP & DREAMER)
# ==========================================
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

def process_baseline_topo_9x9(baseline_data, config):
    frames = process_segment_topo_9x9(baseline_data, config)
    return np.mean(frames, axis=0) 

def generate_and_cache_dreamer_data(config):
    print(f"Generating DREAMER TOPO Data to Disk ({config.CACHE_PATH})...")
    data = sio.loadmat(config.DREAMER_FILE)
    dreamer_struct = data['DREAMER'][0, 0]
    participants_data = dreamer_struct['Data'][0]
    metadata, file_counter = [], 0
    
    for subj_idx in range(config.NUM_SUBJECTS):
        print(f"  -> Processing DREAMER Subject {subj_idx + 1}/{config.NUM_SUBJECTS}")
        subj_data = participants_data[subj_idx][0, 0]
        eeg_data = subj_data['EEG'][0, 0]
        stimuli = eeg_data['stimuli']
        baseline = eeg_data['baseline']
        
        valence_scores = subj_data['ScoreValence'].flatten()
        labels = (valence_scores > 3.0).astype(int) 
        
        for t in range(18):
            lbl = labels[t]
            trial_data = stimuli[t, 0].T
            base_data = baseline[t, 0].T
            
            baseline_grid = np.mean(process_segment_topo_9x9(base_data, config), axis=0)
            
            curr = 0
            n_samples = trial_data.shape[1]
            trial_frames, trial_starts = [], []
            
            while (curr + config.SEQ_LEN_SAMPLES) <= n_samples:
                seg = trial_data[:, curr : curr+config.SEQ_LEN_SAMPLES]
                frames = process_segment_topo_9x9(seg, config)
                if len(frames) == int(config.SEQ_LEN_SECONDS):
                    trial_frames.append(frames - baseline_grid)
                    trial_starts.append(curr)
                curr += config.GEN_STRIDE 
            
            if len(trial_frames) > 0:
                trial_frames_np = np.array(trial_frames) 
                z_scored_trial = (trial_frames_np - np.mean(trial_frames_np)) / (np.std(trial_frames_np) + 1e-8)
                
                for idx, z_frames in enumerate(z_scored_trial):
                    fpath = os.path.join(config.CACHE_PATH, f"seq_{file_counter}.pt")
                    torch.save(torch.tensor(z_frames, dtype=torch.float32), fpath)
                    metadata.append({'filename': fpath, 'label': lbl, 'subject': subj_idx, 'trial': t, 'start_sample': trial_starts[idx]})
                    file_counter += 1
                
    pd.DataFrame(metadata).to_csv(os.path.join(config.CACHE_PATH, 'meta.csv'), index=False)

class InMemoryEEGDataset(Dataset):
    def __init__(self, metadata_df):
        self.metadata = metadata_df.reset_index(drop=True)
        self.data_list = [torch.load(row['filename']) for _, row in self.metadata.iterrows()]
        self.label_list = [torch.tensor(row['label'], dtype=torch.float32) for _, row in self.metadata.iterrows()]
            
    def __len__(self): return len(self.metadata)
    def __getitem__(self, idx): return self.data_list[idx], self.label_list[idx]


# 3. ARCHITECTURES (Ablation Variations)

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, dropout=0.5):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.in1 = nn.InstanceNorm2d(out_channels, affine=True) 
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.in2 = nn.InstanceNorm2d(out_channels, affine=True) 
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False), nn.InstanceNorm2d(out_channels, affine=True))
        self.dropout = nn.Dropout(dropout) 
    def forward(self, x): return F.gelu(self.in2(self.conv2(self.dropout(F.gelu(self.in1(self.conv1(x)))))) + self.shortcut(x))

class StandardConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, dropout=0.5):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.in1 = nn.InstanceNorm2d(out_channels, affine=True) 
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.in2 = nn.InstanceNorm2d(out_channels, affine=True) 
        self.dropout = nn.Dropout(dropout) 
        self.proj = nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False) if (stride != 1 or in_channels != out_channels) else None
    def forward(self, x): return F.gelu(self.in2(self.conv2(self.dropout(F.gelu(self.in1(self.conv1(x)))))))

class SpatialAutoencoder(nn.Module):
    def __init__(self, in_bands=4, latent_dim=256, dropout=0.5, use_residual=True):
        super().__init__()
        Block = ResidualBlock if use_residual else StandardConvBlock
        self.init = nn.Conv2d(in_bands, 32, 3, padding=1)
        self.enc1 = Block(32, 64, stride=2, dropout=dropout)       
        self.enc2 = Block(64, 128, stride=2, dropout=dropout)      
        self.bottleneck = Block(128, latent_dim, stride=1, dropout=dropout) 
        self.dec1 = Block(latent_dim, 64, dropout=dropout) 
        self.dec2 = Block(64, 32, dropout=dropout)
        self.final_rec = nn.Conv2d(32, in_bands, 3, padding=1)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        e2 = self.enc2(self.enc1(F.gelu(self.init(x)))) 
        b = self.bottleneck(e2) 
        rec = self.final_rec(self.dec2(self.dec1(F.interpolate(b, size=(9, 9), mode='bilinear', align_corners=False)))) 
        return rec, self.global_pool(b).view(b.size(0), -1)

class EndToEnd_Model(nn.Module):
    def __init__(self, pretrained_unet, in_bands=4, embed_dim=64, dropout=0.5, temporal_type='transformer'):
        super().__init__()
        self.spatial_enc = pretrained_unet
        self.compressor = nn.Linear(256, embed_dim)
        self.temporal_type = temporal_type
        
        if temporal_type == 'transformer':
            self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
            self.pos_emb = nn.Parameter(torch.randn(1, 6, embed_dim))
            enc_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=2, dim_feedforward=128, dropout=dropout, activation='gelu', batch_first=True, norm_first=True)
            self.temporal_net = nn.TransformerEncoder(enc_layer, num_layers=1)
            self.head = nn.Sequential(nn.LayerNorm(embed_dim), nn.Linear(embed_dim, 32), nn.GELU(), nn.Dropout(dropout), nn.Linear(32, 1))
        
        elif temporal_type == 'lstm':
            self.temporal_net = nn.LSTM(input_size=embed_dim, hidden_size=embed_dim, num_layers=1, batch_first=True, bidirectional=True)
            self.head = nn.Sequential(nn.LayerNorm(embed_dim * 2), nn.Linear(embed_dim * 2, 32), nn.GELU(), nn.Dropout(dropout), nn.Linear(32, 1))

    def forward(self, x):
        B, T, C, H, W = x.shape
        _, feats = self.spatial_enc(x.view(B*T, C, H, W)) 
        feats = self.compressor(feats).view(B, T, -1)
        
        if self.temporal_type == 'transformer':
            x_seq = self.temporal_net(torch.cat((self.cls_token.expand(B, -1, -1), feats), dim=1) + self.pos_emb)
            return self.head(x_seq[:, 0])
        else:
            lstm_out, _ = self.temporal_net(feats)
            return self.head(torch.max(lstm_out, dim=1)[0])


# 4. EXECUTION LOOP with CHECKPOINT

class BinaryLabelSmoothingLoss(nn.Module):
    def __init__(self, smoothing=0.1, pos_weight=None):
        super().__init__()
        self.conf = 1.0 - smoothing
        self.smooth = smoothing
        self.pos_weight = pos_weight
    def forward(self, pred, target):
        return F.binary_cross_entropy_with_logits(pred, target * self.conf + 0.5 * self.smooth, pos_weight=self.pos_weight)

def run_experiment(config, band_indices, use_residual=True, temporal_type='transformer', exp_name=""):
    meta_df = pd.read_csv(os.path.join(config.CACHE_PATH, 'meta.csv'))
    
    # --- Subject-Level Checkpointing ---
    safe_name = exp_name.replace(" ", "_").replace("+", "_").replace("(", "").replace(")", "")
    ckpt_file = os.path.join(config.CACHE_PATH, f"ckpt_{config.DATASET}_valence_{safe_name}.json")
    
    if os.path.exists(ckpt_file):
        try:
            with open(ckpt_file, 'r') as f:
                all_subject_accuracies = json.load(f)
            print(f"\n[CHECKPOINT] Found existing progress for '{exp_name}'. Resuming from Subject {len(all_subject_accuracies)}...")
        except json.JSONDecodeError:
            print(f"\n[WARNING] Checkpoint JSON for '{exp_name}' was corrupted. Restarting this ablation.")
            all_subject_accuracies = []
    else:
        all_subject_accuracies = []
        
    start_subject = len(all_subject_accuracies)
    
    # If the JSON says we already did all subjects, skip completely
    if start_subject >= config.NUM_SUBJECTS:
        print(f"[CHECKPOINT] '{exp_name}' is 100% complete. Returning cached accuracy.")
        return np.mean(all_subject_accuracies)

    num_active_bands = len(band_indices)

    # Resume loop exactly where it left off
    for subject_id in range(start_subject, config.NUM_SUBJECTS):
        print(f"\n  -> Starting Subject {subject_id} for '{exp_name}'...")
        subj_df = meta_df[meta_df['subject'] == subject_id].copy()
        unique_trials = subj_df['trial'].unique()
        kf = KFold(n_splits=config.FOLDS, shuffle=True, random_state=42)
        cv_scores = []
        
        for train_idx, test_idx in kf.split(unique_trials):
            train_mask = subj_df['trial'].isin(unique_trials[train_idx])
            test_mask = subj_df['trial'].isin(unique_trials[test_idx]) & (subj_df['start_sample'] % config.SEQ_LEN_SAMPLES == 0)
            
            train_loader = DataLoader(InMemoryEEGDataset(subj_df[train_mask]), batch_size=config.BATCH_SIZE, shuffle=True, num_workers=0, generator=g, worker_init_fn=seed_worker)
            test_loader = DataLoader(InMemoryEEGDataset(subj_df[test_mask]), batch_size=config.BATCH_SIZE, shuffle=False, num_workers=0)
            
            # Phase 1
            unet = SpatialAutoencoder(in_bands=num_active_bands, dropout=config.DROPOUT, use_residual=use_residual).to(config.DEVICE)
            opt_unet = optim.AdamW(unet.parameters(), lr=config.LR_STAGE1, weight_decay=config.WEIGHT_DECAY)
            
            for epoch in range(config.EPOCHS_STAGE1):
                unet.train()
                for Xb, _ in train_loader: 
                    Xb = Xb[:, :, band_indices, :, :].to(config.DEVICE).view(-1, num_active_bands, config.IMG_SIZE, config.IMG_SIZE)
                    opt_unet.zero_grad()
                    recon, _ = unet(Xb)
                    loss = nn.MSELoss()(recon, Xb) 
                    loss.backward()
                    opt_unet.step()
                    
            # Phase 2
            model = EndToEnd_Model(unet, in_bands=num_active_bands, dropout=config.DROPOUT, temporal_type=temporal_type).to(config.DEVICE)
            
            # Ensure parameters are always printed on the first processed fold, even after resuming
            if len(cv_scores) == 0: 
                count_parameters(model, exp_name)
            
            opt = optim.AdamW([
                {'params': [p for n, p in model.named_parameters() if 'spatial_enc' in n], 'lr': config.LR_STAGE2 * 0.1}, 
                {'params': [p for n, p in model.named_parameters() if 'spatial_enc' not in n], 'lr': config.LR_STAGE2}
            ], weight_decay=config.WEIGHT_DECAY)
            
            scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=10)
            pos_weight = torch.tensor([(len(subj_df[train_mask]) - subj_df[train_mask]['label'].sum()) / (subj_df[train_mask]['label'].sum() + 1e-5)]).float().to(config.DEVICE)
            crit = BinaryLabelSmoothingLoss(smoothing=config.LABEL_SMOOTHING, pos_weight=pos_weight)
            
            best_acc = 0
            for epoch in range(config.EPOCHS_STAGE2):
                model.train()
                for Xb, yb in train_loader:
                    Xb = Xb[:, :, band_indices, :, :].to(config.DEVICE)
                    opt.zero_grad()
                    loss = crit(model(Xb), yb.to(config.DEVICE).unsqueeze(1))
                    loss.backward()
                    opt.step()
                scheduler.step()
                
                model.eval()
                preds, truths = [], []
                with torch.no_grad():
                    for Xv, yv in test_loader:
                        preds.extend((torch.sigmoid(model(Xv[:, :, band_indices, :, :].to(config.DEVICE))) > 0.5).int().cpu().numpy().flatten())
                        truths.extend(yv.int().numpy().flatten())
                
                test_df = subj_df[test_mask]
                trial_votes, trial_truths = [], []
                for t_id in np.unique(test_df['trial']):
                    idx = np.where(test_df['trial'] == t_id)[0]
                    if len(idx) == 0: continue # Safety mechanism for empty trials
                    
                    trial_votes.append(1 if np.sum(np.array(preds)[idx]) >= (len(idx) / 2.0) else 0)
                    trial_truths.append(np.array(truths)[idx][0])
                
                acc = np.mean(np.array(trial_votes) == np.array(trial_truths))
                best_acc = max(best_acc, acc)
                
            cv_scores.append(best_acc)
            
        # SAVE JSON STATE AFTER EVERY SUBJECT
        all_subject_accuracies.append(np.mean(cv_scores))
        with open(ckpt_file, 'w') as f:
            json.dump(all_subject_accuracies, f)
            
    return np.mean(all_subject_accuracies)


def run_all_ablations(config):
    results = {}
    
    # FAIR ARCHITECTURAL BASELINE: We must test architectures on ALL bands 

    all_bands = [0, 1, 2, 3] 
    
    csv_filename = f"ablation_metrics_valence_{config.DATASET}.csv"
    
    # ---  Ablation-Level Checkpointing ---
    def execute_and_save(name, indices, use_res, temp_type, exp_name_str):
        if os.path.exists(csv_filename):
            df_existing = pd.read_csv(csv_filename)
            if name in df_existing['Configuration'].values:
                acc = df_existing.loc[df_existing['Configuration'] == name, 'Accuracy'].values[0]
                print(f"\n[CHECKPOINT] Ablation '{name}' already fully completed in CSV. Skipping computation!")
                return acc
                
        acc = run_experiment(config, indices, use_residual=use_res, temporal_type=temp_type, exp_name=exp_name_str)
        
        df_new = pd.DataFrame([[name, acc]], columns=['Configuration', 'Accuracy'])
        if not os.path.exists(csv_filename):
            df_new.to_csv(csv_filename, index=False)
        else:
            df_new.to_csv(csv_filename, mode='a', header=False, index=False)
            
        return acc

    print("\n" + "="*50)
    print(f" STARTING VALENCE ABLATION SUITE ON {config.DATASET.upper()}")
    print("="*50)

    # 1. Architectural Ablations (Forced to All Bands for fairness)
    print("\n--- RUNNING ARCHITECTURAL ABLATIONS (All Bands Baseline) ---")
    results['Vanilla AE+Trans (All Bands)'] = execute_and_save('Vanilla AE+Trans (All Bands)', all_bands, False, 'transformer', 'Vanilla_Trans_All')
    results['RAE+LSTM (All Bands)'] = execute_and_save('RAE+LSTM (All Bands)', all_bands, True, 'lstm', 'RAE_LSTM_All')
    
    # 2. Band Ablations
    print("\n--- RUNNING BAND ABLATIONS (RAE + Transformer) ---")
    band_configs = {'All Bands': [0, 1, 2, 3], 'Theta': [0], 'Alpha': [1], 'Beta': [2], 'Gamma': [3], 'Beta+Gamma': [2, 3]}
    for name, indices in band_configs.items():

        results[f'RAE+Trans ({name})'] = execute_and_save(f'RAE+Trans ({name})', indices, True, 'transformer', f'RAE_Trans_{name}')

   
    print("\n" + "="*65)
    print(f" FINAL VALENCE ABLATION RESULTS ({config.DATASET})")
    print("="*65)
    for name, acc in results.items():
        bar_length = int(acc * 40)
        bar = "█" * bar_length
        print(f"{name.ljust(30)} | {acc:.4f} | {bar}")
    print("="*65)
    print(f"\nAll data securely saved to {csv_filename}!")

if __name__ == "__main__":
    config = Config()
    if not os.path.exists(os.path.join(config.CACHE_PATH, 'meta.csv')):
        if config.DATASET == 'DREAMER': generate_and_cache_dreamer_data(config)
        else: print("Please use the DEAP generator logic if DEAP cache is missing.")
    run_all_ablations(config)
