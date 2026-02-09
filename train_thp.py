import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
import numpy as np
import argparse
import os
import random
import time

# ==========================================
# 0. Configuration and Utilities
# ==========================================
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def calculate_metrics(pred_matrix, gt_matrix, threshold=0.0):
    if isinstance(pred_matrix, np.ndarray) and pred_matrix.dtype == object:
        pred_matrix = pred_matrix.astype(float)
    if isinstance(gt_matrix, np.ndarray) and gt_matrix.dtype == object:
        gt_matrix = pd.DataFrame(gt_matrix).apply(pd.to_numeric, errors='coerce').fillna(0).values

    pred_binary = (pred_matrix > threshold).astype(int)
    gt_binary = (gt_matrix > 0).astype(int)

    tp = np.sum((pred_binary == 1) & (gt_binary == 1))
    fp = np.sum((pred_binary == 1) & (gt_binary == 0))
    fn = np.sum((pred_binary == 0) & (gt_binary == 1))
    
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
    
    shd = fp + fn
    nhd = shd / (gt_matrix.shape[0] * gt_matrix.shape[1] + 1e-8)
    
    return precision, recall, f1, shd, nhd

# ==========================================
# 1. Data Processing
# ==========================================
def load_data(file_path_data, file_path_answer):
    if not os.path.exists(file_path_data):
        raise FileNotFoundError(f"Data file not found: {file_path_data}")
    df = pd.read_csv(file_path_data)
    
    if not os.path.exists(file_path_answer):
        return df, np.zeros((5, 5))
    
    try:
        df_gt = pd.read_csv(file_path_answer, header=None)
        try: pd.to_numeric(df_gt.iloc[0, 0])
        except ValueError: df_gt = pd.read_csv(file_path_answer, header=0)
        gt_matrix = df_gt.apply(pd.to_numeric, errors='coerce').fillna(0).values
    except:
        gt_matrix = np.zeros((5, 5))
    return df, gt_matrix

def process_sequences(df):
    df = df.sort_values(by='time_stamp').reset_index(drop=True)
    event_types = sorted(df['event_type'].unique())
    event2idx = {e: i for i, e in enumerate(event_types)}
    df['event_idx'] = df['event_type'].map(event2idx)
    
    time_deltas = df['time_stamp'].diff().fillna(1.0).values
    time_deltas = np.log(time_deltas + 1e-6)
    
    events = df['event_idx'].values
    seq_len = 20
    data_list = []
    
    if len(df) <= seq_len: return [], len(event_types)

    for i in range(seq_len, len(df)):
        data_list.append((
            events[i-seq_len:i], 
            time_deltas[i-seq_len:i], 
            time_deltas[i], 
            events[i]
        ))
    return data_list, len(event_types)

# ==========================================
# 2. Model Definition (Modified)
# ==========================================
class ConditionalDiffusion(nn.Module):
    def __init__(self, num_event_types, hidden_dim=64, num_steps=50):
        super().__init__()
        self.num_event_types = num_event_types
        self.num_steps = num_steps
        self.hidden_dim = hidden_dim
        
        self.embedding = nn.Embedding(num_event_types + 1, hidden_dim)
        self.lstm = nn.LSTM(hidden_dim + 1, hidden_dim, batch_first=True)
        
        self.betas = torch.linspace(1e-4, 0.02, num_steps)
        self.alphas = 1. - self.betas
        self.alphas_bar = torch.cumprod(self.alphas, dim=0)
        
        self.time_embed = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        
        input_dim = 1 + hidden_dim + hidden_dim
        
        self.denoise_mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim * 2),
            nn.SiLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1)
        )

    def get_condition_embedding(self, event_seq, dt_seq):
        evt_emb = self.embedding(event_seq)
        dt_in = dt_seq.unsqueeze(-1)
        lstm_in = torch.cat([evt_emb, dt_in], dim=-1)
        _, (h_n, _) = self.lstm(lstm_in)
        return h_n[-1]

    def forward(self, event_seq, dt_seq, target_dt):
        batch_size = event_seq.shape[0]
        device = event_seq.device
        
        condition = self.get_condition_embedding(event_seq, dt_seq)
        t = torch.randint(0, self.num_steps, (batch_size,), device=device).long()
        
        if target_dt.dim() == 1:
            target_dt = target_dt.unsqueeze(-1)
            
        noise = torch.randn_like(target_dt)
        
        alpha_bar_t = self.alphas_bar.to(device)[t].unsqueeze(-1)
        
        x_0 = target_dt
        x_t = torch.sqrt(alpha_bar_t) * x_0 + torch.sqrt(1 - alpha_bar_t) * noise
        
        t_emb = self.time_embed(t.float().unsqueeze(-1))
        
        net_in = torch.cat([x_t, t_emb, condition], dim=-1)
        
        pred_noise = self.denoise_mlp(net_in)
        
        return F.mse_loss(pred_noise, noise)

    def calculate_nll(self, event_seq, dt_seq, target_dt):
        self.eval()
        with torch.no_grad():
            losses = []
            for _ in range(3):
                loss = self.forward(event_seq, dt_seq, target_dt)
                losses.append(loss.item())
        return np.mean(losses)

# ==========================================
# 3. Main Function
# ==========================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--nodes', type=int, required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--epoch', type=int, default=10)
    parser.add_argument('--threshold', type=float, default=0.005)
    args = parser.parse_args()
    
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using Device: {device}", flush=True)

    data_file = f"synthetic_event_sequence_hawkes_{args.nodes}.csv"
    answer_file = f"synth_{args.nodes}_answer.csv"
    
    try:
        df_raw, gt_matrix = load_data(data_file, answer_file)
        data_list, num_events = process_sequences(df_raw)
        
        print("Converting data to tensors...", flush=True)
        all_evts = torch.tensor(np.array([d[0] for d in data_list]), dtype=torch.long).to(device)
        all_dts = torch.tensor(np.array([d[1] for d in data_list]), dtype=torch.float32).to(device)
        target_dts = torch.tensor(np.array([d[2] for d in data_list]), dtype=torch.float32).to(device)
        target_types = torch.tensor(np.array([d[3] for d in data_list]), dtype=torch.long).to(device)
        
        dataset = torch.utils.data.TensorDataset(all_evts, all_dts, target_dts)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)
        
        # 2. Train model
        model = ConditionalDiffusion(num_event_types=num_events).to(device)
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        
        print("Training Started... (Please wait)", flush=True)
        model.train()
        
        start_time = time.time()
        for e in range(args.epoch):
            epoch_loss = 0
            for i, batch in enumerate(dataloader):
                optimizer.zero_grad()
                loss = model(batch[0], batch[1], batch[2])
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            
            elapsed = time.time() - start_time
            print(f"Epoch {e+1}/{args.epoch} | Loss: {epoch_loss:.4f} | Time: {elapsed:.1f}s", flush=True)

        # 3. Causality analysis using perturbation
        print("Analyzing Causality (Perturbation)...", flush=True)
        base_losses = np.zeros(num_events)
        counts = np.zeros(num_events)
        
        model.eval()
        
        with torch.no_grad():
            for i in range(len(all_evts)):
                tgt_type = target_types[i].item()
                loss = model(all_evts[i:i+1], all_dts[i:i+1], target_dts[i:i+1]).item()
                base_losses[tgt_type] += loss
                counts[tgt_type] += 1
        
        base_losses = base_losses / (counts + 1e-9)
        inferred_matrix = np.zeros((num_events, num_events))
        
        for cause_evt in range(num_events):
            masked_evts = all_evts.clone()
            masked_evts[masked_evts == cause_evt] = num_events 
            
            perturbed_losses = np.zeros(num_events)
            with torch.no_grad():
                for i in range(len(masked_evts)):
                    tgt_type = target_types[i].item()
                    loss = model(masked_evts[i:i+1], all_dts[i:i+1], target_dts[i:i+1]).item()
                    perturbed_losses[tgt_type] += loss
            
            perturbed_losses = perturbed_losses / (counts + 1e-9)
            inferred_matrix[cause_evt, :] = perturbed_losses - base_losses
            
        inferred_matrix = np.maximum(inferred_matrix, 0)
        
        min_dim = min(inferred_matrix.shape[0], gt_matrix.shape[0])
        inferred_matrix = inferred_matrix[:min_dim, :min_dim]
        gt_matrix = gt_matrix[:min_dim, :min_dim]
        
        prec, rec, f1, shd, nhd = calculate_metrics(inferred_matrix, gt_matrix, args.threshold)
        
        print(f"{args.nodes},{args.seed},{prec:.4f},{rec:.4f},{f1:.4f},{shd:.4f},{nhd:.4f}", flush=True)
        
        save_filename = f"learned_adj_matrix_n{args.nodes}_s{args.seed}.csv"
        np.savetxt(save_filename, inferred_matrix, delimiter=",", fmt='%.6f')
        print(f"Adjacency Matrix saved to: {save_filename}")

    except Exception:
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()