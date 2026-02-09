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
from itertools import combinations

# ==========================================
# 0. Utilities and Configuration
# ==========================================
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def calculate_metrics(pred_matrix, gt_matrix, threshold=0.1):
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
# 1. Data Processing (Long Horizon Support)
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

def process_sequences(df, seq_len=100):
    """
    Process with long horizons and irregular timestamps
    """
    df = df.sort_values(by='time_stamp').reset_index(drop=True)
    event_types = sorted(df['event_type'].unique())
    event2idx = {e: i for i, e in enumerate(event_types)}
    df['event_idx'] = df['event_type'].map(event2idx)
    
    time_deltas = df['time_stamp'].diff().fillna(1.0).values
    
    # Remove outliers using IQR method before normalization
    q1, q3 = np.percentile(time_deltas, [25, 75])
    iqr = q3 - q1
    lower_bound = max(q1 - 1.5 * iqr, 1e-6)
    upper_bound = q3 + 1.5 * iqr
    time_deltas = np.clip(time_deltas, lower_bound, upper_bound)
    
    # Log transformation for stability
    time_deltas = np.log(time_deltas + 1e-6)
    
    # Robust standardization using median and MAD
    median = np.median(time_deltas)
    mad = np.median(np.abs(time_deltas - median))
    time_deltas = (time_deltas - median) / (mad + 1e-6)
    
    events = df['event_idx'].values
    data_list = []
    
    if len(df) <= seq_len:
        return [], len(event_types)

    for i in range(seq_len, len(df)):
        data_list.append((
            events[i-seq_len:i],
            time_deltas[i-seq_len:i],
            time_deltas[i],
            events[i]
        ))
    
    return data_list, len(event_types)

# ==========================================
# 2. Likelihood-based Diffusion Model (Stage 1)
# ==========================================
class DiffusionLikelihoodEstimator(nn.Module):
    """
    Diffusion-based likelihood estimator for temporal prediction
    This model learns P(next_event | history)
    """
    def __init__(self, num_event_types, hidden_dim=64, num_steps=50):
        super().__init__()
        self.num_event_types = num_event_types
        self.hidden_dim = hidden_dim
        self.num_steps = num_steps
        
        # Event & Time Embedding with LayerNorm
        self.event_emb = nn.Embedding(num_event_types + 1, hidden_dim, padding_idx=num_event_types)
        self.time_proj = nn.Linear(1, hidden_dim)
        self.input_norm = nn.LayerNorm(hidden_dim)
        
        # History Encoder (attention-based, no RNN)
        self.history_attn = nn.MultiheadAttention(hidden_dim, num_heads=4, batch_first=True, dropout=0.1)
        self.history_norm = nn.LayerNorm(hidden_dim)
        
        self.history_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )
        
        # Diffusion Components
        self.betas = torch.linspace(1e-4, 0.02, num_steps)
        self.alphas = 1. - self.betas
        self.alphas_bar = torch.cumprod(self.alphas, dim=0)
        
        self.time_embed = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        
        # Denoising Network - More expressive
        self.denoise_mlp = nn.Sequential(
            nn.Linear(1 + hidden_dim + hidden_dim, hidden_dim * 2),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1)
        )
        
    def encode_history(self, event_seq, dt_seq, event_mask=None):
        """
        Encode history with optional event masking for perturbation
        event_mask: (B, S) binary mask, 0 means mask out
        """
        batch_size, seq_len = event_seq.shape
        
        # Apply masking (replace masked events with padding idx)
        if event_mask is not None:
            event_seq = event_seq * event_mask.long() + (1 - event_mask.long()) * self.num_event_types
        
        # Embed with normalization
        h_emb = self.event_emb(event_seq)  # (B, S, H)
        t_emb = self.time_proj(dt_seq.unsqueeze(-1))  # (B, S, H)
        
        features = self.input_norm(h_emb + t_emb)  # (B, S, H)
        
        # Self-attention over history
        attn_out, _ = self.history_attn(features, features, features)
        features = self.history_norm(features + attn_out)
        
        # MLP
        features = features + self.history_mlp(features)
        
        # Aggregate (mean pooling)
        if event_mask is not None:
            # Masked mean
            mask_expanded = event_mask.unsqueeze(-1)  # (B, S, 1)
            context = (features * mask_expanded).sum(dim=1) / (mask_expanded.sum(dim=1) + 1e-8)
        else:
            context = features.mean(dim=1)  # (B, H)
        
        return context
    
    def forward(self, event_seq, dt_seq, target_dt, event_mask=None):
        """
        Standard forward pass for training
        """
        batch_size = event_seq.shape[0]
        device = event_seq.device
        
        # Encode history
        condition = self.encode_history(event_seq, dt_seq, event_mask)
        
        # Diffusion forward
        t = torch.randint(0, self.num_steps, (batch_size,), device=device).long()
        
        target_dt = target_dt.view(-1, 1)
        noise = torch.randn_like(target_dt)
        
        alpha_bar_t = self.alphas_bar.to(device)[t].unsqueeze(-1)
        x_0 = target_dt
        x_t = torch.sqrt(alpha_bar_t) * x_0 + torch.sqrt(1 - alpha_bar_t) * noise
        
        t_emb = self.time_embed(t.float().unsqueeze(-1))
        
        net_in = torch.cat([x_t, t_emb, condition], dim=-1)
        pred_noise = self.denoise_mlp(net_in)
        
        loss = F.mse_loss(pred_noise, noise)
        return loss
    
    def compute_likelihood(self, event_seq, dt_seq, target_dt, event_mask=None, num_samples=50):
        """
        Estimate log-likelihood using diffusion model
        Higher likelihood = better prediction
        """
        batch_size = event_seq.shape[0]
        device = event_seq.device
        
        with torch.no_grad():
            condition = self.encode_history(event_seq, dt_seq, event_mask)
            
            # Sample multiple noises for better estimation
            log_likelihoods = []
            
            for _ in range(num_samples):
                # Use final timestep for likelihood estimation
                t = torch.full((batch_size,), self.num_steps - 1, device=device).long()
                
                target_dt_exp = target_dt.view(-1, 1)
                noise = torch.randn_like(target_dt_exp)
                
                alpha_bar_t = self.alphas_bar.to(device)[t].unsqueeze(-1)
                x_t = torch.sqrt(alpha_bar_t) * target_dt_exp + torch.sqrt(1 - alpha_bar_t) * noise
                
                t_emb = self.time_embed(t.float().unsqueeze(-1))
                net_in = torch.cat([x_t, t_emb, condition], dim=-1)
                pred_noise = self.denoise_mlp(net_in)
                
                # Reconstruction error as proxy for likelihood
                error = F.mse_loss(pred_noise, noise, reduction='none').mean(dim=1)
                log_likelihood = -error
                log_likelihoods.append(log_likelihood)
            
            # Average over samples
            log_likelihood = torch.stack(log_likelihoods).mean(dim=0)
            
        return log_likelihood  # (B,)
# ==========================================
# 3. Granger Causality Test (Stage 2)
# ==========================================

# def granger_causality_test(model, dataloader, num_event_types, device, 
#                                    significance_level=0.05, 
#                                    num_likelihood_samples=50,
#                                    use_adaptive_threshold=True,
#                                    top_k_percent=0.3):
#     """
#     Improved Granger causality test with adaptive thresholding
#     """
#     print("Stage 2: Improved Granger Causality Testing...", flush=True)
    
#     model.eval()
    
#     # Pre-collect all data for efficiency
#     all_samples = {i: {'event_seq': [], 'dt_seq': [], 'target_dt': []} 
#                    for i in range(num_event_types)}
    
#     # Group by target type
#     for batch in dataloader:
#         event_seq, dt_seq, target_dt, target_types = [b.to(device) for b in batch]
        
#         for target_type in range(num_event_types):
#             mask = (target_types == target_type)
#             if mask.sum() > 0:
#                 all_samples[target_type]['event_seq'].append(event_seq[mask])
#                 all_samples[target_type]['dt_seq'].append(dt_seq[mask])
#                 all_samples[target_type]['target_dt'].append(target_dt[mask])
    
#     # Concatenate
#     for target_type in range(num_event_types):
#         if len(all_samples[target_type]['event_seq']) > 0:
#             all_samples[target_type]['event_seq'] = torch.cat(all_samples[target_type]['event_seq'])
#             all_samples[target_type]['dt_seq'] = torch.cat(all_samples[target_type]['dt_seq'])
#             all_samples[target_type]['target_dt'] = torch.cat(all_samples[target_type]['target_dt'])
    
#     causality_scores = np.zeros((num_event_types, num_event_types))
#     p_values = np.ones((num_event_types, num_event_types))
    
#     # Test all pairs
#     for source_type in range(num_event_types):
#         print(f"  Testing source={source_type}", flush=True)
        
#         for target_type in range(num_event_types):
#             if source_type == target_type:
#                 continue
            
#             if len(all_samples[target_type]['event_seq']) == 0:
#                 continue
            
#             event_seq = all_samples[target_type]['event_seq']
#             dt_seq = all_samples[target_type]['dt_seq']
#             target_dt = all_samples[target_type]['target_dt']
            
#             # Check if source appears in history
#             source_appears = (event_seq == source_type).any(dim=1)
#             if source_appears.sum() < 5:  # Need minimum samples
#                 continue
            
#             # Filter to samples where source appears
#             event_seq = event_seq[source_appears]
#             dt_seq = dt_seq[source_appears]
#             target_dt = target_dt[source_appears]
            
#             # Batch processing for speed
#             batch_size = 128
#             ll_with_list = []
#             ll_without_list = []
            
#             for i in range(0, len(event_seq), batch_size):
#                 batch_event = event_seq[i:i+batch_size]
#                 batch_dt = dt_seq[i:i+batch_size]
#                 batch_target = target_dt[i:i+batch_size]
                
#                 event_mask_without = (batch_event != source_type).float()
                
#                 # WITH source
#                 ll_with = model.compute_likelihood(
#                     batch_event, batch_dt, batch_target,
#                     event_mask=None,
#                     num_samples=num_likelihood_samples
#                 )
                
#                 # WITHOUT source
#                 ll_without = model.compute_likelihood(
#                     batch_event, batch_dt, batch_target,
#                     event_mask=event_mask_without,
#                     num_samples=num_likelihood_samples
#                 )
                
#                 ll_with_list.append(ll_with.cpu().numpy())
#                 ll_without_list.append(ll_without.cpu().numpy())
            
#             likelihood_with = np.concatenate(ll_with_list)
#             likelihood_without = np.concatenate(ll_without_list)
            
#             # Compute improvement
#             improvement = likelihood_with - likelihood_without
#             mean_improvement = improvement.mean()
            
#             # Statistical test with more lenient criteria
#             from scipy.stats import ttest_rel, wilcoxon
            
#             if len(improvement) > 1:
#                 # Use both t-test and effect size
#                 t_stat, p_val = ttest_rel(likelihood_with, likelihood_without, alternative='greater')
                
#                 # Cohen's d (effect size)
#                 cohens_d = mean_improvement / (improvement.std() + 1e-8)
                
#                 causality_scores[source_type, target_type] = mean_improvement
#                 p_values[source_type, target_type] = p_val
                
#                 # More lenient criterion: either significant p-value OR large effect size
#                 # This helps with recall
    
#     # Adaptive thresholding for better recall
#     if use_adaptive_threshold:
#         # Method 1: Top-k% edges by score
#         flat_scores = causality_scores.flatten()
#         threshold = np.percentile(flat_scores[flat_scores > 0], (1 - top_k_percent) * 100)
        
#         # Method 2: Combined criterion
#         # Accept edge if (score > threshold) OR (p-value < significance AND score > 0)
#         final_matrix = np.zeros_like(causality_scores)
#         for i in range(num_event_types):
#             for j in range(num_event_types):
#                 if i == j:
#                     continue
#                 score = causality_scores[i, j]
#                 p_val = p_values[i, j]
                
#                 # Relaxed criterion
#                 if score > threshold or (p_val < significance_level and score > 0):
#                     final_matrix[i, j] = score
        
#         return final_matrix
#     else:
#         # Original method
#         for i in range(num_event_types):
#             for j in range(num_event_types):
#                 if p_values[i, j] >= significance_level or causality_scores[i, j] <= 0:
#                     causality_scores[i, j] = 0
        
#         return causality_scores
    
    
def granger_causality_test(model, dataloader, num_event_types, device, 
                           significance_level=0.05, num_likelihood_samples=100):
    """
    Perform pairwise Granger causality test using controlled perturbations
    
    For each pair (source, target):
    1. Compute likelihood of target WITH source events
    2. Compute likelihood of target WITHOUT source events (mask out)
    3. If likelihood difference is significant, edge exists
    
    Uses both statistical significance AND effect size for better recall
    """
    print("Stage 2: Granger Causality Testing with Controlled Perturbations...", flush=True)
    
    model.eval()
    
    # Initialize causality matrix
    causality_scores = np.zeros((num_event_types, num_event_types))
    
    # Test each potential edge
    for source_type in range(num_event_types):
        for target_type in range(num_event_types):
            if source_type == target_type:
                continue
            
            # Collect samples where target appears
            likelihood_with = []
            likelihood_without = []
            
            for batch in dataloader:
                event_seq, dt_seq, target_dt, target_types = [b.to(device) for b in batch]
                
                # Filter: only consider samples where target is target_type
                mask = (target_types == target_type)
                if mask.sum() == 0:
                    continue
                
                event_seq = event_seq[mask]
                dt_seq = dt_seq[mask]
                target_dt = target_dt[mask]
                
                # Create event mask: mask out source_type
                event_mask_without = (event_seq != source_type).float()  # (B, S)
                
                # Likelihood WITH source
                ll_with = model.compute_likelihood(
                    event_seq, dt_seq, target_dt, 
                    event_mask=None,
                    num_samples=num_likelihood_samples
                )
                
                # Likelihood WITHOUT source (masked)
                ll_without = model.compute_likelihood(
                    event_seq, dt_seq, target_dt,
                    event_mask=event_mask_without,
                    num_samples=num_likelihood_samples
                )
                
                likelihood_with.append(ll_with.cpu().numpy())
                likelihood_without.append(ll_without.cpu().numpy())
            
            if len(likelihood_with) == 0:
                continue
            
            likelihood_with = np.concatenate(likelihood_with)
            likelihood_without = np.concatenate(likelihood_without)
            
            # Compute improvement: LL(with) - LL(without)
            improvement_array = likelihood_with - likelihood_without
            improvement = improvement_array.mean()
            
            # Statistical test (paired t-test) + Effect Size
            from scipy.stats import ttest_rel
            if len(likelihood_with) > 1:
                t_stat, p_value = ttest_rel(likelihood_with, likelihood_without, alternative='greater')
                
                # Compute Cohen's d (effect size)
                std_diff = improvement_array.std() + 1e-8
                cohens_d = improvement / std_diff
                
                # Relaxed criterion: Accept if either condition is met:
                # 1. Statistically significant with positive improvement
                # 2. Large effect size (Cohen's d > 0.5) with positive improvement
                # 3. Moderate effect size (Cohen's d > 0.3) AND p-value < 0.2
                if improvement > 0:
                    if (p_value < significance_level or 
                        cohens_d > 0.5 or 
                        (cohens_d > 0.3 and p_value < 0.2)):
                        causality_scores[source_type, target_type] = improvement
            else:
                # Not enough samples
                causality_scores[source_type, target_type] = 0
        
        print(f"  Tested source={source_type}", flush=True)
    
    return causality_scores

# ==========================================
# Fast Granger Causality Test (Vectorized)
# ==========================================
def granger_causality_test_fast(model, dataloader, num_event_types, device,
                                significance_level=0.05,
                                num_likelihood_samples=100,
                                min_support=10):
    """
    Safe and fast version with effect size criterion and minimum support
    """
    print("Stage 2: Safe Granger Causality Testing...", flush=True)
    
    model.eval()
    
    # Step 1: Collect all data
    print("  Collecting data...", flush=True)
    all_event_seq = []
    all_dt_seq = []
    all_target_dt = []
    all_target_types = []
    
    for batch in dataloader:
        event_seq, dt_seq, target_dt, target_types = [b.to(device) for b in batch]
        all_event_seq.append(event_seq)
        all_dt_seq.append(dt_seq)
        all_target_dt.append(target_dt)
        all_target_types.append(target_types)
    
    if len(all_event_seq) == 0:
        print("  WARNING: No data found!", flush=True)
        return np.zeros((num_event_types, num_event_types))
    
    full_event_seq = torch.cat(all_event_seq, dim=0)
    full_dt_seq = torch.cat(all_dt_seq, dim=0)
    full_target_dt = torch.cat(all_target_dt, dim=0)
    full_target_types = torch.cat(all_target_types, dim=0)
    
    print(f"  Total samples: {len(full_event_seq)}", flush=True)
    
    causality_scores = np.zeros((num_event_types, num_event_types))
    p_values = np.ones((num_event_types, num_event_types))
    
    # Step 2: Process by target
    for target_type in range(num_event_types):
        print(f"  Processing target={target_type}", flush=True)
        
        # Filter by target type
        target_mask = (full_target_types == target_type)
        n_target_samples = target_mask.sum().item()
        
        if n_target_samples < 5:
            print(f"    WARNING: Only {n_target_samples} samples, skipping", flush=True)
            continue
        
        t_event_seq = full_event_seq[target_mask]
        t_dt_seq = full_dt_seq[target_mask]
        t_target_dt = full_target_dt[target_mask]
        
        # Compute baseline likelihood (WITH all events) - batch processing
        batch_size = 128
        ll_with_all = []
        
        for i in range(0, len(t_event_seq), batch_size):
            batch_event = t_event_seq[i:i+batch_size]
            batch_dt = t_dt_seq[i:i+batch_size]
            batch_target = t_target_dt[i:i+batch_size]
            
            ll_batch = model.compute_likelihood(
                batch_event, batch_dt, batch_target,
                event_mask=None,
                num_samples=num_likelihood_samples
            )
            ll_with_all.append(ll_batch.cpu())
        
        ll_with_all = torch.cat(ll_with_all).numpy()
        
        # Test each source
        for source_type in range(num_event_types):
            if source_type == target_type:
                continue
            
            # Check if source appears in history with minimum support
            source_appears = (t_event_seq == source_type).any(dim=1)
            n_relevant = source_appears.sum().item()
            
            if n_relevant < min_support:
                continue
            
            # Filter to samples where source appears
            rel_event_seq = t_event_seq[source_appears]
            rel_dt_seq = t_dt_seq[source_appears]
            rel_target_dt = t_target_dt[source_appears]
            rel_ll_with = ll_with_all[source_appears.cpu().numpy()]
            
            # Compute likelihood WITHOUT source - batch processing
            ll_without_list = []
            
            for i in range(0, len(rel_event_seq), batch_size):
                batch_event = rel_event_seq[i:i+batch_size]
                batch_dt = rel_dt_seq[i:i+batch_size]
                batch_target = rel_target_dt[i:i+batch_size]
                
                # Mask out source
                event_mask = (batch_event != source_type).float()
                
                ll_batch = model.compute_likelihood(
                    batch_event, batch_dt, batch_target,
                    event_mask=event_mask,
                    num_samples=num_likelihood_samples
                )
                ll_without_list.append(ll_batch.cpu())
            
            ll_without = torch.cat(ll_without_list).numpy()
            
            # Statistical test with effect size
            improvement = rel_ll_with - ll_without
            mean_improvement = improvement.mean()
            
            if len(improvement) > 1:
                from scipy.stats import ttest_rel
                try:
                    t_stat, p_val = ttest_rel(rel_ll_with, ll_without, alternative='greater')
                    
                    # Compute Cohen's d (effect size)
                    std_improvement = improvement.std() + 1e-8
                    cohens_d = mean_improvement / std_improvement
                    
                    # Check for safe values
                    if np.isnan(mean_improvement) or np.isinf(mean_improvement):
                        mean_improvement = 0
                    if np.isnan(p_val) or np.isinf(p_val):
                        p_val = 1.0
                    if np.isnan(cohens_d) or np.isinf(cohens_d):
                        cohens_d = 0
                    
                    # Store scores and p-values
                    causality_scores[source_type, target_type] = mean_improvement
                    p_values[source_type, target_type] = p_val
                    
                    if not hasattr(granger_causality_test_fast, 'effect_sizes'):
                        granger_causality_test_fast.effect_sizes = {}
                    granger_causality_test_fast.effect_sizes[(source_type, target_type)] = cohens_d
                    
                except Exception as e:
                    print(f"    WARNING: Statistical test failed for ({source_type}, {target_type}): {e}", flush=True)
                    continue
    
    # Apply improved threshold with effect size and adaptive percentile
    effect_sizes = getattr(granger_causality_test_fast, 'effect_sizes', {})
    
    # Compute adaptive threshold based on score distribution
    all_scores = []
    for i in range(num_event_types):
        for j in range(num_event_types):
            if i != j and causality_scores[i, j] > 0:
                all_scores.append(causality_scores[i, j])
    
    if len(all_scores) > 0:
        # Use 50th percentile as minimum threshold (more strict than before)
        adaptive_threshold = np.percentile(all_scores, 50)
    else:
        adaptive_threshold = 0
    
    for i in range(num_event_types):
        for j in range(num_event_types):
            if i == j:
                continue
            
            score = causality_scores[i, j]
            p_val = p_values[i, j]
            cohens_d = effect_sizes.get((i, j), 0)
            
            # Improved criterion: Keep edge if meets ANY of:
            # 1. Strong evidence: (p < 0.05 AND Cohen's d > 0.3) OR Cohen's d > 0.8
            # 2. Moderate evidence: (p < 0.1 AND Cohen's d > 0.5) AND score > adaptive_threshold
            # 3. Weak evidence: (p < 0.05 AND score > 1.5 * adaptive_threshold)
            if score <= 0:
                causality_scores[i, j] = 0
            else:
                strong = (p_val < 0.05 and cohens_d > 0.3) or cohens_d > 0.8
                moderate = (p_val < 0.1 and cohens_d > 0.5 and score > adaptive_threshold)
                weak = (p_val < 0.05 and score > 1.5 * adaptive_threshold)
                
                if not (strong or moderate or weak):
                    causality_scores[i, j] = 0
    
    # Clean up
    if hasattr(granger_causality_test_fast, 'effect_sizes'):
        delattr(granger_causality_test_fast, 'effect_sizes')
    
    return causality_scores

# ==========================================
# Further optimization: Multi-GPU support
# ==========================================
def granger_causality_test_multi_gpu(model, dataloader, num_event_types, device,
                                     significance_level=0.05,
                                     num_likelihood_samples=20):
    """
    Multi-GPU accelerated version
    """
    import torch.multiprocessing as mp
    from torch.nn.parallel import DataParallel
    
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs", flush=True)
        model = DataParallel(model)
    
    # Same logic as fast version but with DataParallel
    # This automatically distributes batches across GPUs
    
    return granger_causality_test(model, dataloader, num_event_types, device, significance_level, num_likelihood_samples)


# ==========================================
# 4. Main Function
# ==========================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--nodes', type=int, required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--epoch', type=int, default=100)
    parser.add_argument('--significance', type=float, default=0.05)
    args = parser.parse_args()
    
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using Device: {device}", flush=True)

    data_file = f"synthetic_event_sequence_hawkes_{args.nodes}.csv"
    answer_file = f"synth_{args.nodes}_answer.csv"
    
    try:
        # 1. Load data
        df_raw, gt_matrix = load_data(data_file, answer_file)
        data_list, num_events = process_sequences(df_raw, seq_len=100)
        
        if len(data_list) == 0:
            print("ERROR: No valid sequences generated", flush=True)
            print(f"{args.nodes},{args.seed},0.0000,0.0000,0.0000,0.0000,0.0000", flush=True)
            return
        
        # Tensor conversion
        all_evts = torch.tensor(np.array([d[0] for d in data_list]), dtype=torch.long)
        all_dts = torch.tensor(np.array([d[1] for d in data_list]), dtype=torch.float32)
        target_dts = torch.tensor(np.array([d[2] for d in data_list]), dtype=torch.float32)
        target_types = torch.tensor(np.array([d[3] for d in data_list]), dtype=torch.long)
        
        # Check for NaN/Inf
        if torch.isnan(all_dts).any() or torch.isinf(all_dts).any():
            print("WARNING: NaN/Inf in time deltas, replacing...", flush=True)
            all_dts = torch.nan_to_num(all_dts, nan=0.0, posinf=10.0, neginf=-10.0)
        if torch.isnan(target_dts).any() or torch.isinf(target_dts).any():
            target_dts = torch.nan_to_num(target_dts, nan=0.0, posinf=10.0, neginf=-10.0)
        
        dataset = torch.utils.data.TensorDataset(all_evts, all_dts, target_dts, target_types)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True, drop_last=True)
        test_dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=False)
        
        print(f"Dataset size: {len(dataset)}, Event types: {num_events}", flush=True)
        
        # 2. Stage 1: Train with warmup
        print("Stage 1: Training Diffusion Likelihood Estimator...", flush=True)
        model = DiffusionLikelihoodEstimator(num_event_types=num_events, hidden_dim=128).to(device)
        optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
        
        # Warmup scheduler
        warmup_epochs = 10
        def lr_lambda(epoch):
            if epoch < warmup_epochs:
                return (epoch + 1) / warmup_epochs
            else:
                # Cosine annealing after warmup
                progress = (epoch - warmup_epochs) / (args.epoch - warmup_epochs)
                return 0.5 * (1 + np.cos(np.pi * progress))
        
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        
        model.train()
        start_time = time.time()
        best_loss = float('inf')
        patience_counter = 0
        
        # Gradient accumulation
        accumulation_steps = 2
        
        for e in range(args.epoch):
            epoch_loss = 0
            n_batches = 0
            
            optimizer.zero_grad()
            for batch_idx, batch in enumerate(dataloader):
                batch = [b.to(device) for b in batch]
                loss = model(batch[0], batch[1], batch[2])
                
                # Check for NaN
                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"  WARNING: NaN/Inf loss at epoch {e+1}, skipping batch", flush=True)
                    continue
                
                # Normalize loss for gradient accumulation
                loss = loss / accumulation_steps
                loss.backward()
                
                # Update weights every accumulation_steps
                if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == len(dataloader):
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    optimizer.zero_grad()
                
                epoch_loss += loss.item() * accumulation_steps
                n_batches += 1
            
            scheduler.step()
            
            if n_batches > 0:
                avg_loss = epoch_loss / n_batches
            else:
                avg_loss = 0
            
            # Track best model
            if avg_loss < best_loss:
                best_loss = avg_loss
                patience_counter = 0
            else:
                patience_counter += 1
            
            if (e+1) % 10 == 0 or e == 0:
                elapsed = time.time() - start_time
                lr = optimizer.param_groups[0]['lr']
                print(f"  Epoch {e+1}/{args.epoch} | Loss: {avg_loss:.4f} | LR: {lr:.6f} | Time: {elapsed:.1f}s", flush=True)
        
        # 3. Stage 2: Granger causality test with ensemble
        print("Stage 2: Ensemble Granger Causality Testing...", flush=True)
        
        # Run test multiple times with different random seeds for robustness
        num_ensemble = 3
        ensemble_matrices = []
        
        for ens_idx in range(num_ensemble):
            print(f"  Ensemble run {ens_idx+1}/{num_ensemble}...", flush=True)
            
            # Set different seed for each ensemble run
            torch.manual_seed(args.seed * 1000 + ens_idx)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(args.seed * 1000 + ens_idx)
            
            causality_matrix = granger_causality_test_fast(
                model, test_dataloader, num_events, device,
                significance_level=args.significance,
                num_likelihood_samples=100,
                min_support=15  # Require more samples for robust estimation
            )
            ensemble_matrices.append(causality_matrix)
        
        # Average ensemble results (only keep edges that appear in majority)
        causality_matrix = np.zeros_like(ensemble_matrices[0])
        for i in range(num_events):
            for j in range(num_events):
                if i == j:
                    continue
                # Count how many times this edge appears
                edge_count = sum(1 for m in ensemble_matrices if m[i, j] > 0)
                # Keep edge if it appears in at least 2 out of 3 runs
                if edge_count >= 2:
                    # Use average score from runs where edge exists
                    scores = [m[i, j] for m in ensemble_matrices if m[i, j] > 0]
                    causality_matrix[i, j] = np.mean(scores)
        
        # 4. Evaluate
        min_dim = min(causality_matrix.shape[0], gt_matrix.shape[0])
        causality_matrix = causality_matrix[:min_dim, :min_dim]
        gt_matrix = gt_matrix[:min_dim, :min_dim]
        
        # Check for NaN/Inf
        if np.isnan(causality_matrix).any() or np.isinf(causality_matrix).any():
            print("WARNING: NaN/Inf in causality matrix, cleaning...", flush=True)
            causality_matrix = np.nan_to_num(causality_matrix, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Metrics
        prec, rec, f1, shd, nhd = calculate_metrics(causality_matrix, gt_matrix, threshold=0.0)
        
        print(f"{args.nodes},{args.seed},{prec:.4f},{rec:.4f},{f1:.4f},{shd:.4f},{nhd:.4f}", flush=True)

    except Exception as e:
        import traceback
        print("ERROR occurred:", flush=True)
        traceback.print_exc()
        print(f"{args.nodes},{args.seed},0.0000,0.0000,0.0000,0.0000,0.0000", flush=True)

if __name__ == "__main__":
    main()