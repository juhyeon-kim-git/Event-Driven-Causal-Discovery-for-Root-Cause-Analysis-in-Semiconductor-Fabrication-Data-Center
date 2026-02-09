import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================
# 1. Data Preparation and Preprocessing
# ==========================================

def load_and_process_data(df):
    """
    Convert dataframe to model input tensors.
    """
    # 1. Sort by time
    df = df.sort_values(by='time_stamp').reset_index(drop=True)
    
    # 2. Mapping (String -> ID)
    event_types = sorted(df['event_type'].unique())
    devices = sorted(df['seq_id'].unique())
    
    event2idx = {e: i for i, e in enumerate(event_types)}
    device2idx = {d: i for i, d in enumerate(devices)}
    
    df['event_idx'] = df['event_type'].map(event2idx)
    df['device_idx'] = df['seq_id'].map(device2idx)
    
    # 3. Convert to tensors
    times = torch.tensor(df['time_stamp'].values, dtype=torch.float32)
    events = torch.tensor(df['event_idx'].values, dtype=torch.long)
    devices_tensor = torch.tensor(df['device_idx'].values, dtype=torch.long)
    
    return times, events, devices_tensor, event_types, devices

# ==========================================
# 2. THP Model Definition (Topological Hawkes Process)
# ==========================================

class THPModel(nn.Module):
    def __init__(self, num_nodes, num_event_types, decay_rate=1.0):
        super(THPModel, self).__init__()
        self.num_nodes = num_nodes
        self.num_event_types = num_event_types
        self.decay = decay_rate
        
        # --- Learnable Parameters ---
        
        # 1. Base Intensity (Mu): Base occurrence rate for each event
        self.mu = nn.Parameter(torch.rand(num_event_types) * 0.1)
        
        # 2. Event Causal Matrix (Alpha): Causal relationships between event types
        # alpha[i, j]: How much event i triggers event j
        self.alpha = nn.Parameter(torch.rand(num_event_types, num_event_types) * 0.1)
        
        # 3. Topology Matrix (Adjacency): Influence between devices
        # Learnable to discover device-to-device relationships
        self.adj = nn.Parameter(torch.eye(num_nodes) + 0.1)

    def get_causal_matrix(self):
        # Apply Softplus to ensure non-negative values
        return torch.nn.functional.softplus(self.alpha)

    def forward(self, times, events, devices_seq):
        """
        Log-Likelihood calculation
        L = sum(log(lambda(ti))) - integral(lambda(t))
        """
        # Ensure parameters are positive (Intensity must always be positive)
        mu = torch.nn.functional.softplus(self.mu)
        alpha = torch.nn.functional.softplus(self.alpha)
        adj = torch.nn.functional.softplus(self.adj)
        
        seq_len = len(times)
        log_lambda_sum = 0
        integral_sum = 0
        
        # Use loop for computational efficiency (vectorization needed for large-scale data)
        for i in range(seq_len):
            t_now = times[i]
            k_now = events[i]
            u_now = devices_seq[i]
            
            # 1. Intensity Calculation (lambda(t_now))
            # lambda_k(t) = mu_k + sum_{history} (Adj_uv * Alpha_kk' * Decay)
            
            intensity = mu[k_now]
            
            # Sum influence from historical events
            if i > 0:
                hist_times = times[:i]
                hist_events = events[:i]
                hist_devices = devices_seq[:i]
                
                # Time difference (Delta t)
                delta_t = t_now - hist_times
                
                # Decay kernel: exp(-beta * delta_t)
                kernel_val = torch.exp(-self.decay * delta_t)
                
                # Topology influence: Adj[u_now, u_past]
                topo_effect = adj[u_now, hist_devices]
                
                # Event causal influence: Alpha[past_event, current_event]
                event_effect = alpha[hist_events, k_now]
                
                # Sum total influence
                excitation = torch.sum(topo_effect * event_effect * kernel_val)
                intensity += excitation
            
            # Sum log likelihood (log lambda)
            log_lambda_sum += torch.log(intensity + 1e-6)
            
        # 2. Integral Calculation (Total duration integral)
        # sum_{all history j} (Adj * Alpha / beta * (1 - exp(-beta * (T_end - t_j))))
        T_end = times[-1]
        term1 = mu.sum() * T_end
        
        term2 = 0
        for j in range(seq_len):
            k_past = events[j]
            u_past = devices_seq[j]
            t_past = times[j]
            
            # Integral formula: integral_t^T alpha * e^(-beta(t-tj)) dt = (alpha/beta) * (1 - e^(-beta(T-tj)))
            time_integral = (1 / self.decay) * (1 - torch.exp(-self.decay * (T_end - t_past)))
            
            # Sum influence to all target events/devices
            sum_alpha = torch.sum(alpha[k_past, :])
            sum_adj = torch.sum(adj[:, u_past])
            
            term2 += sum_adj * sum_alpha * time_integral

        integral_sum = term1 + term2
        
        # Negative Log Likelihood
        loss = - (log_lambda_sum - integral_sum)
        return loss

# ==========================================
# 3. Main Experiment Code
# ==========================================

# 1. User data input
data = {
    'event_type': ['event_4', 'event_5', 'event_1', 'event_4', 'event_5', 'event_4', 'event_1'],
    'seq_id':     ['device_14', 'device_13', 'device_3', 'device_14', 'device_13', 'device_14', 'device_3'],
    'time_stamp': [8.846, 9.875, 15.063, 16.100, 17.200, 20.500, 25.100]
}
df = pd.DataFrame(data)

print("--- [1] Data Loaded ---")
print(df)

# 2. Preprocessing
times, events, devices_tensor, event_types, device_names = load_and_process_data(df)
num_events = len(event_types)
num_nodes = len(device_names)

# 3. Model initialization
model = THPModel(num_nodes=num_nodes, num_event_types=num_events, decay_rate=0.5)
optimizer = optim.Adam(model.parameters(), lr=0.1)

# 4. Training loop
print("\n--- [2] Training Started ---")
model.train()
for epoch in range(100):
    optimizer.zero_grad()
    
    loss = model(times, events, devices_tensor)
    
    loss.backward()
    optimizer.step()
    
    if (epoch+1) % 20 == 0:
        print(f"Epoch {epoch+1:03d} | Loss: {loss.item():.4f}")

# 5. Visualize results (Causal relationship matrix)
print("\n--- [3] Training Results: Event Causal Relationships (Alpha Matrix) ---")

# Extract learned Causal Matrix
causal_matrix = model.get_causal_matrix().detach().numpy()

# Visualization
plt.figure(figsize=(8, 6))
sns.heatmap(causal_matrix, annot=True, fmt=".2f", cmap="Blues",
            xticklabels=event_types, yticklabels=event_types)
plt.title("Learned Causal Structure (Alpha Matrix)")
plt.xlabel("Target Event (Effect)")
plt.ylabel("Source Event (Cause)")
plt.show()

# Text output
print("\n[Interpretation] When Row event occurs, intensity of subsequent Column event:")
df_res = pd.DataFrame(causal_matrix, index=event_types, columns=event_types)
print(df_res)