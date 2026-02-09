import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================
# 1. 데이터 준비 및 전처리 (Data Preprocessing)
# ==========================================

def load_and_process_data(df):
    """
    데이터프레임을 모델 입력용 텐서로 변환합니다.
    """
    # 1. 시간 순 정렬
    df = df.sort_values(by='time_stamp').reset_index(drop=True)
    
    # 2. 매핑 (String -> ID)
    event_types = sorted(df['event_type'].unique())
    devices = sorted(df['seq_id'].unique())
    
    event2idx = {e: i for i, e in enumerate(event_types)}
    device2idx = {d: i for i, d in enumerate(devices)}
    
    df['event_idx'] = df['event_type'].map(event2idx)
    df['device_idx'] = df['seq_id'].map(device2idx)
    
    # 3. 텐서 변환
    times = torch.tensor(df['time_stamp'].values, dtype=torch.float32)
    events = torch.tensor(df['event_idx'].values, dtype=torch.long)
    devices_tensor = torch.tensor(df['device_idx'].values, dtype=torch.long)
    
    return times, events, devices_tensor, event_types, devices

# ==========================================
# 2. THP 모델 정의 (Topological Hawkes Process)
# ==========================================

class THPModel(nn.Module):
    def __init__(self, num_nodes, num_event_types, decay_rate=1.0):
        super(THPModel, self).__init__()
        self.num_nodes = num_nodes
        self.num_event_types = num_event_types
        self.decay = decay_rate  # 시간 감쇠율 (Beta)
        
        # --- 학습 파라미터 ---
        
        # 1. Base Intensity (Mu): 각 이벤트의 기본 발생 확률
        self.mu = nn.Parameter(torch.rand(num_event_types) * 0.1)
        
        # 2. Event Causal Matrix (Alpha): 이벤트 타입 간의 인과관계 (핵심 결과물)
        # alpha[i, j]: 이벤트 i가 발생했을 때 이벤트 j를 얼마나 유발하는가
        self.alpha = nn.Parameter(torch.rand(num_event_types, num_event_types) * 0.1)
        
        # 3. Topology Matrix (Adjacency): 디바이스 간의 영향력
        # 데이터에 연결 정보가 없으므로 학습 가능하게 설정하거나, 
        # 모든 디바이스가 서로 연결되어 있다고 가정(Identity or Ones)
        # 여기서는 학습을 통해 디바이스 간 관계도 찾도록 설정합니다.
        self.adj = nn.Parameter(torch.eye(num_nodes) + 0.1) # 초기값: 자기 자신 + 약한 연결

    def get_causal_matrix(self):
        # 음수 방지를 위해 Softplus 적용하여 반환
        return torch.nn.functional.softplus(self.alpha)

    def forward(self, times, events, devices_seq):
        """
        Log-Likelihood 계산
        L = sum(log(lambda(ti))) - integral(lambda(t))
        """
        # 파라미터 양수 제약 (Intensity는 항상 양수여야 함)
        mu = torch.nn.functional.softplus(self.mu)
        alpha = torch.nn.functional.softplus(self.alpha)
        adj = torch.nn.functional.softplus(self.adj)
        
        seq_len = len(times)
        log_lambda_sum = 0
        integral_sum = 0
        
        # 계산 효율성을 위해 반복문 사용 (실제 대용량 데이터는 벡터화 필요)
        # t_i: 현재 이벤트 시점
        for i in range(seq_len):
            t_now = times[i]
            k_now = events[i]    # 현재 이벤트 타입
            u_now = devices_seq[i] # 현재 발생한 디바이스
            
            # 1. Intensity Calculation (lambda(t_now))
            # lambda_k(t) = mu_k + sum_{history} (Adj_uv * Alpha_kk' * Decay)
            
            intensity = mu[k_now]
            
            # 과거 이력(History)의 영향 합산
            if i > 0:
                # 과거 이벤트들
                hist_times = times[:i]
                hist_events = events[:i]
                hist_devices = devices_seq[:i]
                
                # 시간 차이 (Delta t)
                delta_t = t_now - hist_times
                
                # 감쇠 커널: exp(-beta * delta_t)
                kernel_val = torch.exp(-self.decay * delta_t)
                
                # 토폴로지 영향: Adj[u_now, u_past]
                # 현재 디바이스(u_now)가 과거 디바이스(u_past)로부터 영향을 받는지
                topo_effect = adj[u_now, hist_devices]
                
                # 이벤트 인과 영향: Alpha[past_event, current_event]
                event_effect = alpha[hist_events, k_now]
                
                # 총 영향력 합산
                excitation = torch.sum(topo_effect * event_effect * kernel_val)
                intensity += excitation
            
            # 로그 우도 합산 (log lambda)
            log_lambda_sum += torch.log(intensity + 1e-6) # log(0) 방지
            
        # 2. Integral Calculation (Total duration integral)
        # sum_{all history j} (Adj * Alpha / beta * (1 - exp(-beta * (T_end - t_j))))
        T_end = times[-1]
        term1 = mu.sum() * T_end # Baseline integral
        
        term2 = 0
        for j in range(seq_len):
            k_past = events[j]
            u_past = devices_seq[j]
            t_past = times[j]
            
            # 이 과거 이벤트가 미래의 모든 가능한 디바이스/이벤트에 미칠 영향 적분
            # 적분 공식: integral_t^T alpha * e^(-beta(t-tj)) dt = (alpha/beta) * (1 - e^(-beta(T-tj)))
            
            time_integral = (1 / self.decay) * (1 - torch.exp(-self.decay * (T_end - t_past)))
            
            # 모든 타겟 이벤트/디바이스에 대한 영향의 합
            sum_alpha = torch.sum(alpha[k_past, :]) # 이 이벤트가 유발할 모든 이벤트 합
            sum_adj = torch.sum(adj[:, u_past])     # 이 디바이스가 영향을 줄 모든 디바이스 합
            
            term2 += sum_adj * sum_alpha * time_integral

        integral_sum = term1 + term2
        
        # Negative Log Likelihood (Minimize하기 위해 음수 변환)
        loss = - (log_lambda_sum - integral_sum)
        return loss

# ==========================================
# 3. 메인 실행 코드 (Main Experiment)
# ==========================================

# 1. 사용자 데이터 입력
data = {
    'event_type': ['event_4', 'event_5', 'event_1', 'event_4', 'event_5', 'event_4', 'event_1'],
    'seq_id':     ['device_14', 'device_13', 'device_3', 'device_14', 'device_13', 'device_14', 'device_3'],
    'time_stamp': [8.846, 9.875, 15.063, 16.100, 17.200, 20.500, 25.100] # 예시 데이터 확장
}
df = pd.DataFrame(data)

print("--- [1] 데이터 로드 완료 ---")
print(df)

# 2. 전처리
times, events, devices_tensor, event_types, device_names = load_and_process_data(df)
num_events = len(event_types)
num_nodes = len(device_names)

# 3. 모델 초기화
model = THPModel(num_nodes=num_nodes, num_event_types=num_events, decay_rate=0.5)
optimizer = optim.Adam(model.parameters(), lr=0.1)

# 4. 학습 루프
print("\n--- [2] 학습 시작 (Training) ---")
model.train()
for epoch in range(100): # 100 Epoch
    optimizer.zero_grad()
    
    loss = model(times, events, devices_tensor)
    
    loss.backward()
    optimizer.step()
    
    if (epoch+1) % 20 == 0:
        print(f"Epoch {epoch+1:03d} | Loss: {loss.item():.4f}")

# 5. 결과 시각화 (인과관계 매트릭스)
print("\n--- [3] 학습 결과: Event 간 인과관계 (Alpha Matrix) ---")

# 학습된 Causal Matrix 추출
causal_matrix = model.get_causal_matrix().detach().numpy()

# 시각화
plt.figure(figsize=(8, 6))
sns.heatmap(causal_matrix, annot=True, fmt=".2f", cmap="Blues",
            xticklabels=event_types, yticklabels=event_types)
plt.title("Learned Causal Structure (Alpha Matrix)")
plt.xlabel("Target Event (Effect)")
plt.ylabel("Source Event (Cause)")
plt.show()

# 텍스트 출력
print("\n[해석] 행(Row) 이벤트가 발생했을 때, 열(Col) 이벤트가 뒤따라 발생할 강도:")
df_res = pd.DataFrame(causal_matrix, index=event_types, columns=event_types)
print(df_res)