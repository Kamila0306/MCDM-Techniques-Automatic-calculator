import numpy as np
import pandas as pd

def run_vikor(df, criteria, weights, impacts, v=0.5):
    alt_col = df.columns[0]
    work_df = df.copy()
    work_df = work_df.rename(columns={alt_col: 'Alternative'})
    
    matrix = work_df[criteria].to_numpy().astype(float)
    weights = np.array(weights)
    
    f_plus = []
    f_minus = []
    for idx, imp in enumerate(impacts):
        col_data = matrix[:, idx]
        if imp == '+':
            f_plus.append(np.max(col_data))
            f_minus.append(np.min(col_data))
        else:
            f_plus.append(np.min(col_data))
            f_minus.append(np.max(col_data))
            
    f_plus = np.array(f_plus)
    f_minus = np.array(f_minus)
    
    num_alts = len(work_df)
    S = np.zeros(num_alts)
    R = np.zeros(num_alts)
    
    for i in range(num_alts):
        diff = f_plus - f_minus
        diff[diff == 0] = 1e-9
        term = weights * ((f_plus - matrix[i, :]) / diff)
        S[i] = np.sum(term)
        R[i] = np.max(term)
        
    s_min, s_max = np.min(S), np.max(S)
    r_min, r_max = np.min(R), np.max(R)
    
    s_diff = (s_max - s_min) if (s_max - s_min) != 0 else 1e-9
    r_diff = (r_max - r_min) if (r_max - r_min) != 0 else 1e-9
    
    Q = v * ((S - s_min) / s_diff) + (1 - v) * ((R - r_min) / r_diff)
    
    res_df = pd.DataFrame({
        'Alternative': work_df['Alternative'],
        'S_Value': S,
        'R_Value': R,
        'Score': Q
    })
    res_df['Rank'] = res_df['Score'].rank(ascending=True, method='min').astype(int)
    
    # 🌟 மிக முக்கியம்: வெளியீட்டு காலமின் பெயரை அசல் பெயருக்கே மாற்றுகிறோம்
    res_df = res_df.rename(columns={'Alternative': alt_col})
    return res_df.sort_values(by='Rank')