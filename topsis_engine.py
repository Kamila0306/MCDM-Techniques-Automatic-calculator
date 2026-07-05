import numpy as np
import pandas as pd

def run_topsis(df, criteria, weights, impacts):
    alt_col = df.columns[0]
    work_df = df.copy()
    work_df = work_df.rename(columns={alt_col: 'Alternative'})
    
    matrix = work_df[criteria].to_numpy().astype(float)
    weights = np.array(weights)
    
    sq_sum = np.sqrt(np.sum(matrix**2, axis=0))
    sq_sum[sq_sum == 0] = 1e-9
    norm_matrix = matrix / sq_sum
    
    weighted_matrix = norm_matrix * weights
    
    p_ideal = []
    n_ideal = []
    for idx, imp in enumerate(impacts):
        col_data = weighted_matrix[:, idx]
        if imp == '+':
            p_ideal.append(np.max(col_data))
            n_ideal.append(np.min(col_data))
        else:
            p_ideal.append(np.min(col_data))
            n_ideal.append(np.max(col_data))
            
    p_ideal = np.array(p_ideal)
    n_ideal = np.array(n_ideal)
    
    d_plus = np.sqrt(np.sum((weighted_matrix - p_ideal)**2, axis=1))
    d_minus = np.sqrt(np.sum((weighted_matrix - n_ideal)**2, axis=1))
    
    denom = d_plus + d_minus
    denom[denom == 0] = 1e-9
    scores = d_minus / denom
    
    res_df = pd.DataFrame({
        'Alternative': work_df['Alternative'],
        'Score': scores
    })
    res_df['Rank'] = res_df['Score'].rank(ascending=False, method='min').astype(int)
    
    # 🌟 மிக முக்கியம்: வெளியீட்டு காலமின் பெயரை அசல் பெயருக்கே மாற்றுகிறோம்
    res_df = res_df.rename(columns={'Alternative': alt_col})
    return res_df.sort_values(by='Rank')