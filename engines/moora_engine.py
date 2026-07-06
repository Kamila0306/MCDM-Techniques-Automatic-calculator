import numpy as np
import pandas as pd

def run_moora(df, criteria, weights, impacts):
    alt_col = df.columns[0]
    work_df = df.copy()
    work_df = work_df.rename(columns={alt_col: 'Alternative'})
    
    matrix = work_df[criteria].to_numpy().astype(float)
    weights = np.array(weights)
    
    sq_sum = np.sqrt(np.sum(matrix**2, axis=0))
    sq_sum[sq_sum == 0] = 1e-9
    norm_matrix = matrix / sq_sum
    
    scores = np.zeros(len(work_df))
    for i in range(len(work_df)):
        pos_sum = 0
        neg_sum = 0
        for idx, imp in enumerate(impacts):
            val = norm_matrix[i, idx] * weights[idx]
            if imp == '+':
                pos_sum += val
            else:
                neg_sum += val
        scores[i] = pos_sum - neg_sum
        
    res_df = pd.DataFrame({
        'Alternative': work_df['Alternative'],
        'Score': scores
    })
    res_df['Rank'] = res_df['Score'].rank(ascending=False, method='min').astype(int)
    
    # 🌟 மிக முக்கியம்: வெளியீட்டு காலமின் பெயரை அசல் பெயருக்கே மாற்றுகிறோம்
    res_df = res_df.rename(columns={'Alternative': alt_col})
    return res_df.sort_values(by='Rank')