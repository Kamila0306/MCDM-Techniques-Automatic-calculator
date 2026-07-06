import numpy as np
import pandas as pd

def run_fuzzy_topsis(df, criteria, fuzzy_weights, impacts):
    alt_col = df.columns[0]
    work_df = df.copy()
    work_df = work_df.rename(columns={alt_col: 'Alternative'})
    
    num_alts = len(work_df)
    
    fuzzy_matrices = {}
    for crit in criteria:
        vals = work_df[crit].astype(float).to_numpy()
        max_v = np.max(vals)
        if max_v == 0:
            max_v = 1e-9
            
        triplets = []
        for v in vals:
            ratio = v / max_v
            l_bound = max(0.0, ratio - 0.15)
            m_peak = ratio
            u_bound = min(1.0, ratio + 0.15)
            triplets.append((l_bound, m_peak, u_bound))
        fuzzy_matrices[crit] = triplets

    fpis = {}
    fnis = {}
    weighted_matrix = {crit: [] for crit in criteria}
    
    for crit in criteria:
        w_l, w_m, w_u = fuzzy_weights[crit]
        imp = impacts[crit]
        
        current_weighted = []
        for (l, m, u) in fuzzy_matrices[crit]:
            current_weighted.append((l * w_l, m * w_m, u * w_u))
            
        weighted_matrix[crit] = current_weighted
        all_u = [t[2] for t in current_weighted]
        all_l = [t[0] for t in current_weighted]
        
        if imp == '+':
            fpis[crit] = (max(all_u), max(all_u), max(all_u))
            fnis[crit] = (min(all_l), min(all_l), min(all_l))
        else:
            fpis[crit] = (min(all_l), min(all_l), min(all_l))
            fnis[crit] = (max(all_u), max(all_u), max(all_u))

    d_plus = np.zeros(num_alts)
    d_minus = np.zeros(num_alts)
    
    for i in range(num_alts):
        for crit in criteria:
            a_l, a_m, a_u = weighted_matrix[crit][i]
            p_l, p_m, p_u = fpis[crit]
            n_l, n_m, n_u = fnis[crit]
            
            dist_p = np.sqrt((1/3.0) * ((a_l - p_l)**2 + (a_m - p_m)**2 + (a_u - p_u)**2))
            dist_n = np.sqrt((1/3.0) * ((a_l - n_l)**2 + (a_m - n_m)**2 + (a_u - n_u)**2))
            
            d_plus[i] += dist_p
            d_minus[i] += dist_n

    denom = d_plus + d_minus
    denom[denom == 0] = 1e-9
    cc_scores = d_minus / denom
    
    res_df = pd.DataFrame({
        'Alternative': work_df['Alternative'],
        'Fuzzy_d_Plus': d_plus,
        'Fuzzy_d_Minus': d_minus,
        'Score': cc_scores
    })
    res_df['Rank'] = res_df['Score'].rank(ascending=False, method='min').astype(int)
    
    
    res_df = res_df.rename(columns={'Alternative': alt_col})
    return res_df.sort_values(by='Rank')