import numpy as np

def calculate_ahp_weights(matrix):
    """
    Core AHP Weight Discovery Engine.
    Computes priority weights, Consistency Index (CI), and Consistency Ratio (CR).
    """
    # 1. மேட்ரிக்ஸின் அளவைக் கண்டறிதல் (எண்ணிக்கை)
    n = matrix.shape[0]
    
    # 2. காலம்களின் கூட்டுத்தொகையைக் கணக்கிடுதல் (Column Sums)
    col_sums = np.sum(matrix, axis=0)
    col_sums[col_sums == 0] = 1e-9  # ஜீரோ டிவிஷன் பிழையைத் தவிர்க்க (Zero division guard)
    
    # 3. மேட்ரிக்ஸை நார்மலைஸ் செய்தல் (Normalization)
    norm_matrix = matrix / col_sums
    
    # 4. ஒவ்வொரு வரிசையின் சராசரி மூலம் எடைகளைக் கண்டறிதல் (Criteria Weights)
    weights = np.mean(norm_matrix, axis=1)
    
    # 5. மேட்ரிக்ஸ் மற்றும் எடைகளின் பெருக்கற்பலன் (Weighted Sum Vector)
    weighted_sum = np.dot(matrix, weights)
    
    # 6. லம்ப்டா மேக்ஸ் கணக்கீடு (Principal Eigenvalue - Lambda Max)
    weights_safe = np.where(weights == 0, 1e-9, weights)
    lambda_max = np.mean(weighted_sum / weights_safe)
    
    # 7. கன்சிஸ்டென்சி இண்டெக்ஸ் (Consistency Index - CI)
    ci = (lambda_max - n) / (n - 1) if n > 1 else 0.0
    
    # 8. சாத்தியக்கூறு அட்டவணை (Random Index Values Lookup Table)
    ri_dict = {
        1: 0.0, 2: 0.0, 3: 0.58, 4: 0.90, 5: 1.12, 
        6: 1.24, 7: 1.32, 8: 1.41, 9: 1.45, 10: 1.49
    }
    ri = ri_dict.get(n, 1.45)
    
    # 9. கன்சிஸ்டென்சி விகிதம் (Consistency Ratio - CR)
    cr = (ci / ri) if ri > 0 else 0.0
    
    return weights, cr, ci