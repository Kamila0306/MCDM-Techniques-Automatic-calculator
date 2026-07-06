import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="TOPSIS Core Calculation Engine", layout="wide")

st.title("🎯 TOPSIS Core Decision Engine (Step-by-Step Verification)")
st.markdown("This module provides transparent intermediate matrices to perfectly match your verification models.")
st.markdown("---")

# 1. Verification of Uploaded Dataset
if st.session_state.get('saved_dataset') is None:
    st.warning("⚠️ **Dataset Required:** Please upload your dataset matrix in the main dashboard first.")
    st.stop()

df_input = st.session_state['saved_dataset'].copy()
id_col = st.session_state.get('saved_id_col', df_input.columns[0])

# Extract numeric columns for optimization
numeric_cols = [col for col in df_input.columns if col != id_col]

if len(numeric_cols) == 0:
    st.error("❌ No numeric calculation criteria found in the loaded dataset structure.")
    st.stop()

# --- INITIALIZE RAW DATASET ARRAY FOR MATHEMATICAL COMPUTATION ---
X = df_input[numeric_cols].values.astype(float)
# --- 📊 NEW EXTRA FEATURE: DATASET PREVIEW ZONE ---
st.markdown("### 📋 Uploaded Evaluation Matrix Preview")
with st.expander("🔍 Click to view/verify your uploaded Dataset Matrix Values", expanded=False):
    # அப்லோட் செய்யப்பட்ட டேட்டாவை அப்படியே காட்டும் (Index-ஐ 1-லிருந்து மாற்றி)
    preview_df = df_input.copy()
    preview_df.index = preview_df.index + 1
    st.dataframe(preview_df, use_container_width=True)

st.markdown("---") # ஒரு சின்ன செப்பரேட்டர் கோடு
# --- Existing Step 1 Code Starts Here ---
st.subheader("⚖️ Step 1: Criteria Weights & Type Configuration")

# 2. Extract AHP Weights directly from Multi-Format Safe Session Storage
st.subheader("⚖️ Step 1: Criteria Weights & Type Configuration")

weights_dict = {}
criteria_types = {}

# 🛠️ MULTI-FORMAT SAFE RETRIEVAL: Pull directly from criteria list or dict maps
ahp_weights = st.session_state.get('ahp_weights', {})
saved_criteria_list = st.session_state.get('saved_criteria', [])
saved_weights_list = st.session_state.get('saved_weights', [])

# If the dict is empty but the parallel lists exist, stitch them on the fly!
if not ahp_weights and len(saved_criteria_list) == len(saved_weights_list) and len(saved_criteria_list) > 0:
    ahp_weights = dict(zip(saved_criteria_list, saved_weights_list))

col_w1, col_w2 = st.columns(2)

with col_w1:
    st.markdown("### Criteria Direction Setup")
    for col in numeric_cols:
        default_type = "Cost (Minimization)" if "cost" in col.lower() or "price" in col.lower() else "Benefit (Maximization)"
        choice = st.selectbox(
            f"Type for '{col}':", 
            ["Benefit (Maximization)", "Cost (Minimization)"], 
            index=0 if default_type == "Benefit (Maximization)" else 1, 
            key=f"type_{col}"
        )
        criteria_types[col] = "max" if "Benefit" in choice else "min"

with col_w2:
    st.markdown("### Matrix Evaluation Weights")
    
    # Deep Key Lowercase Strip Normalization
    normalized_ahp = {}
    if isinstance(ahp_weights, dict) and len(ahp_weights) > 0:
        for k, v in ahp_weights.items():
            normalized_ahp[str(k).strip().lower()] = v
            
    ahp_keys_list = list(ahp_weights.keys()) if isinstance(ahp_weights, dict) else []

    for idx, col in enumerate(numeric_cols):
        clean_col_name = str(col).strip().lower()
        matched_weight = None
        
        # Priority 1: Direct structural string key matching
        if clean_col_name in normalized_ahp:
            matched_weight = float(normalized_ahp[clean_col_name])
            label_text = f"✅ Weight for '{col}' (Synced from AHP Profile):"
            is_disabled = True
            
        # Priority 2: Safe positional alignment tracking
        elif idx < len(saved_weights_list):
            matched_weight = float(saved_weights_list[idx])
            label_text = f"🧬 Weight for '{col}' (Position Tracked [{idx+1}]):"
            is_disabled = True
            
        # Priority 3: Fallback equal assignment split
        else:
            matched_weight = 1.0 / len(numeric_cols)
            label_text = f"⚠️ Weight for '{col}' (Manual Equal Fallback):"
            is_disabled = False
            
        weight_val = st.number_input(
            label=label_text, 
            min_value=0.0, 
            max_value=1.0, 
            value=float(matched_weight), 
            step=0.0001, 
            format="%.4f", 
            key=f"w_{col}",
            disabled=is_disabled
        )
        weights_dict[col] = weight_val

# Strict Dynamic Normalization Check
total_w_sum = sum(weights_dict.values())
if total_w_sum > 0:
    for col in weights_dict:
        weights_dict[col] /= total_w_sum

# Display Active Weight Summary Table
st.markdown("#### Live Active Weight Summary Configuration")
config_data = []
for col in numeric_cols:
    config_data.append({
        "Criteria": col,
        "Active Weight": f"{weights_dict[col]:.4f}",
        "Optimization Mode": "Minimum is Best (Cost)" if criteria_types[col] == "min" else "Maximum is Best (Benefit)"
    })
st.table(pd.DataFrame(config_data))

# --- CORE TOPSIS MATHEMATICAL PIPELINE ENGINE ---
# Step 1: Vector Normalization Denominators
st.markdown("---")
st.subheader("Step 2: Vector Normalization Denominators")
st.markdown(r"Formula Used: $\text{Denominator}_j = \sqrt{\sum_{i=1}^{m} x_{ij}^2}$")

denominators = np.sqrt(np.sum(X**2, axis=0))
denom_df = pd.DataFrame({
    "Criteria": numeric_cols,
    "Calculated Denominator Value": [f"{v:.4f}" for v in denominators]
})
st.table(denom_df)

# Step 2: Formulate Normalized Matrix
st.subheader("Step 3: Normalized Decision Matrix")
st.markdown(r"Formula Used: $r_{ij} = \frac{x_{ij}}{\sqrt{\sum x_{ij}^2}}$")

R = X / denominators
df_norm = df_input.copy()
df_norm[numeric_cols] = R
st.dataframe(df_norm.style.format({c: "{:.4f}" for c in numeric_cols}), use_container_width=True)

# Step 3: Weighted Normalized Matrix
st.subheader("Step 4: Weighted Normalized Matrix")
st.markdown(r"Formula Used: $v_{ij} = r_{ij} \times w_j$")

W = np.array([weights_dict[col] for col in numeric_cols])
V = R * W
df_weighted = df_input.copy()
df_weighted[numeric_cols] = V
st.dataframe(df_weighted.style.format({c: "{:.4f}" for c in numeric_cols}), use_container_width=True)

# Step 4 & 5: Determine Ideal Bounds
st.subheader("Step 5 & 6: Positive ($A^+$) and Negative ($A^-$) Ideal Solutions")

a_positive = []
a_negative = []

for idx, col in enumerate(numeric_cols):
    col_values = V[:, idx]
    if criteria_types[col] == 'max':
        a_positive.append(np.max(col_values))
        a_negative.append(np.min(col_values))
    else:
        a_positive.append(np.min(col_values))
        a_negative.append(np.max(col_values))

bounds_df = pd.DataFrame({
    "Criteria": numeric_cols,
    "Positive Ideal (A+)": [f"{v:.4f}" for v in a_positive],
    "Negative Ideal (A-)": [f"{v:.4f}" for v in a_negative]
})
st.table(bounds_df)

# Step 6: Separation Measures Processing
st.subheader("Step 7: Separation Measures ($S^+$ and $S^-$)")
st.markdown(r"Formulas: $S_i^+ = \sqrt{\sum (v_{ij} - A_j^+)^2} \quad | \quad $S_i^- = \sqrt{\sum (v_{ij} - A_j^-)^2}$")

s_plus = np.sqrt(np.sum((V - np.array(a_positive))**2, axis=1))
s_minus = np.sqrt(np.sum((V - np.array(a_negative))**2, axis=1))

df_separation = pd.DataFrame({
    id_col: df_input[id_col],
    "S+ (Distance to Ideal Best)": s_plus,
    "S- (Distance to Ideal Worst)": s_minus
})
st.dataframe(df_separation.style.format({
    "S+ (Distance to Ideal Best)": "{:.4f}",
    "S- (Distance to Ideal Worst)": "{:.4f}"
}), use_container_width=True)

# Step 7: Closeness Coefficient Computation
st.subheader("Step 8: Closeness Coefficient ($CC_i$) Evaluation")


closeness_scores = np.where((s_plus + s_minus) == 0, 0.0, s_minus / (s_plus + s_minus))

df_final_scores = pd.DataFrame({
    id_col: df_input[id_col],
    "Closeness Coefficient (CC)": closeness_scores
})

df_final_scores['Rank'] = df_final_scores['Closeness Coefficient (CC)'].rank(ascending=False, method='min').astype(int)
df_final_scores = df_final_scores.sort_values(by='Rank').reset_index(drop=True)

st.session_state['topsis_final_res'] = df_final_scores.copy()

# Render Final Ranking Results (Matches image_1d6ac1.png Layout)
st.markdown("---")
st.subheader(" Final Verified TOPSIS Rankings Matrix Table")

# Format display metrics safely
display_final_df = df_final_scores.copy()

# 1. S.No Column-ah custom-ah insert seigirom
display_final_df.insert(0, 'S.No', range(1, 1 + len(display_final_df)))

display_final_df['Closeness Coefficient (CC)'] = display_final_df['Closeness Coefficient (CC)'].map(lambda x: f"{x:.4f}")

# 2. ⚠️ MUKKIYAM: hide_index=True poduvathaal andha side grey numbers (0,1,2..) oadiyadum!
st.dataframe(display_final_df, use_container_width=True, hide_index=True)

# Celebration/Success note highlighting top item
winner_element = df_final_scores.iloc[0][id_col]
st.success(f"🎉 **Mathematical Verdict:** Alternative **{winner_element}** achieves highest relative mathematical performance proximity ranking!")

# --- 🔒 NEW: TOPSIS RESULT LOCK & SAVE ZONE ---
st.markdown("### 💾 Save and Lock TOPSIS Model State")

# Button to commit the finalized calculations into global memory state
if st.button("🔒 Save and Lock TOPSIS Rankings for Final Comparison", type="primary"):
    st.session_state['topsis_locked_results'] = df_final_scores.copy()
    st.session_state['topsis_winner'] = winner_element
    st.success("💾 TOPSIS Engine Analysis State Locked Successfully! Global comparison charts can now fetch this database matrix.")

# Live Status Indicator Card at the very bottom
if st.session_state.get('topsis_locked_results') is not None:
    st.markdown("---")
    st.info(f"ℹ️ **Active System Memory Status:** TOPSIS model is currently locked. Top Alternative: **{st.session_state.get('topsis_winner')}**.")
