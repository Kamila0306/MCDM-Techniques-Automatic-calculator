import streamlit as st
import pandas as pd
import numpy as np

st.title("📐 Research-Grade Fuzzy TOPSIS Decision Engine")
st.markdown("---")

# =========================================================================
# DATASET INITIALIZATION & FETCHING CONTROL
# =========================================================================
if st.session_state.get('saved_dataset') is None:
    st.warning("⚠️ **Dataset Reference:** No active master matrix found in state cache. Running simulation sandbox fallback context.")
    data = {
        "Supplier": ["Alpha", "Beta", "Gamma", "Delta", "Omega"],
        "Cost": [52, 48, 60, 55, 50],
        "Quality": [82, 78, 90, 88, 84],
        "Delivery": [7, 10, 6, 8, 9],
        "Service": [8.5, 8.0, 9.2, 8.8, 8.3],
        "Capacity": [1500, 1700, 1400, 1600, 1800],
        "Sustainability": [70, 80, 85, 75, 90]
    }
    df_input = pd.DataFrame(data)
    id_col = "Supplier"
else:
    df_input = st.session_state['saved_dataset'].copy()
    id_col = st.session_state.get('saved_id_col', df_input.columns[0])

numeric_cols = [col for col in df_input.columns if col != id_col]

# Extract weights from prior AHP state layer if available
saved_weights = st.session_state.get('saved_weights')
saved_criteria = st.session_state.get('saved_criteria', [])
weights_dict = {}

if saved_weights is not None:
    if isinstance(saved_weights, dict):
        weights_dict = saved_weights
    elif len(saved_criteria) == len(saved_weights):
        weights_dict = dict(zip(saved_criteria, saved_weights))
    else:
        for idx, col in enumerate(numeric_cols):
            if idx < len(saved_weights):
                weights_dict[col] = float(saved_weights[idx])
else:
    for col in numeric_cols:
        weights_dict[col] = 1.0 / len(numeric_cols)

# =========================================================================
# DYNAMIC USER INTERFACE DIRECTION SELECTION (MAX / MIN CONTROL)
# =========================================================================
st.subheader("🔄 Configure Criteria Target Directions")
st.markdown("Select whether each criterion should be Maximized (Benefit) or Minimized (Cost) for this custom dataset:")

# Fetch any previously configured dashboard-wide directions if available
saved_directions = st.session_state.get('saved_directions', {})

criteria_types = {}
# Creating a dynamic row layout for column configuration inputs
input_cols = st.columns(len(numeric_cols))

for idx, col in enumerate(numeric_cols):
    with input_cols[idx]:
        # Identify default state based on column name or prior state layers
        if col in saved_directions:
            default_index = 0 if "max" in str(saved_directions[col]).lower() else 1
        else:
            # Fallback smart automated rule
            col_lower = col.lower()
            default_index = 1 if ("cost" in col_lower or "price" in col_lower or "delivery" in col_lower or "delay" in col_lower) else 0
            
        user_choice = st.selectbox(
            f"Direction for '{col}':",
            options=["Beneficial (Max)", "Non-beneficial (Min)"],
            index=default_index,
            key=f"fuzzy_dir_{col}"
        )
        # Mapping selected choice string back into lowercase 'max' / 'min' for the mathematical backend engine
        criteria_types[col] = "max" if "max" in user_choice.lower() else "min"

st.markdown("---")

# =========================================================================
# RESEARCH METHODOLOGY SCALE SELECTION INTERFACE
# =========================================================================
st.subheader("⚙️ TFN Weight Derivation Strategy")
approach_mode = st.radio(
    "Select TFN Mapping Standard For AHP Weights Integration:",
    ["Method 1: Linguistic Scale Mapping (Standard Qualitative Grouping Framework)", 
     "Method 2: Direct Expert Uncertainty Bounds (IEEE/Elsevier Precision Format)"],
    index=1
)
st.markdown("---")

# =========================================================================
# STEP 1: AHP WEIGHTS -> FUZZY WEIGHTS TFN GENERATION
# =========================================================================
st.subheader("📋 Step 1: AHP Weights ➔ Fuzzy Weights (TFN Representation)")

fuzzy_weights = {}
tfn_summary = []

if "Method 1" in approach_mode:
    st.markdown("**Methodology Applied:** Linguistic TFN Mapping Strategy.")
    for col in numeric_cols:
        w = float(weights_dict.get(col, 1.0 / len(numeric_cols)))
        if w <= 0.10:
            ling, tfn = "Very Low (VL)", (0.00, 0.00, 0.25)
        elif w <= 0.30:
            ling, tfn = "Low (L)", (0.00, 0.25, 0.50)
        elif w <= 0.50:
            ling, tfn = "Medium (M)", (0.25, 0.50, 0.75)
        elif w <= 0.70:
            ling, tfn = "High (H)", (0.50, 0.75, 1.00)
        else:
            ling, tfn = "Very High (VH)", (0.75, 1.00, 1.00)
            
        fuzzy_weights[col] = tfn
        tfn_summary.append({
            "Criteria/Dimension": col, "AHP Weight (Modal)": f"{w:.4f}", "Linguistic Assignment": ling, "Resultant TFN (l, m, u)": f"({tfn[0]:.2f}, {tfn[1]:.2f}, {tfn[2]:.2f})"
        })
    st.table(pd.DataFrame(tfn_summary))

else:
    st.markdown("> **Academic Methodology Statement:** *The crisp AHP weights were transformed into triangular fuzzy weights by assigning lower, modal, and upper bounds based on expert uncertainty around each criterion weight.*")
    uncertainty_delta = st.slider("🔧 Calibrate Expert Uncertainty Spread Radius ($\pm \Delta$):", min_value=0.01, max_value=0.10, value=0.04, step=0.01)
    
    for col in numeric_cols:
        w = float(weights_dict.get(col, 1.0 / len(numeric_cols)))
        l = max(0.001, w - uncertainty_delta)
        m = w
        u = min(1.000, w + uncertainty_delta)
        
        fuzzy_weights[col] = (l, m, u)
        tfn_summary.append({
            "Criteria/Dimension": col, "AHP Weight Matrix (m)": f"{w:.4f}", "Lower Bound (l)": f"{l:.3f}", "Modal Parameter (m)": f"{m:.3f}", "Upper Bound (u)": f"{u:.3f}"
        })
    st.table(pd.DataFrame(tfn_summary))

st.markdown("---")

# =========================================================================
# STEP 2: VECTOR NORMALIZATION MATRIX GENERATION
# =========================================================================
st.subheader("📊 Step 2: Vector Normalized Decision Matrix Estimation")
st.latex(r"n_{ij} = \frac{x_{ij}}{\sqrt{\sum_{k=1}^{m} x_{kj}^2}}")

df_norm = df_input.copy()
for col in numeric_cols:
    sq_sum = np.sqrt(np.sum(df_input[col] ** 2))
    df_norm[col] = df_input[col] / sq_sum if sq_sum != 0 else 0.0

df_norm_disp = df_input.copy()
for col in numeric_cols:
    df_norm_disp[col] = df_norm[col].map(lambda x: f"{x:.4f}")
st.dataframe(df_norm_disp, use_container_width=True)
st.markdown("---")

# =========================================================================
# STEP 3: WEIGHTED FUZZY MATRIX CALCULATIONS (Ref: image_292942.png)
# =========================================================================
st.subheader("🧱 Step 3: Weighted Fuzzy Matrix Mapping Calculations")
st.markdown("Multiplication of every profile normalized crisp value against its assigned criteria TFN elements:")
st.latex(r"v_{ij} = n_{ij} \otimes w_j = (n_{ij} \times l_j, \; n_{ij} \times m_j, \; n_{ij} \times u_j)")

weighted_fuzzy_matrix = {}
for col in numeric_cols:
    l_w, m_w, u_w = fuzzy_weights[col]
    weighted_fuzzy_matrix[col] = [(val * l_w, val * m_w, val * u_w) for val in df_norm[col]]

# Render Complete Analytical Tabular Matrix View for Step 3
step3_records = []
for idx in range(len(df_input)):
    row_dict = {id_col: df_input.iloc[idx][id_col]}
    for col in numeric_cols:
        tfn_val = weighted_fuzzy_matrix[col][idx]
        row_dict[col] = f"({tfn_val[0]:.3f}, {tfn_val[1]:.3f}, {tfn_val[2]:.3f})"
    step3_records.append(row_dict)

df_step3_final = pd.DataFrame(step3_records)
st.dataframe(df_step3_final, use_container_width=True)

# Step 3 Dynamic Verification Callout Box
sample_col = numeric_cols[0]
sample_profile = df_input.iloc[0][id_col]
st.info(f"💡 **Step 3 Analytical Sample Check:** — **{sample_profile} ({sample_col}):** "
        f"{df_norm.iloc[0][sample_col]:.3f} × ({fuzzy_weights[sample_col][0]:.3f}, {fuzzy_weights[sample_col][1]:.3f}, {fuzzy_weights[sample_col][2]:.3f}) = "
        f"({weighted_fuzzy_matrix[sample_col][0][0]:.3f}, {weighted_fuzzy_matrix[sample_col][0][1]:.3f}, {weighted_fuzzy_matrix[sample_col][0][2]:.3f})")
st.markdown("---")

# =========================================================================
# STEP 4 & 5: FPIS & FNIS IDEAL BOUNDARY IDENTIFICATION (Ref: image_292962.png)
# =========================================================================
st.subheader("🎯 Step 4 & Step 5: Fuzzy Ideal Solutions Allocation Boundaries")

fpis_dict = {}
fnis_dict = {}

for col in numeric_cols:
    tfn_array = np.array(weighted_fuzzy_matrix[col])
    if criteria_types[col] == "max":
        # Benefit criteria orientation logic
        fpis_dict[col] = (tfn_array[:, 0].max(), tfn_array[:, 1].max(), tfn_array[:, 2].max())
        fnis_dict[col] = (tfn_array[:, 0].min(), tfn_array[:, 1].min(), tfn_array[:, 2].min())
    else:
        # Cost criteria orientation logic (Inverted allocation rules)
        fpis_dict[col] = (tfn_array[:, 0].min(), tfn_array[:, 1].min(), tfn_array[:, 2].min())
        fnis_dict[col] = (tfn_array[:, 0].max(), tfn_array[:, 1].max(), tfn_array[:, 2].max())

col_view_4, col_view_5 = st.columns(2)

with col_view_4:
    st.markdown("### Step 4: FPIS ($A^+$) Vector Table")
    fpis_summary = [{"Criteria Element": k, "Target Max/Min Type": criteria_types[k].upper(), "FPIS TFN (l*, m*, u*)": f"({v[0]:.3f}, {v[1]:.3f}, {v[2]:.3f})"} for k, v in fpis_dict.items()]
    st.table(pd.DataFrame(fpis_summary))

with col_view_5:
    st.markdown("### Step 5: FNIS ($A^-$) Vector Table")
    fnis_summary = [{"Criteria Element": k, "Target Max/Min Type": criteria_types[k].upper(), "FNIS TFN (l-, m-, u-)": f"({v[0]:.3f}, {v[1]:.3f}, {v[2]:.3f})"} for k, v in fnis_dict.items()]
    st.table(pd.DataFrame(fnis_summary))

st.markdown("---")

# =========================================================================
# STEP 6: VERTEX EUCLIDEAN DISTANCE EVALUATION (Ref: image_292983.png)
# =========================================================================
st.subheader("📐 Step 6: Vertex Euclidean Distance Metrics Summarization")
st.latex(r"d(A, B) = \sqrt{\frac{(l_1 - l_2)^2 + (m_1 - m_2)^2 + (u_1 - u_2)^2}{3}}")

d_plus_list = []
d_minus_list = []
distance_breakdown_records = []

for i in range(len(df_input)):
    d_plus_sum = 0.0
    d_minus_sum = 0.0
    profile_name = df_input.iloc[i][id_col]
    
    for col in numeric_cols:
        val_tfn = weighted_fuzzy_matrix[col][i]
        fpis_tfn = fpis_dict[col]
        fnis_tfn = fnis_dict[col]
        
        # Apply strict Euclidean Vertex formulas
        dist_p = np.sqrt(((val_tfn[0] - fpis_tfn[0])**2 + (val_tfn[1] - fpis_tfn[1])**2 + (val_tfn[2] - fpis_tfn[2])**2) / 3.0)
        dist_m = np.sqrt(((val_tfn[0] - fnis_tfn[0])**2 + (val_tfn[1] - fnis_tfn[1])**2 + (val_tfn[2] - fnis_tfn[2])**2) / 3.0)
        
        d_plus_sum += dist_p
        d_minus_sum += dist_m

    d_plus_list.append(d_plus_sum)
    d_minus_list.append(d_minus_sum)
    
    distance_breakdown_records.append({
        id_col: profile_name,
        "Total Distance to FPIS ($D^+$)": f"{d_plus_sum:.4f}",
        "Total Distance to FNIS ($D^-$)": f"{d_minus_sum:.4f}"
    })

st.table(pd.DataFrame(distance_breakdown_records))
st.markdown("---")

# =========================================================================
# STEP 7: CLOSENESS COEFFICIENT ($CC_i$) PERFORMANCE ESTIMATION (Ref: image_292983.png)
# =========================================================================
st.subheader("📈 Step 7: Closeness Coefficient ($CC_i$) Core Formulations")
st.latex(r"CC_i = \frac{D_i^-}{D_i^- + D_i^+}")

df_results = pd.DataFrame({id_col: df_input[id_col]})
df_results["D+"] = d_plus_list
df_results["D-"] = d_minus_list
df_results["CC"] = df_results["D-"] / (df_results["D-"] + df_results["D+"])

df_results_disp = df_results.copy()
st.dataframe(df_results_disp.style.format({"D+": "{:.4f}", "D-": "{:.4f}", "CC": "{:.4f}"}), use_container_width=True)
st.markdown("---")

# =========================================================================
# FINAL OUTPUT OPTIMAL RANKING GRID
# =========================================================================
st.subheader("🏆 Final Standardized Fuzzy TOPSIS Ranking Grid")

df_rankings = df_results[[id_col, "CC"]].sort_values(by="CC", ascending=False).reset_index(drop=True)
df_rankings.index = df_rankings.index + 1
df_rankings.index.name = "Rank Position"

df_rankings_disp = df_rankings.reset_index()
df_rankings_disp["Rank Position"] = df_rankings_disp["Rank Position"].apply(lambda r: f"🥇 1" if r==1 else (f"🥈 2" if r==2 else (f"🥉 3" if r==3 else str(r))))
df_rankings_disp = df_rankings_disp.set_index("Rank Position")
df_rankings_disp["CC Performance Score"] = df_rankings_disp["CC"].map(lambda x: f"{x:.4f}")
df_rankings_disp = df_rankings_disp.drop(columns=["CC"])

st.dataframe(df_rankings_disp, use_container_width=True)

fuzzy_winner = df_rankings.iloc[0][id_col]
st.success(f"🎉 **Mathematical System Verdict:** Optimal choice profile candidate identified as **{fuzzy_winner}** with a Closeness score index of **{df_rankings.iloc[0]['CC']:.4f}**")
st.markdown("---")

# =========================================================================
# 🔒 SYSTEM DATA MEMORY STATE ENGINE LOCKING MECHANISM
# =========================================================================
st.subheader("💾 Lock Operational Decision State Models")

if st.button("🔒 Execute Permanent Cache State Lock", type="primary"):
    st.session_state['fuzzy_topsis_locked_results'] = df_rankings.copy()
    st.session_state['fuzzy_topsis_winner'] = fuzzy_winner
    st.session_state['selected_fuzzy_method'] = approach_mode
    st.success("💾 Mathematical pipeline matrix structures safely synchronized into system data cache modules.")

if st.session_state.get('fuzzy_topsis_locked_results') is not None:
    st.info(f"ℹ️ **System Active Memory Status:** Running Target Core Selection: **{st.session_state.get('fuzzy_topsis_winner')}** (Methodology Setting Locked: *{st.session_state.get('selected_fuzzy_method')}*)")