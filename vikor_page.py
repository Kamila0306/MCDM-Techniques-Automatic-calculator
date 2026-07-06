import streamlit as st
import pandas as pd
import numpy as np

st.title("🧮 VIKOR Multi-Criteria Decision Analysis Module")

# 1. Verification of Uploaded Dataset Status
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

# 2. Dynamic Weights Retrieval Engine (Multi-Format Alignment Protection)
saved_weights = st.session_state.get('saved_weights')
saved_criteria = st.session_state.get('saved_criteria', [])

weights_dict = {}

# Reconstruct a robust dictionary format out of lists or numpy arrays
if saved_weights is not None:
    # If it's a dictionary map directly
    if isinstance(saved_weights, dict):
        weights_dict = saved_weights
    # If it's an array or list, align it directly with saved_criteria or dataset order
    elif len(saved_criteria) == len(saved_weights):
        weights_dict = dict(zip(saved_criteria, saved_weights))
    else:
        # Fallback alignment tracking by dataset column order
        for idx, col in enumerate(numeric_cols):
            if idx < len(saved_weights):
                weights_dict[col] = float(saved_weights[idx])

# Dynamic Validation UI Check
if weights_dict:
    st.success("🎯 **AHP Global Weights Successfully Detected and Auto-Reflected!**")
else:
    st.warning("⚠️ **AHP Weights Profile Not Found.** Utilizing calculated equal fallback allocations.")
    for col in numeric_cols:
        weights_dict[col] = 1.0 / len(numeric_cols)
st.markdown("### 📋 Uploaded Evaluation Matrix Preview")
with st.expander("🔍 Click to view/verify your uploaded Dataset Matrix Values", expanded=False):
    # செஷன் ஸ்டேட்ல இருந்து வர்ற df_input-ஐ அப்படியே பிரிவியூ காட்டுகிறோம்
    preview_df = df_input.copy()
    preview_df.index = preview_df.index + 1
    st.dataframe(preview_df, use_container_width=True)

st.markdown("---") 
        

# Display Active Configured Variables Table & DYNAMIC DIRECTION SETUP
st.markdown("### 📊 Active Analysis Matrix Setup & Direction Configuration")
st.markdown("Configure your objective paths dynamically for each criterion below:")

criteria_types = {}
config_summary = []

# --- HERE USER SELECTS MAX OR MIN INDEPENDENTLY ---
col_left, col_right = st.columns(2)

for i, col in enumerate(numeric_cols):
    # Defaulting "price" or "cost" keywords to Minimum, others to Maximum as guidance
    default_idx = 1 if "cost" in col.lower() or "price" in col.lower() else 0
    
    target_ui_col = col_left if i % 2 == 0 else col_right
    with target_ui_col:
        user_choice = st.selectbox(
            f"Direction for '{col}':",
            ["Maximum is Best (Benefit)", "Minimum is Best (Cost)"],
            index=default_idx,
            key=f"vikor_user_dir_{col}"
        )
    
    criteria_types[col] = "min" if "Minimum" in user_choice else "max"
    
    config_summary.append({
        "Criteria Name": col,
        "Active Assigned Weight": f"{weights_dict.get(col, 1.0/len(numeric_cols)):.4f}",
        "Direction Setup": "Minimum is Best (Cost)" if criteria_types[col] == "min" else "Maximum is Best (Benefit)"
    })

st.markdown("#### Configuration Review Matrix:")
st.table(pd.DataFrame(config_summary))


# --- STEP 1: Best (f*) and Worst (f-) Extraction ---
st.markdown("---")
st.subheader("Step 1: Best ($f^*$) and Worst ($f^-$) Ideal Values")

f_best = {}
f_worst = {}

for col in numeric_cols:
    if criteria_types[col] == "max":
        f_best[col] = float(df_input[col].max())
        f_worst[col] = float(df_input[col].min())
    else:
        f_best[col] = float(df_input[col].min())
        f_worst[col] = float(df_input[col].max())

df_bounds = pd.DataFrame([f_best, f_worst], index=["Best (f*)", "Worst (f-)"])
st.dataframe(df_bounds, use_container_width=True)


# --- STEP 2: Weighted Normalized Regret Matrix ---
st.markdown("---")
st.subheader("Step 2: Weighted Normalized Regret Matrix")
st.markdown("**Formulas Applied:**")
st.latex(r"\text{Benefit: } \frac{f^* - f_{ij}}{f^* - f^-} \times W_j \quad\lvert\quad \text{Cost: } \frac{f_{ij} - f^*}{f^- - f^*} \times W_j")

df_norm = df_input.copy()
for col in numeric_cols:
    w = float(weights_dict.get(col, 1.0 / len(numeric_cols)))
    fb = f_best[col]
    fw = f_worst[col]
    
    if fb == fw:
        df_norm[col] = 0.0
    else:
        if criteria_types[col] == "max":
            df_norm[col] = df_input.apply(lambda row: ((fb - float(row[col])) / (fb - fw)) * w, axis=1)
        else:
            df_norm[col] = df_input.apply(lambda row: ((float(row[col]) - fb) / (fw - fb)) * w, axis=1)

# Display Matrix with Clean Formatting
df_norm_disp = df_input.copy()
for col in numeric_cols:
    df_norm_disp[col] = df_norm[col].map(lambda x: f"{x:.4f}")
st.dataframe(df_norm_disp, use_container_width=True)


# --- STEP 3 & 4: Utility Measure (S) and Regret Measure (R) ---
st.markdown("---")
col_s, col_r = st.columns(2)

df_measures = pd.DataFrame({id_col: df_input[id_col]})
df_measures["S"] = df_norm[numeric_cols].sum(axis=1)
df_measures["R"] = df_norm[numeric_cols].max(axis=1)

with col_s:
    st.subheader("Step 3: Utility Measure ($S$)")
    st.latex(r"S_i = \sum W_j \cdot \text{Norm}_{ij}")
    st.dataframe(df_measures[[id_col, "S"]].style.format({"S": "{:.4f}"}), use_container_width=True)

with col_r:
    st.subheader("Step 4: Regret Measure ($R$)")
    st.latex(r"R_i = \max_j (\text{Norm}_{ij})")
    st.dataframe(df_measures[[id_col, "R"]].style.format({"R": "{:.4f}"}), use_container_width=True)


# --- STEP 5 & 6: Compute Q Index Values ---

st.markdown("---")
st.subheader("Step 5 & 6: Compute $Q$ Values Configuration")

# 📥 INTHA SLIDER CODE-AI INGE PASTE SEIYUNGA
v = st.slider(
    "🔧 Select Weight of Decision-Making Strategy ($v$):", 
    min_value=0.0, max_value=1.0, value=0.5, step=0.05
)
st.markdown(f"#### Active Computation Matrix (with $v = {v}$)")

# Atharku apparam unga matha lines automatic ah work aagum
S_star = float(df_measures["S"].min())
S_minus = float(df_measures["S"].max())
R_star = float(df_measures["R"].min())
R_minus = float(df_measures["R"].max())

st.info(f"💡 **Boundary Anchors:** $S^* = {S_star:.4f}$, $S^- = {S_minus:.4f}$ | $R^* = {R_star:.4f}$, $R^- = {R_minus:.4f}$")

denom_S = (S_minus - S_star) if S_minus != S_star else 1.0
denom_R = (R_minus - R_star) if R_minus != R_star else 1.0

# Intha loop ippo melae user select panna 'v' ai automatic ah use panni calculate aagidum!
df_measures["Q"] = df_measures.apply(
    lambda r: v * ((float(r["S"]) - S_star) / denom_S) + (1 - v) * ((float(r["R"]) - R_star) / denom_R),
    axis=1
)


st.dataframe(df_measures[[id_col, "Q"]].style.format({"Q": "{:.4f}"}), use_container_width=True)


# --- FINAL VIKOR RANKINGS matrix rendering ---
st.markdown("---")
st.subheader("🏆 Final VIKOR Comprehensive Rankings")
st.markdown("> *Note: In VIKOR framework computation, the alternative keeping the **Lowest Q value** is awarded Rank 1.*")

df_final_vikor = df_measures[[id_col, "Q"]].sort_values(by="Q").reset_index(drop=True)
df_final_vikor.index = df_final_vikor.index + 1
df_final_vikor.index.name = "Rank"

# Assign aesthetic medals to top items
def assign_medals(rank):
    if rank == 1: return " 1"
    elif rank == 2: return " 2"
    elif rank == 3: return " 3"
    return str(rank)

df_final_vikor_disp = df_final_vikor.reset_index()
df_final_vikor_disp["Rank"] = df_final_vikor_disp["Rank"].apply(assign_medals)
df_final_vikor_disp = df_final_vikor_disp.set_index("Rank")
df_final_vikor_disp["Q"] = df_final_vikor_disp["Q"].map(lambda x: f"{x:.4f}")

st.dataframe(df_final_vikor_disp, use_container_width=True)

vikor_winner = df_final_vikor.iloc[0][id_col]
st.success(f"🎉 **VIKOR Decision Verdict:** Best compromise solution is **{vikor_winner}** (Lowest $Q$ value = {df_final_vikor.iloc[0]['Q']:.4f})")


# --- 🔒 LOCKING SUBSYSTEM STATE ---
st.markdown("### 💾 Lock VIKOR Computation Model")

if st.button("🔒 Save and Lock VIKOR Rankings", type="primary"):
    st.session_state['vikor_locked_results'] = df_final_vikor.copy()
    st.session_state['vikor_winner'] = vikor_winner
    st.success("💾 VIKOR Multi-Criteria Matrix State has been successfully saved to System Memory.")

if st.session_state.get('vikor_locked_results') is not None:
    st.markdown("---")
    st.info(f"ℹ️ **Active System Memory Status:** VIKOR module data is locked. Optimal Target: **{st.session_state.get('vikor_winner')}**")