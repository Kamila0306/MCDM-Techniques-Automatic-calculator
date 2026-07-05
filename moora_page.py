import streamlit as st
import pandas as pd
import numpy as np

st.title("🧮 MOORA Multi-Objective Optimization Module")

# 1. Dataset Initialization & Check (Matches image_27c7eb.png)
if st.session_state.get('saved_dataset') is None:
    st.warning("⚠️ **Dataset Required:** Please upload your dataset matrix in the main dashboard first.")
    # Fallback simulation sample to avoid structural app crashes if accessed empty
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
st.subheader("📊 Active Dataset Matrix")
st.dataframe(df_input, use_container_width=True)

# 2. Dynamic AHP Weights Auto-Reflection & Dictionary Safety (Matches image_27c824.png)
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
    st.success("🎯 **AHP Global Weights Successfully Detected and Auto-Reflected!**")
else:
    st.warning("⚠️ **AHP Weights Profile Not Found.** Utilizing equal fallback distribution.")
    for col in numeric_cols:
        weights_dict[col] = 1.0 / len(numeric_cols)

# 🎛️ DYNAMIC CRITERIA DIRECTION SETUP FOR MOORA
st.subheader("⚖️ Reflected AHP Configurations & Direction Setup")
st.markdown("Configure target parameters independently for this MOORA analysis run:")

criteria_types = {}
config_summary = []

# Splitting selection panel into 2 neat columns
col_left, col_right = st.columns(2)

for i, col in enumerate(numeric_cols):
    # Default guidance setup: "price" or "cost" keywords as Min, others as Max
    initial_guess = 1 if "cost" in col.lower() or "price" in col.lower() else 0
    
    target_ui_col = col_left if i % 2 == 0 else col_right
    with target_ui_col:
        user_selection = st.selectbox(
            f"Direction Setup for '{col}':", 
            ["Beneficial (Max)", "Non-beneficial (Min)"], 
            index=initial_guess,
            key=f"moora_dir_select_{col}" # Isolated Unique Key
        )
        
    criteria_types[col] = "min" if "Min" in user_selection else "max"
    
    config_summary.append({
        "Criteria": col,
        "Reflected Weight": f"{weights_dict.get(col, 1.0/len(numeric_cols)):.4f}",
        "Nature": "Non-beneficial (Min)" if criteria_types[col] == "min" else "Beneficial (Max)"
    })

st.markdown("#### Final Configuration Review:")
st.table(pd.DataFrame(config_summary))


# --- STEP 1: Normalize Decision Matrix (Matches image_27c849.png) ---
st.markdown("---")
st.subheader("Step 1: Normalize Decision Matrix")
st.latex(r"x_{ij}^* = \frac{x_{ij}}{\sqrt{\sum_{i=1}^{m} x_{ij}^2}}")

df_norm = df_input.copy()
for col in numeric_cols:
    sq_sum = np.sqrt(np.sum(df_input[col] ** 2))
    if sq_sum == 0:
        df_norm[col] = 0.0
    else:
        df_norm[col] = df_input[col] / sq_sum

# Display formatted step 1
df_norm_disp = df_input.copy()
for col in numeric_cols:
    df_norm_disp[col] = df_norm[col].map(lambda x: f"{x:.4f}")
st.dataframe(df_norm_disp, use_container_width=True)


# --- STEP 2: Weighted Normalized Matrix (Matches image_27c883.png) ---
st.markdown("---")
st.subheader("Step 2: Weighted Normalized Matrix")
st.latex(r"v_{ij} = x_{ij}^* \times W_j")

df_weighted = df_input.copy()
for col in numeric_cols:
    w = float(weights_dict.get(col, 1.0 / len(numeric_cols)))
    df_weighted[col] = df_norm[col] * w

# Display formatted step 2
df_weighted_disp = df_input.copy()
for col in numeric_cols:
    df_weighted_disp[col] = df_weighted[col].map(lambda x: f"{x:.4f}")
st.dataframe(df_weighted_disp, use_container_width=True)


# --- STEP 3 & 4: Benefit Sum and Cost Sum Calculation (Matches image_27c8bc.png & image_27cb67.png) ---
st.markdown("---")
col1, col2 = st.columns(2)

max_cols = [col for col in numeric_cols if criteria_types[col] == "max"]
min_cols = [col for col in numeric_cols if criteria_types[col] == "min"]

df_sums = pd.DataFrame({id_col: df_input[id_col]})
df_sums["Benefit Sum"] = df_weighted[max_cols].sum(axis=1) if max_cols else 0.0
df_sums["Cost Sum"] = df_weighted[min_cols].sum(axis=1) if min_cols else 0.0

with col1:
    st.subheader("Step 3: Calculate Benefit Sum")
    st.caption(f"Sum of: {', '.join(max_cols) if max_cols else 'None'}")
    st.dataframe(df_sums[[id_col, "Benefit Sum"]].style.format({"Benefit Sum": "{:.4f}"}), use_container_width=True)

with col2:
    st.subheader("Step 4: Calculate Cost Sum")
    st.caption(f"Sum of: {', '.join(min_cols) if min_cols else 'None'}")
    st.dataframe(df_sums[[id_col, "Cost Sum"]].style.format({"Cost Sum": "{:.4f}"}), use_container_width=True)


# --- STEP 5: Assessment Value (Yi Score) (Matches image_27cb88.png) ---
st.markdown("---")
st.subheader("Step 5: Assessment Value ($Y_i$)")
st.latex(r"Y_i = \sum_{j=1}^{g} v_{ij} - \sum_{j=g+1}^{n} v_{ij} \quad \Rightarrow \quad Y_i = \text{Benefit Sum} - \text{Cost Sum}")

df_sums["Yi Score"] = df_sums["Benefit Sum"] - df_sums["Cost Sum"]
st.dataframe(df_sums[[id_col, "Yi Score"]].style.format({"Yi Score": "{:.4f}"}), use_container_width=True)


# --- FINAL MOORA RANKINGS (Matches image_27cbac.png layout) ---
st.markdown("---")
st.subheader("🏆 Final MOORA Ranking")
st.markdown("> *Note: In MOORA framework optimization, the alternative with the **Highest $Y_i$ score** is ranked 1st.*")

df_final_moora = df_sums[[id_col, "Yi Score"]].sort_values(by="Yi Score", ascending=False).reset_index(drop=True)
df_final_moora.index = df_final_moora.index + 1
df_final_moora.index.name = "Rank"

# Rank Medals Mapper
def assign_medals(rank):
    if rank == 1: return "🥇 1"
    elif rank == 2: return "🥈 2"
    elif rank == 3: return "🥉 3"
    return str(rank)

df_final_moora_disp = df_final_moora.reset_index()
df_final_moora_disp["Rank"] = df_final_moora_disp["Rank"].apply(assign_medals)
df_final_moora_disp = df_final_moora_disp.set_index("Rank")
df_final_moora_disp["Yi Score"] = df_final_moora_disp["Yi Score"].map(lambda x: f"{x:.4f}")

st.dataframe(df_final_moora_disp, use_container_width=True)

moora_winner = df_final_moora.iloc[0][id_col]
st.success(f"🎉 **MOORA Decision Verdict:** The optimal solution is **{moora_winner}** with a maximum $Y_i$ score of {df_final_moora.iloc[0]['Yi Score']:.4f}")


# --- 🔒 STATE ENGINE LOCKING SYSTEM ---
st.markdown("### 💾 Lock MOORA Computation Model")

if st.button("🔒 Save and Lock MOORA Rankings", type="primary"):
    st.session_state['moora_locked_results'] = df_final_moora.copy()
    st.session_state['moora_winner'] = moora_winner
    st.success("💾 MOORA Multi-Criteria Evaluation has been securely locked in global system state memory.")

if st.session_state.get('moora_locked_results') is not None:
    st.markdown("---")
    st.info(f"ℹ️ **Memory Dashboard Link Active:** MOORA results are tracked. Selected Optimal Target: **{st.session_state.get('moora_winner')}**")