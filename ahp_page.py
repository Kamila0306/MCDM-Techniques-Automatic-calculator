import streamlit as st
import pandas as pd
import numpy as np
from engines.ahp_engine import calculate_ahp_weights

st.title("⚖️ 1. AHP Weight Discovery & Matrix Hub")
st.markdown("Upload your core dataset, determine or auto-generate criteria weight profiles, and lock them securely into the session database.")
st.markdown("---")

# 1. செஷன் ஸ்டேட்களை முன்கூட்டியே உருவாக்குதல் (Persistence)
if 'saved_dataset' not in st.session_state:
    st.session_state['saved_dataset'] = None
if 'saved_criteria' not in st.session_state:
    st.session_state['saved_criteria'] = []
if 'saved_weights' not in st.session_state:
    st.session_state['saved_weights'] = None
if 'saved_id_col' not in st.session_state:
    st.session_state['saved_id_col'] = None

# 2. டேட்டாசெட் பதிவேற்றம்
uploaded_file = st.file_uploader("Upload Core Dataset (CSV or Excel)", type=["csv", "xlsx"])

# தரவை ஏற்றி சேமித்தல்
if uploaded_file is not None:
    df_input = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file)
    st.session_state['saved_dataset'] = df_input # தரவை பெட்டியில் சேமித்தல்
else:
    df_input = st.session_state['saved_dataset']

if df_input is not None:
    id_col = df_input.columns[0]
    criteria = [c for c in df_input.columns if c != id_col]
    n = len(criteria)
    
    st.success(f"📊 Dataset active: **{len(df_input)} Alternatives** and **{len(criteria)} Criteria** identified.")
    
    # மோட் செலக்சன்
    weight_mode = st.radio(
        "Choose AHP Weight Extraction Strategy:",
        ["🤖 Automatic AI Weight Mode (Statistical Means)", "✍️ Manual Pairwise Comparison Mode"],
        horizontal=True,
        key="ahp_mode_selection"
    )

    weights, cr = None, None

    # =========================================================================
    # 🤖 1. ஆட்டோமேட்டிக் மோட் (AUTOMATIC BACKGROUND SAVE)
    # =========================================================================
    if "Automatic" in weight_mode:
        auto_pairwise_matrix = np.ones((n, n))
        means = df_input[criteria].mean().values
        for i in range(n):
            for j in range(i + 1, n):
                ratio = means[i] / (means[j] if means[j] != 0 else 1e-9)
                val = min(9.0, max(1.0, round(ratio, 1) if ratio >= 1 else round(1.0/ratio, 1)))
                if ratio >= 1:
                    auto_pairwise_matrix[i, j] = val
                    auto_pairwise_matrix[j, i] = 1.0 / val
                else:
                    auto_pairwise_matrix[i, j] = 1.0 / val
                    auto_pairwise_matrix[j, i] = val

        weights, cr, _ = calculate_ahp_weights(auto_pairwise_matrix)
        st.metric(label="Automatically Derived Consistency Ratio (CR)", value=f"{cr:.4f}")

        # 🔥 AUTOMATIC SAVE: பட்டன் இல்லாமலே உடனே பின்னணியில் சேமிக்கிறது!
        st.session_state['saved_weights'] = weights
        st.session_state['saved_criteria'] = criteria
        st.session_state['saved_id_col'] = id_col
        st.session_state['ahp_weights'] = dict(zip(criteria, weights))
        st.caption("🔄 *Global weights profiles are automatically aligned and synchronized with the memory session state.*")

    # =========================================================================
    # ✍️ 2. மேனுவல் மோட் (MANUAL MODE - REQUIRES BUTTON CLICK)
    # =========================================================================
    else:
        st.info("Establish the priority matrix by inputting criteria significance scores:")
        pairwise_matrix = np.ones((n, n))
        with st.expander("🔗 Open Pairwise Assessment Inputs", expanded=True):
            for i in range(n):
                for j in range(i + 1, n):
                    val = st.number_input(
                        f"Importance of [{criteria[i]}] vs [{criteria[j]}] (1-9):", 
                        min_value=0.11, max_value=9.0, value=1.0, step=1.0, 
                        key=f"pwise_{i}_{j}"
                    )
                    pairwise_matrix[i, j] = val
                    pairwise_matrix[j, i] = 1.0 / val
        
        if st.button("Calculate Weights from Matrix", type="secondary"):
            weights, cr, _ = calculate_ahp_weights(pairwise_matrix)
            st.session_state['temp_weights'] = weights
            st.session_state['temp_cr'] = cr

        if 'temp_weights' in st.session_state:
            weights = st.session_state['temp_weights']
            cr = st.session_state['temp_cr']
            st.metric(label="Calculated Consistency Ratio (CR)", value=f"{cr:.4f}")
            if cr >= 0.10:
                st.error("❌ Matrix is inconsistent (CR >= 0.10). Please recalibrate values.")

    # =========================================================================
    # 📋 PREVIEW & LOCK DISPLAY SUB-SYSTEM
    # =========================================================================
    if weights is not None:
        st.markdown("### 📋 Generated Weights Profile Preview")
        weight_preview_df = pd.DataFrame({"Criterion": criteria, "Calculated Weight": [f"{w*100:.2f}%" for w in weights]})
        st.dataframe(weight_preview_df, use_container_width=True)
        
        # Manual Mode-ல் இருந்தால் மட்டும் இந்த Lock பட்டன் வெளியே தெரியும்
        if "Manual" in weight_mode and cr < 0.10:
            if st.button("🔒 Save and Lock Weights for Multi-Model Analysis", type="primary"):
                st.session_state['saved_weights'] = weights
                st.session_state['saved_criteria'] = criteria
                st.session_state['saved_id_col'] = id_col
                st.session_state['ahp_weights'] = dict(zip(criteria, weights))
                st.success("💾 Settings Locked Successfully! You can now navigate to individual method sections in the sidebar.")
        
        # Automatic Mode-ல் இருந்தால் லாக் ஆனதை உடனே கன்ஃபார்ம் செய்யும்
        elif "Automatic" in weight_mode:
            st.success("🎯 **AHP Weights Auto-Locked Successfully into System State Memory!**")

# --- நினைவூட்டல் (Safe Variable Check Profile) ---
if st.session_state.get('saved_weights') is not None:
    st.markdown("---")
    st.info("ℹ️ **Active System Memory Status:** Weights are currently locked. You can navigate freely to TOPSIS, VIKOR, and other optimization engines.")