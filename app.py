import streamlit as st

# 1. அனைத்து பக்கங்களுக்கும் தேவையான "நினைவகப் பெட்டிகளை" (Session State) இங்கே உருவாக்கவும்
if 'data' not in st.session_state:
    st.session_state['data'] = None
if 'weights' not in st.session_state:
    st.session_state['weights'] = None
if 'criteria' not in st.session_state:
    st.session_state['criteria'] = None

# 2. பக்க அமைப்புகள்
st.set_page_config(page_title="Smart Decision AI", page_icon="🚀", layout="wide")

# 3. பக்கங்களின் வரிசைமுறை
pages = {
    "Dashboard": [
        st.Page("home_page.py", title="1. Enterprise Dashboard", icon="📊")
    ],
    "Weight Discovery": [
        st.Page("ahp_page.py", title="2. AHP Weight Matrix", icon="⚖️")
    ],
    "Decision Optimization Engines": [
        st.Page("topsis_page.py", title="3. TOPSIS Rank", icon="🎯"),
        st.Page("vikor_page.py", title="4. VIKOR Compromise", icon="🛡️"),
        st.Page("moora_page.py", title="5. MOORA Optimization", icon="⚡"),
        st.Page("fuzzy_topsis_page.py", title="6. Fuzzy TOPSIS Engine", icon="🧠")
    ],
    "Consensus & Reports": [
        st.Page("overall_consensus_page.py", title="7. Overall Consensus Hub", icon="🏆")
    ]
}

# 4. நேவிகேஷன் பார் உருவாக்குதல்
pg = st.navigation(pages)
pg.run()