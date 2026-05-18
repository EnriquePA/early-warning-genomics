import streamlit as st
import json
import numpy as np
import pandas as pd

# --- ALIAS DICTIONARY ---
MUTATION_ALIASES = {
    "A23403G": "Spike D614G (Ancestral/Global Driver)",
    "C14408T": "ORF1b P314L (Global Basal)",
    "C3037T": "Silent Basal Haplotype",
    "C241T": "5' UTR Basal",
    "G28881A": "Nucleocapsid R203K (Alpha/Omicron)",
    "G28882A": "Nucleocapsid R203K (Alpha/Omicron)",
    "G28883C": "Nucleocapsid G204R (Alpha/Omicron)",
    "T24469A": "Spike T547K (Omicron BA.1)",
    "C22995A": "Spike T478K (Delta/Omicron)",
    "C10029T": "ORF1a T3255I (Omicron)",
    "G23948T": "Spike D1118H (Alpha)",
    "A28271T": "Nucleocapsid D3L (Omicron)",
}

def format_feature_label(mutation):
    if mutation in MUTATION_ALIASES:
        return f"{mutation}  |  {MUTATION_ALIASES[mutation]}"
    return mutation

# 1. App Configuration
st.set_page_config(page_title="Pandemic Early Warning System", page_icon="🧬", layout="wide")

# Initialize Session State for the "One-Click" buttons
if "active_profile" not in st.session_state:
    st.session_state.active_profile = []

def set_profile(profile_mutations):
    st.session_state.active_profile = profile_mutations

# 2. Load the Bayesian Model Weights
@st.cache_data
def load_model():
    with open('early_warning_weights.json', 'r') as f:
        return json.load(f)

try:
    model_data = load_model()
    theta_0 = model_data["intercept"]
    features = model_data["features"]
    weights = np.array(model_data["weights"])
except FileNotFoundError:
    st.error("Error: 'early_warning_weights.json' not found.")
    st.stop()

# 3. The Core Predictive Engine
def predict_pandemic_potential(selected_mutations):
    x_new = np.zeros(len(features))
    for mutation in selected_mutations:
        if mutation in features:
            idx = features.index(mutation)
            x_new[idx] = 1.0
            
    log_odds = theta_0 + np.dot(x_new, weights)
    probability = 1 / (1 + np.exp(-log_odds))
    return probability

# 4. User Interface: Sidebar
st.sidebar.title("🧬 Variant Configuration")

# Feature 2: One-Click Historical Profiles
st.sidebar.markdown("### Quick Load Profiles")
colA, colB = st.sidebar.columns(2)
with colA:
    if st.button("Load Omicron BA.1"):
        # Select Omicron-specific mutations plus the ancestral basals
        set_profile(["T24469A", "C10029T", "C22995A", "G28881A", "G28882A", "G28883C", "A28271T", "A23403G", "C14408T"])
with colB:
    if st.button("Load Basal Only"):
        # Load only the early 2020 basal mutations
        set_profile(["A23403G", "C14408T", "C3037T", "C241T"])
if st.sidebar.button("Clear Dashboard"):
    set_profile([])

st.sidebar.markdown("---")

# Feature Selector (Bound to session state)
selected_mutations = st.sidebar.multiselect(
    "Detected Mutations:",
    options=features,
    default=st.session_state.active_profile,
    format_func=format_feature_label,
    help="Select one or multiple mutations to evaluate epistatic interactions."
)

st.sidebar.markdown("---")
st.sidebar.markdown("**Methodology:** Multivariate Bayesian GLM with L2 regularization.")

# 5. User Interface: Main Body
st.title("SARS-CoV-2 Early Warning System")
st.markdown("This dashboard leverages a Bayesian model trained on $>16$ million global genomic sequences to predict the evolutionary success of novel viral variants.")

# Feature 3: Context Tabs
tab1, tab2 = st.tabs(["🔴 Live Forecast & XAI", "🌍 Regional vs. Global Context"])

with tab1:
    st.markdown("### Variant Threat Forecast")
    if selected_mutations:
        prob = predict_pandemic_potential(selected_mutations)
        
        if prob >= 0.75:
            risk_color, risk_label = "#FF4B4B", "CRITICAL THREAT: High Probability of Displacement Wave"
        elif prob >= 0.50:
            risk_color, risk_label = "#FFA500", "ELEVATED RISK: Active Monitoring Required"
        else:
            risk_color, risk_label = "#00CC96", "LOW RISK: Sequence Lacks Epistatic Drivers"

        st.markdown(f"<h3 style='color: {risk_color};'>{risk_label}</h3>", unsafe_allow_html=True)
        
        # Metric layout
        m_col1, m_col2, m_col3 = st.columns([1, 2, 1])
        with m_col2:
            st.metric(label="Predicted Probability of Dominance", value=f"{prob:.2%}")
            st.progress(float(prob))
            
        st.markdown("---")
        
        # Feature 1: Explainable AI (XAI)
        st.markdown("### Explainable AI: Feature Importance")
        st.markdown("This chart displays the specific Bayesian selection coefficients ($\mathbf{\\theta}$) for the mutations you selected. Positive values drive the pandemic risk upward, while neutral or negative values represent 'passenger' mutations.")
        
        # Extract the specific weights for the selected mutations
        xai_data = []
        for mut in selected_mutations:
            if mut in features:
                idx = features.index(mut)
                xai_data.append({"Mutation": mut, "Impact (Selection Coefficient)": weights[idx]})
        
        df_xai = pd.DataFrame(xai_data)
        if not df_xai.empty:
            df_xai.set_index("Mutation", inplace=True)
            # Render a beautiful native bar chart
            st.bar_chart(df_xai, color="#4A90E2")

    else:
        st.info("👈 Please select mutations from the sidebar or click a 'Quick Load' profile to generate a forecast.")

with tab2:
    st.markdown("### Contextual Evolution: Mexico vs. Global")
    st.markdown("""
    While a specific mutation may cause a sudden localized wave (founder effect), true evolutionary drivers exhibit consistent dominance on a planetary scale. 
    
    * **The Mexico Context:** Initial modeling on ~27,000 regional sequences established the temporal trajectories and identified localized displacement waves.
    * **The Global Context:** By utilizing a streaming MapReduce architecture, this model validated those findings against **>16 million global sequences**, mathematically isolating true epistatic drivers from regional noise.
    
    *Note: In a full production environment, this tab would render the live `matplotlib` trajectory charts mapping the specific mutations selected in the sidebar across time.*
    """)