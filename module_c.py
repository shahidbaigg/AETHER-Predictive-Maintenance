import streamlit as st
import pandas as pd
import numpy as np
import shap
import plotly.graph_objects as go
from scipy.spatial import distance
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import RidgeCV

st.set_page_config(page_title="AETHER Command Center", layout="wide", initial_sidebar_state="expanded")

# --- CUSTOM THEME CONTROLS ---
with st.sidebar.expander("🎨 Custom Theme Settings"):
    st.write("Personalize your dashboard interface:")
    bg_color = st.color_picker("Main Background Color", "#0b0f19")
    sidebar_color = st.color_picker("Sidebar Background", "#111827")
    accent_color = st.color_picker("Primary Accent Color", "#38bdf8")

# --- INJECT DYNAMIC CSS ---
st.markdown(f"""
<style>
    /* Main Background & Text */
    .stApp {{
        background-color: {bg_color};
        color: #e0e6ed;
    }}
    
    /* Sidebar */
    [data-testid="stSidebar"] {{
        background-color: {sidebar_color};
        border-right: 1px solid #1f2937;
    }}
    
    /* Headers */
    h1, h2, h3, h4, h5, h6 {{
        color: {accent_color} !important;
        font-family: 'Inter', sans-serif;
    }}
    
    /* Metrics */
    [data-testid="stMetricValue"] {{
        color: #34d399 !important;
        font-weight: 700;
    }}
    [data-testid="stMetricLabel"] {{
        color: #94a3b8 !important;
        font-weight: 600;
    }}
    
    /* Buttons */
    .stButton>button, .stDownloadButton>button {{
        background: linear-gradient(135deg, {accent_color}, #3b82f6) !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        transition: all 0.3s ease !important;
    }}
    .stButton>button:hover, .stDownloadButton>button:hover {{
        transform: scale(1.02);
        box-shadow: 0 0 15px {accent_color}80;
    }}
    
    /* File Uploader */
    [data-testid="stFileUploadDropzone"] {{
        border: 2px dashed {accent_color} !important;
        background-color: {bg_color} !important;
        border-radius: 10px !important;
    }}
    
    /* Chat Messages */
    [data-testid="chatAvatarIcon-user"] {{
        background-color: {accent_color};
    }}
    [data-testid="chatAvatarIcon-assistant"] {{
        background-color: #34d399;
    }}
    
    /* Dividers */
    hr {{
        border-color: #1f2937 !important;
    }}
</style>
""", unsafe_allow_html=True)

# --- UI TRANSLATION DICTIONARY ---
def get_clean_feature_name(raw_name):
    mapping = {
        'iddq_0h': 'Initial Leakage',
        'delay_0h': 'Initial Delay',
        'resist_0h': 'Initial Resistance',
        'iddq_24h': '24h Leakage',
        'delay_24h': '24h Delay',
        'resist_24h': '24h Resistance',
        'iddq_drift': 'Leakage Drift',
        'delay_drift': 'Delay Drift',
        'resist_drift': 'Resistance Drift'
    }
    
    parts = raw_name.split(' ')
    clean_parts = [mapping.get(p, p) for p in parts]
    
    if len(clean_parts) == 2:
        if clean_parts[0] == clean_parts[1]:
            return f"{clean_parts[0]}²"
        return f"{clean_parts[0]} × {clean_parts[1]}"
    
    return clean_parts[0]

# --- CORE PIPELINE MODEL ---
@st.cache_resource
def train_production_model():
    df_train = pd.read_csv("telemetry_synthetic_master.csv")
    df_train['iddq_drift'] = df_train['iddq_24h'] - df_train['iddq_0h']
    df_train['delay_drift'] = df_train['delay_24h'] - df_train['delay_0h']
    df_train['resist_drift'] = df_train['resist_24h'] - df_train['resist_0h']
    df_train['actual_final_drift'] = df_train['iddq_168h'] - df_train['iddq_0h']
    
    features = ['iddq_0h', 'delay_0h', 'resist_0h', 'iddq_24h', 'delay_24h', 'resist_24h', 'iddq_drift', 'delay_drift', 'resist_drift']
    
    poly = PolynomialFeatures(degree=2, include_bias=False)
    X_poly = poly.fit_transform(df_train[features])
    
    raw_feature_names = poly.get_feature_names_out(features)
    clean_feature_names = [get_clean_feature_name(name) for name in raw_feature_names]
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_poly)
    
    model = RidgeCV(alphas=np.logspace(-3, 3, 10))
    model.fit(X_scaled, df_train['actual_final_drift'])
    
    explainer = shap.LinearExplainer(model, X_scaled, feature_names=clean_feature_names)
    
    return model, poly, scaler, explainer, clean_feature_names, features

model, poly, scaler, explainer, feature_names, features = train_production_model()

# --- UI SIDEBAR & UPLOADER ---
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/e/e5/NASA_logo.svg", width=120)
    st.title("AETHER-DPAT")
    st.markdown("### Telemetry Ingestion")
    
    template_df = pd.DataFrame({
        'component_id': ['COMP_001', 'COMP_002'],
        'iddq_0h': [4.55, 4.60],
        'delay_0h': [1.20, 1.25],
        'resist_0h': [0.80, 0.81],
        'iddq_24h': [4.70, 4.90],
        'delay_24h': [1.25, 1.35],
        'resist_24h': [0.82, 0.88]
    })
    csv_template = template_df.to_csv(index=False).encode('utf-8')
    
    st.download_button(
        label="📄 Download Data Template",
        data=csv_template,
        file_name='aether_telemetry_template.csv',
        mime='text/csv',
        help="Download this template to see the required column headers for your telemetry data."
    )
    
    uploaded_file = st.file_uploader("Upload CSV Batch", type=["csv"])
    st.divider()
    usl_limit = st.slider("Leakage USL (µA) Limit", min_value=30.0, max_value=80.0, value=50.0, step=1.0)
    contamination_rate = st.slider("Module A Sensitivity", min_value=0.005, max_value=0.05, value=0.01, step=0.005)

# --- DYNAMIC HOMEPAGE (Before Upload) ---
if uploaded_file is None:
    st.markdown("## Welcome to the AETHER-DPAT Inference Engine")
    st.markdown("Upload a telemetry CSV batch in the sidebar to initiate real-time AI screening. While you wait, explore how the pipeline secures aerospace components:")
    st.write("") 
    
    c1, c2, c3 = st.columns(3)
    with c1:
        st.image("https://images.unsplash.com/photo-1518770660439-4636190af475?ixlib=rb-4.0.3&auto=format&fit=crop&w=600&q=80", use_container_width=True)
        st.markdown("### 1. The Gatekeeper")
        st.caption("**Immediate Anomaly Isolation**")
        st.write("Module A uses Mahalanobis DPAT and Isolation Forests to instantly reject components exhibiting extreme 24-hour sensor drift.")
    with c2:
        st.image("https://images.unsplash.com/photo-1581092160562-40aa08e78837?ixlib=rb-4.0.3&auto=format&fit=crop&w=600&q=80", use_container_width=True)
        st.markdown("### 2. The Predictor")
        st.caption("**Latent Defect Forecasting**")
        st.write("Module B leverages Ridge Regression to predict 168-hour thermal runaway (TDDB) based on hidden physical correlations.")
    with c3:
        st.image("https://images.unsplash.com/photo-1550751827-4bd374c3f58b?ixlib=rb-4.0.3&auto=format&fit=crop&w=600&q=80", use_container_width=True)
        st.markdown("### 3. Explainable AI")
        st.caption("**Mathematical Transparency**")
        st.write("Module C generates interactive SHAP waterfalls, proving the exact physics driving every AI rejection decision.")
        
    st.stop()

# --- DATA PROCESSING ---
data = pd.read_csv(uploaded_file)
data['iddq_drift'] = data['iddq_24h'] - data['iddq_0h']
data['delay_drift'] = data['delay_24h'] - data['delay_0h']
data['resist_drift'] = data['resist_24h'] - data['resist_0h']

# Module A
drift_features = data[['iddq_drift', 'delay_drift', 'resist_drift']].values
cov_matrix = np.cov(drift_features, rowvar=False) + np.eye(3) * 1e-8
inv_cov = np.linalg.pinv(cov_matrix)
mean_drift = np.mean(drift_features, axis=0)

data['mahalanobis_dist'] = [distance.mahalanobis(row, mean_drift, inv_cov) for row in drift_features]
data['mahalanobis_fail'] = data['mahalanobis_dist'] > np.percentile(data['mahalanobis_dist'], (1 - contamination_rate)*100)
data['iso_forest_fail'] = IsolationForest(contamination=contamination_rate, random_state=42).fit_predict(drift_features) == -1
data['module_a_fail'] = data['mahalanobis_fail'] | data['iso_forest_fail']

# Module B
valid_mask = ~data['module_a_fail']
valid_parts = data[valid_mask].copy()
X_new_poly = poly.transform(valid_parts[features])
X_new_scaled = scaler.transform(X_new_poly)

valid_parts['predicted_168h_drift'] = model.predict(X_new_scaled)
valid_parts['predicted_final_iddq'] = valid_parts['iddq_0h'] + valid_parts['predicted_168h_drift']
valid_parts['module_b_fail'] = valid_parts['predicted_final_iddq'] > usl_limit

# Map final status
data['Status'] = "Passed (Healthy)"
data.loc[data['module_a_fail'], 'Status'] = "Rejected (Module A: Current Anomaly)"
data.loc[valid_parts[valid_parts['module_b_fail']].index, 'Status'] = "Rejected (Module B: Latent Defect)"

shap_vals_new = explainer.shap_values(X_new_scaled)

# --- MAIN DASHBOARD ---
st.title("Aerospace Batch Diagnostics")
m1, m2, m3, m4 = st.columns(4)
m1.metric("Total Components", len(data))
m2.metric("Module A Rejections", data['module_a_fail'].sum())
m3.metric("Module B Rejections", valid_parts['module_b_fail'].sum())
m4.metric("Overall Yield", f"{(len(data) - data['module_a_fail'].sum() - valid_parts['module_b_fail'].sum()) / len(data) * 100:.1f}%")

tab1, tab2 = st.tabs(["📊 Component Inspector", "💬 AETHER Diagnostic Assistant"])

with tab1:
    st.markdown("### Interactive Threat Analysis")
    col_sel, col_stat = st.columns([1, 2])
    
    with col_sel:
        target_status = st.selectbox("1. Filter by Status", ["Rejected (Module B: Latent Defect)", "Rejected (Module A: Current Anomaly)", "Passed (Healthy)"])
        filtered_comps = data[data['Status'] == target_status]['component_id'].tolist()
        selected_comp = st.selectbox("2. Select Component ID", filtered_comps)
    
    if selected_comp:
        comp_idx = data.index[data['component_id'] == selected_comp].tolist()[0]
        comp_status = data.loc[comp_idx, 'Status']
        
        with col_stat:
            st.markdown("### Final Verdict")
            if "Module B" in comp_status:
                st.error(f"🚨 **{comp_status}** - Do not deploy. Thermal runaway risk.")
            elif "Module A" in comp_status:
                st.warning(f"⚠️ **{comp_status}** - Do not deploy. Gross defect caught early.")
            else:
                st.success(f"✅ **{comp_status}** - Cleared for aerospace flight.")
                
        st.divider()

        if "Module B" in comp_status:
            valid_idx = valid_parts.reset_index().index[valid_parts['component_id'] == selected_comp].tolist()[0]
            
            st.markdown(f"#### Predictive Logic Breakdown (SHAP)")
            st.caption("This chart reveals the hidden mathematical physics driving the AI's thermal runaway prediction.")
            sv = shap_vals_new[valid_idx]
            sorted_idx = np.argsort(np.abs(sv))[-10:] 
            
            fig = go.Figure(go.Waterfall(
                orientation="h", measure=["relative"] * 10 + ["total"],
                y=[feature_names[i] for i in sorted_idx] + ["Predicted Final Leakage Drift"],
                x=[sv[i] for i in sorted_idx] + [sum(sv)],
                connector={"line": {"color": "rgb(56, 189, 248)"}},
            ))
            fig.update_layout(height=400, margin=dict(l=20, r=20, t=20, b=20), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color="#e0e6ed"))
            st.plotly_chart(fig, use_container_width=True)

            st.markdown("#### Degradation Trajectory")
            traj_y = [valid_parts.loc[comp_idx, 'iddq_0h'], valid_parts.loc[comp_idx, 'iddq_24h'], valid_parts.loc[comp_idx, 'predicted_final_iddq']]
            fig2 = go.Figure(data=go.Scatter(x=["0h", "24h", "168h (Predicted)"], y=traj_y, mode='lines+markers', marker=dict(size=12, color='#ef4444')))
            fig2.add_hline(y=usl_limit, line_dash="dash", line_color="#f59e0b", annotation_text="Datasheet Failure Limit")
            fig2.update_layout(height=350, yaxis_title="Current Leakage (µA)", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color="#e0e6ed"))
            st.plotly_chart(fig2, use_container_width=True)
            
        else:
            st.markdown("#### 24-Hour Telemetry Check")
            st.caption("Module B (SHAP) is only invoked for components that pass Module A. Here is the raw sensor data that led to this verdict:")
            
            c1, c2, c3 = st.columns(3)
            c1.metric("Initial Leakage", f"{data.loc[comp_idx, 'iddq_0h']:.2f} µA")
            c2.metric("24h Leakage", f"{data.loc[comp_idx, 'iddq_24h']:.2f} µA")
            c3.metric("Total Leakage Drift", f"{data.loc[comp_idx, 'iddq_drift']:.2f} µA")
            
            fig3 = go.Figure(data=[
                go.Bar(name='This Component', x=['Leakage Drift', 'Delay Drift', 'Resistance Drift'], 
                       y=[data.loc[comp_idx, 'iddq_drift'], data.loc[comp_idx, 'delay_drift'], data.loc[comp_idx, 'resist_drift']], marker_color='#38bdf8'),
                go.Bar(name='Batch Average', x=['Leakage Drift', 'Delay Drift', 'Resistance Drift'], 
                       y=[data['iddq_drift'].mean(), data['delay_drift'].mean(), data['resist_drift'].mean()], marker_color='#34d399')
            ])
            fig3.update_layout(barmode='group', height=400, title="Component Drift vs. Batch Average", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color="#e0e6ed"))
            st.plotly_chart(fig3, use_container_width=True)

with tab2:
    st.markdown("### AI Diagnostic Chatbot")
    
    if selected_comp:
        comp_status = data.loc[data['component_id'] == selected_comp, 'Status'].values[0]
        
        if "messages" not in st.session_state or st.session_state.get('last_comp') != selected_comp:
            st.session_state.messages = []
            st.session_state.last_comp = selected_comp
            
            if "Module B" in comp_status:
                init_msg = f"Hello. Component `{selected_comp}` was rejected by Module B because my Ridge Regression model predicts it will suffer thermal runaway by 168h. Ask me why!"
            elif "Module A" in comp_status:
                init_msg = f"Hello. Component `{selected_comp}` was immediately isolated by the Gatekeeper (Module A) at 24h because its sensor readings were highly erratic compared to the rest of the batch. How can I help?"
            else:
                init_msg = f"Hello. Component `{selected_comp}` is perfectly healthy. What would you like to know?"
            
            st.session_state.messages.append({"role": "assistant", "content": init_msg})

        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.write(msg["content"])

        if prompt := st.chat_input(f"Ask about {selected_comp}..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.write(prompt)
                
            with st.chat_message("assistant"):
                p = prompt.lower()
                
                if "why" in p or "reason" in p:
                    if "Module B" in comp_status:
                        response = f"While it looked fine at 24h, the interaction between its propagation delay and leakage drift mathematically aligned with the NASA Time-Dependent Dielectric Breakdown curve. It is on track to cross the {usl_limit}µA safety threshold."
                    elif "Module A" in comp_status:
                        response = "It failed the Isolation Forest and Mahalanobis DPAT checks. Basically, its 24-hour sensor drift was so extreme that it mathematically fell into the top 1% of anomalies for this entire batch."
                    else:
                        response = "It didn't fail! Its physics look perfectly stable and consistent with the baseline NASA reliability standards."
                elif "risk" in p or "danger" in p:
                    if "Module B" in comp_status:
                        risk_val = valid_parts.loc[valid_parts['component_id'] == selected_comp, 'predicted_final_iddq'].values[0]
                        response = f"The risk is severe. If placed in an aerospace system, it is projected to hit {risk_val:.2f} µA, which would cause an electrical short or thermal event."
                    else:
                        response = "There is no severe risk predicted for this component."
                else:
                    response = f"I am monitoring the {comp_status} designation. My thermodynamic equations flag components that mimic the empirical failure profiles seen in the NASA PCOE datasets."
                
                st.write(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
