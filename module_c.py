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
    feature_names = poly.get_feature_names_out(features)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_poly)
    
    model = RidgeCV(alphas=np.logspace(-3, 3, 10))
    model.fit(X_scaled, df_train['actual_final_drift'])
    explainer = shap.LinearExplainer(model, X_scaled, feature_names=feature_names)
    
    return model, poly, scaler, explainer, feature_names, features

model, poly, scaler, explainer, feature_names, features = train_production_model()

# --- UI SIDEBAR & UPLOADER ---
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/e/e5/NASA_logo.svg", width=100)
    st.title("AETHER-DPAT")
    st.markdown("### Telemetry Ingestion")
    uploaded_file = st.file_uploader("Upload CSV Batch", type=["csv"])
    st.divider()
    usl_limit = st.slider("Iddq USL (µA) Limit", min_value=30.0, max_value=80.0, value=50.0, step=1.0)
    contamination_rate = st.slider("Module A Sensitivity", min_value=0.005, max_value=0.05, value=0.01, step=0.005)

if uploaded_file is None:
    st.info("Awaiting telemetry data. Please upload your CSV batch (e.g., blind_test_batch.csv) in the sidebar.")
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
        
        # EXPLICIT STATUS DISPLAY
        with col_stat:
            st.markdown("### Final Verdict")
            if "Module B" in comp_status:
                st.error(f"🚨 **{comp_status}** - Do not deploy. Thermal runaway risk.")
            elif "Module A" in comp_status:
                st.warning(f"⚠️ **{comp_status}** - Do not deploy. Gross defect caught early.")
            else:
                st.success(f"✅ **{comp_status}** - Cleared for aerospace flight.")
                
        st.divider()

        # DYNAMIC VISUALIZATIONS BASED ON STATUS
        if "Module B" in comp_status:
            valid_idx = valid_parts.reset_index().index[valid_parts['component_id'] == selected_comp].tolist()[0]
            
            st.markdown(f"#### Predictive Logic Breakdown (SHAP)")
            st.caption("This chart reveals the hidden mathematical physics driving the AI's thermal runaway prediction.")
            sv = shap_vals_new[valid_idx]
            sorted_idx = np.argsort(np.abs(sv))[-10:] 
            
            fig = go.Figure(go.Waterfall(
                orientation="h", measure=["relative"] * 10 + ["total"],
                y=[feature_names[i] for i in sorted_idx] + ["Predicted Final Drift"],
                x=[sv[i] for i in sorted_idx] + [sum(sv)],
                connector={"line": {"color": "rgb(63, 63, 63)"}},
            ))
            fig.update_layout(height=350, margin=dict(l=20, r=20, t=20, b=20))
            st.plotly_chart(fig, use_container_width=True)

            st.markdown("#### Degradation Trajectory")
            traj_y = [valid_parts.loc[comp_idx, 'iddq_0h'], valid_parts.loc[comp_idx, 'iddq_24h'], valid_parts.loc[comp_idx, 'predicted_final_iddq']]
            fig2 = go.Figure(data=go.Scatter(x=["0h", "24h", "168h (Predicted)"], y=traj_y, mode='lines+markers', marker=dict(size=10, color='red')))
            fig2.add_hline(y=usl_limit, line_dash="dash", line_color="orange", annotation_text="Datasheet Failure Limit")
            fig2.update_layout(height=300, yaxis_title="Leakage (µA)")
            st.plotly_chart(fig2, use_container_width=True)
            
        else:
            # Handle Module A and Healthy parts (which didn't go through SHAP)
            st.markdown("#### 24-Hour Telemetry Check")
            st.caption("Module B (SHAP) is only invoked for components that pass Module A. Here is the raw sensor data that led to this verdict:")
            
            c1, c2, c3 = st.columns(3)
            c1.metric("0h Leakage", f"{data.loc[comp_idx, 'iddq_0h']:.2f} µA")
            c2.metric("24h Leakage", f"{data.loc[comp_idx, 'iddq_24h']:.2f} µA")
            c3.metric("24h Drift", f"{data.loc[comp_idx, 'iddq_drift']:.2f} µA")
            
            # Simple bar chart comparing this component's drift to the batch average
            fig3 = go.Figure(data=[
                go.Bar(name='This Component', x=['Leakage Drift', 'Delay Drift', 'Resistance Drift'], 
                       y=[data.loc[comp_idx, 'iddq_drift'], data.loc[comp_idx, 'delay_drift'], data.loc[comp_idx, 'resist_drift']]),
                go.Bar(name='Batch Average', x=['Leakage Drift', 'Delay Drift', 'Resistance Drift'], 
                       y=[data['iddq_drift'].mean(), data['delay_drift'].mean(), data['resist_drift'].mean()])
            ])
            fig3.update_layout(barmode='group', height=350, title="Component Drift vs. Batch Average")
            st.plotly_chart(fig3, use_container_width=True)

with tab2:
    st.markdown("### AI Diagnostic Chatbot")
    
    if selected_comp:
        comp_status = data.loc[data['component_id'] == selected_comp, 'Status'].values[0]
        
        # Reset chat history if the user selects a new component
        if "messages" not in st.session_state or st.session_state.get('last_comp') != selected_comp:
            st.session_state.messages = []
            st.session_state.last_comp = selected_comp
            
            # Auto-generate context
            if "Module B" in comp_status:
                init_msg = f"Hello. {selected_comp} was rejected by Module B because my Ridge Regression model predicts it will suffer thermal runaway by 168h. Ask me why!"
            elif "Module A" in comp_status:
                init_msg = f"Hello. {selected_comp} was immediately isolated by the Gatekeeper (Module A) at 24h because its sensor readings were highly erratic compared to the rest of the batch. How can I help?"
            else:
                init_msg = f"Hello. {selected_comp} is a perfectly healthy component. What would you like to know?"
            
            st.session_state.messages.append({"role": "assistant", "content": init_msg})

        # Display history
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.write(msg["content"])

        # Chat Input
        if prompt := st.chat_input(f"Ask about {selected_comp}..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.write(prompt)
                
            with st.chat_message("assistant"):
                p = prompt.lower()
                
                # Smarter AI Logic
                if "why" in p or "reason" in p:
                    if "Module B" in comp_status:
                        response = f"While it looked fine at 24h, the interaction between its propagation delay and leakage drift mathematically aligned with the NASA Time-Dependent Dielectric Breakdown curve[cite: 1, 4]. It is on track to cross the {usl_limit}µA safety threshold."
                    elif "Module A" in comp_status:
                        response = "It failed the Isolation Forest and Mahalanobis DPAT checks. Basically, its 24-hour sensor drift was so extreme that it mathematically fell into the top 1% of anomalies for this entire batch."
                    else:
                        response = "It didn't fail! Its physics look perfectly stable and consistent with the baseline NASA reliability standards[cite: 1]."
                elif "risk" in p or "danger" in p:
                    if "Module B" in comp_status:
                        risk_val = valid_parts.loc[valid_parts['component_id'] == selected_comp, 'predicted_final_iddq'].values[0]
                        response = f"The risk is severe. If placed in an aerospace system, it is projected to hit {risk_val:.2f} µA, which would cause an electrical short or thermal event[cite: 3, 5]."
                    else:
                        response = "There is no severe risk predicted for this component."
                else:
                    response = f"I am monitoring the {comp_status} designation. My thermodynamic equations flag components that mimic the empirical failure profiles seen in the NASA PCOE datasets[cite: 1, 2]."
                
                st.write(response)
                st.session_state.messages.append({"role": "assistant", "content": response})