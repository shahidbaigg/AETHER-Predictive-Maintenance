AETHER-DPAT: Aerospace Predictive Maintenance Engine

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://aether-prediction-engine-qy5brp2yhyhs8rfyytjpij.streamlit.app/)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Real-time telemetry analysis, latent defect forecasting, and explainable AI for next-generation aerospace reliability.

The Problem
In the aerospace sector, a single undetected semiconductor defect can lead to catastrophic thermal runaway in orbit. Traditional static testing using standard pass/fail limits is insufficient. Engineers need to know not just if a component is broken today, but if it will silently degrade and fail during prolonged operation.

The Solution
AETHER-DPAT is a multi-stage machine learning inference engine designed to evaluate 24-hour sensor drift (Leakage, Delay, Resistance) and forecast 168-hour breakdown events before they happen. Rather than functioning as a black-box model, AETHER is built on Explainable AI (XAI) principles, providing engineers with interactive mathematical proofs for every rejected component.

Core Architecture

1. The Gatekeeper (Immediate Anomaly Isolation)
Uses Mahalanobis Distance and Isolation Forests to mathematically evaluate 24-hour sensor drift. It instantly flags and quarantines gross defects that deviate significantly from the lot average.

2. The Predictor (Latent Defect Forecasting)
Healthy components are passed to a highly calibrated Ridge Regression model utilizing second-degree polynomial feature interactions. It identifies Time-Dependent Dielectric Breakdown (TDDB) signatures to predict which components will cross the critical failure threshold by hour 168.

3. Explainable AI (Mathematical Transparency)
Integrated SHAP (SHapley Additive exPlanations) generates interactive waterfall charts on the fly. Engineers see the exact physical interactions (e.g., Leakage Drift × Delay Drift) that triggered a rejection, ensuring full accountability.

4. Heuristic Diagnostic Assistant
A context-aware chatbot is integrated directly into the dashboard. By selecting any component, the AI assistant explains the specific physics, trajectory, and risk factors behind its current status in plain English.

Technical Stack
- Frontend: Streamlit, Custom CSS (Dark Theme)
- Visualizations: Plotly Interactive Graphs, Matplotlib
- Machine Learning: Scikit-Learn (RidgeCV, IsolationForest, PolynomialFeatures)
- Explainability: SHAP (LinearExplainer)
- Data Processing: Pandas, NumPy, SciPy

Quick Start

Live Cloud Engine
Experience the inference engine directly in the browser. No installation or local environment configuration required.

[Access the Live AETHER Dashboard Here](https://aether-prediction-engine-qy5brp2yhyhs8rfyytjpij.streamlit.app/)

Note: You can download the CSV template directly from the dashboard sidebar to see the required data format, or generate a 5,000-component test batch using the local generation script.

Run Locally
To inspect the codebase and run the architecture locally:
1. Clone the repository:git clone [https://github.com/YourUsername/AETHER-Predictive-Maintenance.git](https://github.com/YourUsername/AETHER-Predictive-Maintenance.git)
   cd AETHER-Predictive-Maintenance

2. Install the required dependencies:
pip install -r requirements.txt

3. Launch the Streamlit server:
python -m streamlit run module_c_dashboard.py

Repository Structure

* `module_c_dashboard.py`: The main Streamlit application and integrated ML pipeline.
* `generate_test.py`: Script to synthesize a 5,000-component blind test batch.
* `telemetry_synthetic_master.csv`: The baseline training data used to calibrate the models.
* `requirements.txt`: Python dependencies for cloud deployment.

```

```
