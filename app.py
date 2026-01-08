import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(page_title="FedEx DCA Management System", layout="wide")

# Interface Header
st.title("Debt Collection Agency (DCA) Control Tower")
st.markdown("---")

# Data Simulation Engine
def load_internal_data():
    records = {
        'Account_ID': [f'FX-{i}' for i in range(5001, 5011)],
        'Balance_Due': [450, 1350, 280, 5200, 920, 2400, 180, 850, 3900, 1250],
        'Aging_Days': [32, 91, 14, 125, 41, 65, 12, 88, 115, 38],
        'Assigned_Partner': ['Global Collect', 'Apex Recovery', 'Global Collect', 'Zenith Assets', 'Apex Recovery', 'Global Collect', 'Zenith Assets', 'Apex Recovery', 'Global Collect', 'Zenith Assets']
    }
    return pd.DataFrame(records)

registry = load_internal_data()

# Predictive Scoring Logic (AI)
features = registry[['Balance_Due', 'Aging_Days']]
labels = ((registry['Aging_Days'] > 90) | (registry['Balance_Due'] > 2500)).astype(int)
clf = RandomForestClassifier(n_estimators=100)
clf.fit(features, labels)
registry['Recovery_Priority'] = clf.predict(features)

# Executive Dashboard Metrics
m1, m2, m3 = st.columns(3)
m1.metric("Portfolio Value", f"${registry['Balance_Due'].sum():,}")
m2.metric("Critical Accounts", len(registry[registry['Recovery_Priority'] == 1]))
m3.metric("Partner Agencies", registry['Assigned_Partner'].nunique())

# Portfolio View
st.subheader("Account Prioritization Matrix")
st.dataframe(registry.style.background_gradient(subset=['Recovery_Priority'], cmap='Reds'))

# Distribution Analytics
st.subheader("Agency Workload Distribution")
chart = px.bar(registry, x='Assigned_Partner', y='Balance_Due', color='Recovery_Priority', 
             color_continuous_scale='RdYlGn_r', barmode='group')
st.plotly_chart(chart, use_container_width=True)
