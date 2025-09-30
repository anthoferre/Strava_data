import streamlit as st
import numpy as np
import pandas as pd

# --- Fonctions utilitaires et de Formatage ---
def format_allure(allure_decimal):
    """Convertit une allure décimale (ex: 5.45 min/km) en une chaîne formatée 'min'sec'''."""
    if pd.isna(allure_decimal) or allure_decimal <= 0:
        return np.nan
    minutes = int(allure_decimal)
    secondes = int(round((allure_decimal - minutes) * 60))
    if secondes == 60:
        minutes += 1
        secondes = 0
    return f"{minutes}'{secondes:02d}''"

def format_allure_std(allure_decimal):
    """Convertit une allure décimale (ex: 5.45 min/km) en une chaîne formatée '+- min'sec'''."""
    if pd.isna(allure_decimal) or allure_decimal <= 0:
        return np.nan
    minutes = int(allure_decimal)
    secondes = int(round((allure_decimal - minutes) * 60))
    if secondes == 60:
        minutes += 1
        secondes = 0
    return f"± {minutes}'{secondes:02d}''"

def display_metric_card(title, value, icon, sub_value=None):
    """Affiche une carte de métrique stylisée avec une sous-valeur."""
    sub_html = f'<p style="margin: 0; font-size: 0.8em; color: #666; font-style: italic; white-space: pre-line;">{sub_value}</p>' if sub_value else ''
    st.markdown(f"""
    <div style="
        border: 1px solid #ddd;
        border-radius: 5px;
        padding: 10px;
        margin: 5px;
        background-color: #f9f9f9;
        text-align: center;
        white-space: normal;
        overflow-wrap: break-word;
        height: 100%;
    ">
        <h4 style="margin: 0; font-size: 1em;">{title}</h4>
        <h3 style="margin: 5px 0 0; font-size: 1.5em; color: #333; white-space: pre-line;">{value} {icon}</h3>
        {sub_html}
    </div>
    """, unsafe_allow_html=True)