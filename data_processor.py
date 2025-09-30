import streamlit as st
import pandas as pd
import numpy as np

# Importation de l'utilitaire de formatage (déplacé dans components/utils.py)
from components.utils import format_allure

# --- Fonctions de Traitement de Données ---

@st.cache_data
def process_data(df):
    """Effectue tous les calculs de lissage et dérive"""
    df_result = df.copy()
    
    
    df_result['temps'] = pd.to_datetime('2025-01-01') + pd.to_timedelta(df_result['temps_relatif_sec'], unit='s')
    df_result['distance_km'] = df_result['distance_m'] / 1000
    
    # Calcul des dérivées et de l'allure
    #df_result['delta_altitude'] = df_result['altitude_m'].diff().fillna(0)
    #df_result['delta_distance'] = df_result['distance_m'].diff().fillna(1)
    df_result['delta_temps_min'] = df_result['temps'].diff().dt.total_seconds().fillna(0) / 60
    df_result['allure_min_km'] = df_result['delta_temps_min'] / (df_result['distance_m'].diff().fillna(0.001)/1000)
    df_result['allure_min_km'].replace([np.inf, -np.inf], np.nan, inplace=True)
    
    
    df_result = df_result.iloc[1:].copy()
    # Lissage des valeurs
    WINDOW_SIZE = 20
    
    df_result['allure_min_km'] = df_result['allure_min_km'].rolling(
    window=WINDOW_SIZE, 
    min_periods=1,          # Pour commencer le lissage dès le premier point
    center=True             # Centre la fenêtre pour lisser uniformément
    ).mean()

   
    
    df_result['pente'] = df_result['pente'].rolling(
    window=WINDOW_SIZE, 
    min_periods=1,          # Pour commencer le lissage dès le premier point
    center=True             # Centre la fenêtre pour lisser uniformément
    ).mean()
    
    
    # Calcul de la vitesse et de l'allure formatée
    df_result['vitesse_kmh'] = (60 / df_result['allure_min_km']).round(1)

    # Calcul de l'efficacité de course (vitesse / FC)
    df_result['efficacite_course'] = df_result['vitesse_kmh'] / df_result['frequence_cardiaque'] if 'frequence_cardiaque' in df_result.columns else np.nan

    # Calcul de l'efficacité de pédalage (Puissance / FC)
    df_result['efficacite_pedalage'] = df_result['puissance_watts'] / df_result['frequence_cardiaque'] if 'frequence_cardiaque' or 'watts' in df_result.columns else np.nan
    
    return df_result

    