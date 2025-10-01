import streamlit as st
import pandas as pd
import numpy as np
# Importation de scipy.signal pour Savitzky-Golay
from scipy.signal import savgol_filter 

# Importation de l'utilitaire de formatage (déplacé dans components/utils.py)
from components.utils import format_allure



# --- Paramètres de Nettoyage ---
# Allure maximale tolérée. Au-delà, c'est considéré comme une erreur ou un arrêt.
ALLURE_MAX_ACCEPTABLE = 45.0  # min/km (Ex: 20 min/km = 3 km/h, vitesse de marche très lente)
# Allure minimale tolérée. En dessous, c'est considéré comme une erreur de capteur/GPS.
ALLURE_MIN_ACCEPTABLE = 1.0   # min/km (Ex: 1 min/km = 60 km/h)
# ------------------------------

def calculate_vap(allure_min_km: pd.Series, pente_perc: pd.Series) -> pd.Series:
    """Calcule la Vitesse Ajustée à la Pente (VAP)."""
    i = pente_perc / 100
    # Coefficients tirés d'un modèle (ex: Minetti)
    Cr = (155.4 * (i**5) - 30.4 * (i**4) - 43.3 * (i**3) + 46.3 * (i**2) + 19.5 * i + 3.6)
    Cout_Plat = 3.6
    # allure_vap = allure_plate * (Coût_Plat / Coût_Pente)
    allure_vap = allure_min_km * (Cout_Plat / Cr)
    return allure_vap

@st.cache_data
def process_data(df):
    """Effectue tous les calculs de lissage et dérive, y compris le nettoyage des extrêmes."""
    df_result = df.copy()
    
    # 1. Préparation des colonnes
    df_result['temps'] = pd.to_datetime('2025-01-01') + pd.to_timedelta(df_result['temps_relatif_sec'], unit='s')
    df_result['distance_km'] = df_result['distance_m'] / 1000
    
    # 2. Calcul des dérivées et de l'allure brute
    df_result['delta_temps_min'] = df_result['temps'].diff().dt.total_seconds().fillna(0) / 60
    # On utilise un petit dénominateur par défaut (0.001 m/s) pour éviter la division par zéro
    delta_distance_km = df_result['distance_m'].diff().fillna(0.001) / 1000
    df_result['allure_min_km'] = df_result['delta_temps_min'] / delta_distance_km
    
    # Remplacement des infinis par NaN
    df_result['allure_min_km'].replace([np.inf, -np.inf], np.nan, inplace=True)
    
    # Suppression de la première ligne (qui contient des NaN dans les deltas)
    df_result = df_result.iloc[1:].copy()

    # 3. NOUVEAU: NETTOYAGE DES VALEURS EXTRÊMES DE L'ALLURE
    # Remplacer les allures non-réalistes par NaN pour que le filtre Savitzky-Golay les ignore et les interpole
    
    # Allures trop lentes (typiques des fins d'arrêt non filtrés par 'moving')
    df_result.loc[df_result['allure_min_km'] > ALLURE_MAX_ACCEPTABLE, 'allure_min_km'] = np.nan
    # Allures trop rapides (typiques des erreurs GPS)
    df_result.loc[df_result['allure_min_km'] < ALLURE_MIN_ACCEPTABLE, 'allure_min_km'] = np.nan
    
    # 4. Lissage avec Savitzky-Golay
    
    # --- Paramètres du filtre Savitzky-Golay ---
    # Fenêtre élargie pour un lissage plus agressif sur l'allure
    WINDOW = 31     
    POLY_ORDER = 2  

    pente_non_nan = df_result['pente'].dropna().values
    allure_non_nan = df_result['allure_min_km'].dropna().values

    # Application du filtre Savitzky-Golay uniquement si suffisamment de données
    if len(pente_non_nan) >= WINDOW:
        # Lissage de la pente (déjà "smooth" par Strava, mais on l'améliore)
        pente_lisse = savgol_filter(pente_non_nan, WINDOW, POLY_ORDER)
        # Lissage de l'allure (y compris l'interpolation des NaNs créés au point 3)
        allure_lisse = savgol_filter(allure_non_nan, WINDOW, POLY_ORDER)
        
        # Réinsertion des valeurs lissées
        df_result.loc[df_result['pente'].dropna().index, 'pente'] = pente_lisse
        df_result.loc[df_result['allure_min_km'].dropna().index, 'allure_min_km'] = allure_lisse
    else:
        # Option de secours : moyenne mobile simple
        df_result['pente'] = df_result['pente'].rolling(window=WINDOW, center=True, min_periods=1).mean()
        df_result['allure_min_km'] = df_result['allure_min_km'].rolling(window=WINDOW, center=True, min_periods=1).mean()
    
    
    # 5. Calcul de la VAP (basé sur l'allure lissée)
    df_result['allure_vap'] = calculate_vap(df_result['allure_min_km'], df_result['pente'])
    
    
    # 6. Calcul des métriques finales
    df_result['vitesse_kmh'] = (60 / df_result['allure_min_km']).round(1)
    df_result['vitesse_kmh_vap'] = (60 / df_result['allure_vap']).round(1)

    # Calcul de l'efficacité de course (vitesse / FC)
    df_result['efficacite_course'] = df_result['vitesse_kmh'] / df_result['frequence_cardiaque'] if 'frequence_cardiaque' in df_result.columns else np.nan
    df_result['efficacite_course_vap'] = df_result['vitesse_kmh_vap'] / df_result['frequence_cardiaque'] if 'frequence_cardiaque' in df_result.columns else np.nan
    # Calcul de l'efficacité de pédalage (Puissance / FC)
    # Vérification que les deux colonnes existent pour le calcul
    if 'frequence_cardiaque' in df_result.columns and 'puissance_watts' in df_result.columns:
        df_result['efficacite_pedalage'] = df_result['puissance_watts'] / df_result['frequence_cardiaque']
    else:
        df_result['efficacite_pedalage'] = np.nan
        
    return df_result
