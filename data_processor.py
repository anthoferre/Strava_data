import streamlit as st
import pandas as pd
import numpy as np

# Importation de l'utilitaire de formatage (déplacé dans components/utils.py)
from components.utils import format_allure

# --- Fonctions de Traitement de Données ---

@st.cache_data
def process_data(df, window_size, trim_seconds, window_size_fc=None):
    """Effectue tous les calculs de lissage et dérive"""
    df_result = df.copy()
    if 'temps_relatif_sec' not in df_result.columns or df_result['temps_relatif_sec'].empty:
        return None
        
    if window_size_fc is None:
        window_size_fc = window_size # Utiliser la même taille par défaut

    # Rognage du début
    df_result = df_result[df_result['temps_relatif_sec'] > trim_seconds].copy()
    
    if df_result.empty or len(df_result) < 2:
        return None
        
    df_result.reset_index(drop=True, inplace=True)
    
    # Remise à zéro des valeurs
    df_result['temps_relatif_sec'] = df_result['temps_relatif_sec'] - df_result['temps_relatif_sec'].iloc[0]
    df_result['distance_m'] = df_result['distance_m'] - df_result['distance_m'].iloc[0]
    
    df_result['temps'] = pd.to_datetime('2025-01-01') + pd.to_timedelta(df_result['temps_relatif_sec'], unit='s')
    df_result['distance_km'] = df_result['distance_m'] / 1000
    
    # Calcul des dérivées et de l'allure
    df_result['delta_altitude'] = df_result['altitude_m'].diff().fillna(0)
    df_result['delta_distance'] = df_result['distance_m'].diff().fillna(1)
    df_result['pente_deg'] = np.degrees(np.arctan(df_result['delta_altitude'] / df_result['delta_distance'].replace(0, 1)))
    df_result['delta_temps_min'] = df_result['temps'].diff().dt.total_seconds().fillna(0) / 60
    df_result['allure_min_km'] = df_result['delta_temps_min'] / (df_result['distance_m'].diff().fillna(0.001)/1000)
    
    df_result = df_result.iloc[1:].copy()
    
    # Lissage
    df_result['pente_lisse'] = df_result['pente_deg'].rolling(window=window_size * 2, min_periods=1).mean()
    df_result['allure_lisse'] = df_result['allure_min_km'].rolling(window=window_size, min_periods=1).mean()
    
    
    # Utilisation de l'Allure lissée normale comme 'corrigee' par défaut pour la compatibilité
    df_result['allure_lisse_corrigee'] = df_result['allure_lisse'] 
    
    # Lissage de la FC (avec imputation et taille de fenêtre FC spécifique)
    if 'frequence_cardiaque' in df_result.columns and df_result['frequence_cardiaque'].any():
        # Imputation linéaire pour les petites coupures de signal (max 3 points)
        df_result['fc_interp'] = df_result['frequence_cardiaque'].interpolate(method='linear', limit=3, limit_direction='both')
        df_result['fc_lisse'] = df_result['fc_interp'].rolling(window=window_size_fc, min_periods=1).mean().round(0)
    
    # Calcul de la vitesse et de l'allure formatée
    df_result['vitesse_kmh'] = (60 / df_result['allure_lisse_corrigee']).round(1)
    df_result['allure_str'] = df_result['allure_lisse_corrigee'].apply(format_allure)

    # Calcul de l'efficacité de course (vitesse / FC)
    df_result['efficacite_course'] = df_result['vitesse_kmh'] / df_result['fc_lisse'] if 'fc_lisse' in df_result.columns else np.nan
    
    return df_result

def segmenter_segments_majeurs(df_processed, seuil_pente_perc=3.0, min_denivele_m=50.0):
    """
    Identifie et regroupe les segments de Montée et de Descente majeurs 
    en utilisant les données déjà lissées.
    
    Args:
        df_processed (pd.DataFrame): DataFrame traité par process_data, contenant 'pente_lisse_perc' et 'delta_alt_lisse'.
        seuil_pente_perc (float): Seuil de pente en pourcentage (%) pour définir un segment.
        min_denivele_m (float): Dénivelé cumulé minimal (en mètres) pour considérer un segment comme 'majeur'.
        
    Returns:
        pd.DataFrame: Tableau récapitulatif des segments majeurs.
    """
    
    df = df_processed.copy()
    
    # 1. Catégorisation des points en fonction du seuil de pente lissée
    df['segment_type'] = np.select(
        [
            df['pente_lisse_perc'] >= seuil_pente_perc,
            df['pente_lisse_perc'] <= -seuil_pente_perc
        ],
        [
            'Montée', 
            'Descente'
        ],
        default='Plat'
    )
    
    # 2. Identification des blocs contigus de même type
    df['segment_id'] = (df['segment_type'] != df['segment_type'].shift(1)).cumsum()
    
    # 3. Aggrégation et Résumé de chaque segment
    segments_agg = df.groupby('segment_id').agg(
        type=('segment_type', 'first'),
        denivele_cumule=('delta_alt_lisse', 'sum'),
        distance_total_m=('delta_distance', 'sum'),
        distance_debut=('distance_km', 'first'),
        distance_fin=('distance_km', 'last')
    ).reset_index()
    
    # 4. Filtrage des Segments Majeurs
    
    # On garde seulement les Montées/Descentes qui respectent le dénivelé minimal
    segments_majeurs = segments_agg[
        (segments_agg['type'] != 'Plat') & 
        (segments_agg['denivele_cumule'].abs() >= min_denivele_m)
    ].copy()
    
    # Formatage des résultats pour l'affichage
    segments_majeurs['Dénivelé (m)'] = segments_majeurs['denivele_cumule'].round(1)
    segments_majeurs['Distance Totale (km)'] = (segments_majeurs['distance_total_m'] / 1000).round(2)
    segments_majeurs['Début (km)'] = segments_majeurs['distance_debut'].round(2)
    segments_majeurs['Fin (km)'] = segments_majeurs['distance_fin'].round(2)
    
    return segments_majeurs.rename(columns={'type': 'Type'})[[
        'Type', 'Début (km)', 'Fin (km)', 'Distance Totale (km)', 'Dénivelé (m)'
    ]]