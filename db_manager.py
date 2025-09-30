import streamlit as st
import sqlite3
import pandas as pd
import ast
from datetime import datetime

# --- FONCTIONS DE GESTION DE LA BASE DE DONNÉES SQLite ---

@st.cache_resource
def get_db_connection():
    """
    Crée ou se connecte au fichier de base de données et met la connexion en cache.
    Désactive le contrôle de thread pour fonctionner dans Streamlit.
    """
    conn = sqlite3.connect('strava_cache.db', check_same_thread=False)
    return conn

def init_db():
    """Initialise la table de cache si elle n'existe pas, y compris les nouvelles colonnes agrégées."""
    conn = get_db_connection()
    cursor = conn.cursor()
    # Ajout des colonnes pour les métriques agrégées (cache optimisé)
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS activities_cache (
            activity_id INTEGER PRIMARY KEY,
            activity_name TEXT,
            sport_type TEXT,
            dataframe_json TEXT,
            timestamp_cache REAL,
            activity_start_date TEXT,
            total_distance_km REAL,         -- NOUVEAU: Métrique agrégée
            total_duration_h REAL,          -- NOUVEAU: Métrique agrégée
            total_elevation_gain_m REAL     -- NOUVEAU: Métrique agrégée
        )
    ''')
    conn.commit()

def save_activity_to_db(activity_id, activity_name, sport_type, df, activity_start_date):
    """Sauvegarde un DataFrame d'activité et ses métriques agrégées dans la DB."""
    conn = get_db_connection()
    
    # 1. Calcul des métriques agrégées à stocker
    total_distance_m = df['distance_m'].iloc[-1] if 'distance_m' in df.columns and not df['distance_m'].empty else 0
    total_duration_sec = df['temps_relatif_sec'].iloc[-1] if 'temps_relatif_sec' in df.columns and not df['temps_relatif_sec'].empty else 0
    
    # Estimation du dénivelé positif
    if 'altitude_m' in df.columns:
        denivele_positif = df['altitude_m'].diff().clip(lower=0).sum()
    else:
        denivele_positif = 0.0

    # 2. Convertir le DataFrame en JSON pour le stockage des streams (détails)
    df_json = df.to_json(orient='split', date_format='iso')
    timestamp = datetime.now().timestamp()
    
    cursor = conn.cursor()
    cursor.execute('''
        INSERT OR REPLACE INTO activities_cache 
        (activity_id, activity_name, sport_type, dataframe_json, timestamp_cache, activity_start_date, total_distance_km, total_duration_h, total_elevation_gain_m)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        activity_id, 
        activity_name, 
        sport_type, 
        df_json, 
        timestamp, 
        activity_start_date, 
        total_distance_m / 1000, 
        total_duration_sec / 3600, 
        denivele_positif
    ))
    conn.commit()
    
def load_activity_from_db(activity_id):
    """Charge un DataFrame d'activité depuis la DB."""
    conn = get_db_connection()
    cursor = conn.cursor()
    # On sélectionne uniquement les colonnes nécessaires pour l'analyse détaillée
    cursor.execute('''
        SELECT activity_name, sport_type, dataframe_json, activity_start_date
        FROM activities_cache 
        WHERE activity_id = ?
    ''', (activity_id,))
    
    result = cursor.fetchone()
    if result:
        activity_name, sport_type, df_json, activity_start_date = result
        df = pd.read_json(df_json, orient='split')
        
        # S'assurer que les colonnes complexes comme latlng sont des listes si nécessaire
        if 'latlng' in df.columns:
            try:
                df['latlng'] = df['latlng'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) and x.startswith('[') else x)
            except:
                pass 
                
        start_date_display = activity_start_date.split('T')[0] if activity_start_date else 'Inconnue'
        st.sidebar.success(f"Cache DB : Activité **{activity_name}** chargée (Début: {start_date_display}).")
        return df, activity_name, sport_type
    
    
    return None, None, None

@st.cache_data
def extract_metrics_from_cache(df_cache_in):
    """
    Extrait les métriques clés *directement* des colonnes agrégées du cache. 
    Plus besoin de lire les DataFrames JSON.
    """
    data_list = []
    
    # On s'assure d'avoir les nouvelles colonnes
    if 'total_distance_km' not in df_cache_in.columns:
        st.warning("Relancez l'application : La structure du cache doit être mise à jour pour le tableau de bord.")
        return pd.DataFrame() 

    for index, row in df_cache_in.iterrows():
        try:
            date_start_str = row.get('activity_start_date')
            if date_start_str:
                date_activity = pd.to_datetime(date_start_str).date() 
            else:
                # Utiliser la date du cache comme fallback
                date_activity = datetime.fromtimestamp(row['timestamp_cache']).date()

            data_list.append({
                'id': row['activity_id'],
                'nom': row['activity_name'],
                'type_sport': row['sport_type'],
                'date': date_activity,
                'distance_km': row['total_distance_km'],
                'duree_h': row['total_duration_h'],
                'denivele_positif_m': row['total_elevation_gain_m']
            })
        except Exception as e:
            # st.error(f"Erreur lors de l'extraction de l'activité {row['activity_id']}: {e}")
            continue
    
    return pd.DataFrame(data_list)