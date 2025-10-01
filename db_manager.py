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

def _add_missing_column(conn, column_name, column_type):
    """Fonction utilitaire pour ajouter une colonne si elle n'existe pas."""
    cursor = conn.cursor()
    try:
        cursor.execute(f"ALTER TABLE activities_cache ADD COLUMN {column_name} {column_type}")
        st.info(f"Colonne '{column_name}' ajoutée à activities_cache.")
        conn.commit()
    except sqlite3.OperationalError as e:
        # La colonne existe déjà ou autre erreur (ex: si la DB est vide, pas grave)
        if "duplicate column name" not in str(e):
             # Afficher l'erreur si elle n'est pas due à une colonne dupliquée
             pass


def init_db():
    """
    Initialise la table de cache si elle n'existe pas.
    Met à jour la structure pour inclure les nouvelles colonnes de métriques agrégées.
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    # Création de base (si la table n'existe pas)
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS activities_cache (
            activity_id INTEGER PRIMARY KEY,
            activity_name TEXT,
            sport_type TEXT,
            dataframe_json TEXT,
            timestamp_cache REAL,
            activity_start_date TEXT,
            total_distance_km REAL,
            total_duration_h REAL,
            total_elevation_gain_m REAL
        )
    ''')
    
    # --- AJOUT DES NOUVELLES COLONNES DE PROGRESSION (Mise à jour du schéma) ---
    _add_missing_column(conn, 'allure_vap_moy', 'REAL')
    _add_missing_column(conn, 'score_effort', 'REAL')
    _add_missing_column(conn, 'score_effort_efficacite', 'REAL')
    # Supprime l'ancienne colonne 'efficacite_course_vit_fc' si elle était présente dans la structure précédente.
    # Note: SQLite ne permet pas de supprimer facilement une colonne, on la laisse si elle existe, mais on n'y touche plus.
    
    conn.commit()

def save_activity_to_db(activity_id, activity_name, sport_type, df, activity_start_date):
    """Sauvegarde un DataFrame d'activité et ses métriques agrégées dans la DB (INSERT OR REPLACE)."""
    conn = get_db_connection()
    
    # 1. Calcul des métriques agrégées de base à stocker
    total_distance_m = df['distance_m'].iloc[-1] if 'distance_m' in df.columns and not df['distance_m'].empty else 0
    total_duration_sec = df['temps_relatif_sec'].iloc[-1] if 'temps_relatif_sec' in df.columns and not df['temps_relatif_sec'].empty else 0
    
    # Estimation du dénivelé positif
    denivele_positif = df['altitude_m'].diff().clip(lower=0).sum() if 'altitude_m' in df.columns else 0.0

    # 2. Convertir le DataFrame en JSON pour le stockage des streams (détails)
    df_json = df.to_json(orient='split', date_format='iso')
    timestamp = datetime.now().timestamp()
    
    # Les colonnes de progression calculées seront ajoutées par la fonction update_activity_metrics_to_db
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

# --- NOUVELLE FONCTION POUR AJOUTER LES SCORES CALCULÉS ---

# (Le reste du code de db_manager.py est ici, incluant get_db_connection)

def update_activity_metrics_to_db(metrics_dict):
    """
    Met à jour un enregistrement existant dans activities_cache avec les métriques calculées.
    La clé 'id' (activity_id Strava) est utilisée pour la clause WHERE.

    Args:
        metrics_dict (dict): Dictionnaire contenant les clés/valeurs des métriques à sauvegarder. 
                             DOIT contenir la clé 'id'.
    """
    if 'activity_id' not in metrics_dict:
        # Ceci est un cas d'erreur critique, l'ID est la clé principale.
        raise ValueError("metrics_dict doit contenir la clé 'id' pour la mise à jour en base de données.")
        
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # 1. Préparation des clauses SET (pour les colonnes à mettre à jour)
    # Exclure l'ID qui sert de condition WHERE
    set_keys = [key for key in metrics_dict.keys() if key != 'activity_id']
    set_clauses = ', '.join([f"{key} = ?" for key in set_keys])
    
    # 2. Préparation des valeurs pour les clauses SET
    set_values = [metrics_dict[key] for key in set_keys]
    
    # 3. Préparation de la valeur pour la clause WHERE
    activity_id = metrics_dict['activity_id']
    
    # 4. Fusionner les valeurs: (valeurs SET) + (valeur WHERE)
    all_values = set_values + [activity_id] 
    
    # 5. Construction de la requête finale (Utilisation de l'ID pour la mise à jour)
    sql_query = f'''
        UPDATE activities_cache 
        SET {set_clauses} 
        WHERE activity_id = ?
    '''
    
    try:
        # Exécuter avec le tuple de toutes les valeurs
        cursor.execute(sql_query, all_values)
        conn.commit()
    except Exception as e:
        conn.rollback() # On ne rollback qu'en cas d'erreur
        raise Exception(f"Erreur d'exécution SQL: {e}")
    finally:
        # NOTE: Si vous fermez la connexion ici, assurez-vous de la rouvrir à chaque appel.
        # Sinon, laissez le get_db_connection gérer une connexion persistante.
        pass
        
    
    # Suppression du conn.rollback() inutile après conn.commit()



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
    Ceci est optimisé pour le tableau de bord de progression.
    """
    data_list = []
    
    # On s'assure d'avoir les nouvelles colonnes nécessaires pour le tableau de bord
    required_cols = ['total_distance_km', 'allure_vap_moy', 'score_effort', 'score_effort_efficacite']
    if not all(col in df_cache_in.columns for col in required_cols):
        # Si des colonnes critiques manquent, demander à l'utilisateur de réexécuter l'initialisation
        st.warning("Structure du cache obsolète. Relancez l'application ou l'initialisation de la DB pour mettre à jour le schéma.")
        return pd.DataFrame() 

    for index, row in df_cache_in.iterrows():
        try:
            date_start_str = row.get('activity_start_date')
            date_activity = pd.to_datetime(date_start_str).date() if date_start_str else datetime.fromtimestamp(row['timestamp_cache']).date()

            data_list.append({
                'id': row['activity_id'],
                'nom': row['activity_name'],
                'type_sport': row['sport_type'],
                'date': date_activity,
                'distance_km': row['total_distance_km'],
                'duree_h': row['total_duration_h'],
                'denivele_positif_m': row['total_elevation_gain_m'],
                'allure_vap_moy': row['allure_vap_moy'],
                'score_effort': row['score_effort'],
                'score_effort_efficacite': row['score_effort_efficacite'],
            })
        except Exception as e:
            # Gérer les lignes incomplètes silencieusement pour ne pas bloquer le tableau de bord
            # st.error(f"Erreur lors de l'extraction de l'activité {row['activity_id']}: {e}")
            continue
    
    return pd.DataFrame(data_list)
