import streamlit as st
import requests
import pandas as pd
from datetime import datetime
import json
import time

# Importation des fonctions de la DB pour l'utilisation du cache
from db_manager import save_activity_to_db, load_activity_from_db

# --- Fonctions utilitaires API Strava ---

def get_access_token():
    """Rafraîchit le jeton d'accès s'il est expiré en utilisant l'état de la session."""
    if 'EXPIRES_AT' not in st.session_state:
        st.error("Jeton Strava non initialisé. Veuillez configurer les secrets.")
        return None
        
    if datetime.now().timestamp() > st.session_state.EXPIRES_AT - 60:
        st.info("Le jeton d'accès est expiré, rafraîchissement en cours...")
        url = "https://www.strava.com/api/v3/oauth/token"
        payload = {
            'client_id': st.session_state.CLIENT_ID,
            'client_secret': st.session_state.CLIENT_SECRET,
            'grant_type': 'refresh_token',
            'refresh_token': st.session_state.REFRESH_TOKEN
        }
        try:
            response = requests.post(url, data=payload)
            response.raise_for_status()
            new_tokens = response.json()
            
            st.session_state.ACCESS_TOKEN = new_tokens['access_token']
            st.session_state.REFRESH_TOKEN = new_tokens['refresh_token']
            st.session_state['EXPIRES_AT'] = new_tokens['expires_at']
            
            st.success("Jeton d'accès rafraîchi avec succès.")
        except requests.exceptions.RequestException as e:
            st.error(f"Erreur lors du rafraîchissement du jeton : {e}")
            return None
    return st.session_state.ACCESS_TOKEN

@st.cache_data(ttl=3600)
def get_last_activity_ids(num_activities=200):
    """Récupère une liste des IDs et noms des dernières activités de l'athlète."""
    if 'CLIENT_ID' not in st.session_state:
        return []
        
    access_token = get_access_token()
    if not access_token:
        return []

    url = "https://www.strava.com/api/v3/athlete/activities"
    # On augmente le nombre d'activités pour le tableau de bord
    params = {'access_token': access_token, 'per_page': num_activities} 
    
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        activities = response.json()
        
        activity_info = [{'id': activity['id'], 'name': activity['name']} for activity in activities]
        return activity_info
    except requests.exceptions.RequestException as e:
        st.error(f"Erreur lors de la récupération des activités : {e}")
        return []

def get_activity_data_from_api(activity_id):
    """
    Récupère les données brutes d'une activité Strava.
    Vérifie d'abord si les données sont en cache (DB). Si non, appelle l'API et sauvegarde.
    """
    if 'CLIENT_ID' not in st.session_state:
        st.error("Impossible de récupérer les données : Authentification Strava non configurée.")
        return pd.DataFrame(), None, None
        
    # 1. Tenter de charger depuis la DB (cache)
    df, activity_name, sport_type, activity_start_date = load_activity_from_db(activity_id)
    if df is not None:
        return df, activity_name, sport_type, activity_start_date
    
    # 2. Si non trouvé, appeler l'API
    st.sidebar.warning(f"Cache DB : Activité {activity_id} non trouvée. Appel API Strava en cours...")
    access_token = get_access_token()
    if not access_token:
        return pd.DataFrame(), None, None, None

    # Récupération des détails (nom, type de sport et DATE DE DÉBUT)
    url_details = f"https://www.strava.com/api/v3/activities/{activity_id}"
    params_details = {'access_token': access_token}
    response_details = requests.get(url_details, params=params_details)
    
    activity_name = "Activité sans nom"
    sport_type = None
    activity_start_date = None # Pour stocker la date de début Strava

    if response_details.status_code == 200:
        activity_details = response_details.json()
        activity_name = activity_details.get('name', activity_name)
        sport_type = activity_details.get('sport_type')
        activity_start_date = activity_details.get('start_date_local')
        has_streams = activity_details.get('has_heartrate') or activity_details.get('has_latlng') # Indice rapide
    else:
        st.error(f"Erreur lors de la requête API pour les détails: {response_details.status_code}")
        return pd.DataFrame(), None, None, None
    
    # Vérification : si l'activité est manuelle ou sans données (ex: Workout)
    if sport_type in ['Workout', 'Yoga', 'VirtualRide', 'WeightTraining'] or not has_streams:
        st.warning(f"L'activité est de type {sport_type} et n'a probablement pas de données de stream détaillées.")
        # On sauve quand même une activité vide dans la DB pour ne pas la re-requêter
        save_activity_to_db(activity_id, activity_name, sport_type, pd.DataFrame({
            'temps_relatif_sec': [0], 'distance_m': [0], 'altitude_m': [0]
        }), activity_start_date)
        return pd.DataFrame(), activity_name, sport_type, activity_start_date
        
    # Récupération des streams
    streams_types = ['time','distance', 'velocity_smooth','altitude', 'latlng', 'heartrate', 'watts',
                     'cadence','grade_smooth','moving','resting','outlier','surface']
    url_streams = f"https://www.strava.com/api/v3/activities/{activity_id}/streams"
    params_streams = {
        'access_token': access_token,
        'keys': ','.join(streams_types),
        'key_by_type': 'true'
    }
    response_streams = requests.get(url_streams, params=params_streams)
    
    if response_streams.status_code != 200:
        st.warning(f"Erreur lors de la requête API pour les streams : L'activité est peut-être manuelle.")
        return pd.DataFrame(), activity_name, sport_type, activity_start_date
        
    data = response_streams.json()
    df = pd.DataFrame({
        'temps_relatif_sec': data.get('time', {}).get('data'), # Temps en secondes
        'distance_m': data.get('distance', {}).get('data'), # Distance en m
        'vitesse_lissee': data.get('velocity_smooth', {}).get('data'), # Vitesse lissée
        'altitude_m': data.get('altitude', {}).get('data'), # Altitude en m
        'latlng': data.get('latlng', {}).get('data'), # Coordonnées GPS
        'frequence_cardiaque': data.get('heartrate', {}).get('data'), # Fréquence cardiaque
        'puissance_watts': data.get('watts', {}).get('data'), # Puissance (watts)
        'cadence': data.get('cadence', {}).get('data'), # Cadence (RPM ou SPM)
        'pente_lissee': data.get('grade_smooth', {}).get('data'), # pente lissée
        'moving' : data.get('moving', {}).get('data'), # En mouvement
        'resting' : data.get('resting', {}).get('data'),# Au repos
        'outlier' : data.get('outlier', {}).get('data'), # Valeurs aberrantes
        'surface' : data.get('surface', {}).get('data'), # Surface
    })

    
    
    if df.empty or 'temps_relatif_sec' not in df.columns:
        st.warning("Aucune donnée de stream temporelle disponible pour cette activité.")
        return pd.DataFrame(), activity_name, sport_type, activity_start_date
    
    # Vérification après filtrage (pour les activités courtes ou avec beaucoup d'arrêts)
    if df.empty:
        # Si après filtrage, le DF est vide, on renvoie un DF minimal pour éviter des erreurs
        empty_df = pd.DataFrame({'temps_relatif_sec': [0], 'distance_m': [0], 'altitude_m': [0]})
        save_activity_to_db(activity_id, activity_name, sport_type, empty_df, activity_start_date)
        return True, activity_name

    # Nettoyage et préparation de base
    df = df.dropna(subset=['temps_relatif_sec'])
    if 'latlng' in df.columns and df['latlng'].any() and not df['latlng'].isnull().all():
        try:
             # Utilisation de dropna(subset=['latlng']) pour éviter les erreurs si la ligne est vide
             latlng_data = df.dropna(subset=['latlng'])['latlng'].tolist()
             df[['latitude', 'longitude']] = pd.DataFrame(latlng_data, index=df.dropna(subset=['latlng']).index)
        except:
             st.warning("Impossible d'extraire les coordonnées latitude/longitude.")
    
    df['sport_type'] = sport_type
    
    # 3. Sauvegarder dans la DB avant de retourner
    save_activity_to_db(activity_id, activity_name, sport_type, df, activity_start_date)
    
    return df, activity_name, sport_type, activity_start_date