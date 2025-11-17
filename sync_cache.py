import requests
import pandas as pd
from datetime import datetime
import time
import os
import sys
# --- Importations pour l'environnement local ---
from tqdm import tqdm 
from dotenv import load_dotenv 

# Charger les variables d'environnement depuis le fichier .env
# Assurez-vous que ce fichier est présent et non versionné (.gitignore)
load_dotenv()

# ----------------------------------------------------------------

# Ajouter le répertoire courant au chemin pour les imports locaux
# Ceci est nécessaire pour s'assurer que Python trouve db_manager.py
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Importation des fonctions du gestionnaire de DB
# ATTENTION: Le fichier doit s'appeler 'db_manager.py'
from db_manager import init_db 
from db_manager import save_activity_to_db, load_activity_from_db 


# --- Configuration et variables (chargées par load_dotenv()) ---
CLIENT_ID = os.getenv("STRAVA_CLIENT_ID")
CLIENT_SECRET = os.getenv("STRAVA_CLIENT_SECRET")
REFRESH_TOKEN = os.getenv("STRAVA_REFRESH_TOKEN")
# Chemin de la base de données. Par défaut, utilise le fichier local.
DB_PATH = os.getenv("DB_PATH", "strava_cache.db") 

# LIMITE POUR L'EXÉCUTION DU CACHE : Limite le nombre d'activités détaillées téléchargées par exécution
# Ceci permet de respecter la limite de l'API Strava (900 requêtes / 15 minutes)
MAX_ACTIVITIES_TO_CACHE_PER_RUN = 10
# Limite pour la récupération de la liste d'activités (on lit le maximum pour voir les nouvelles)
MAX_ACTIVITIES_LIST = 500
# ---------------------------------------------------------------


# --- Fonctions utilitaires API Strava (Adaptées pour CLI/Automation) ---

def get_access_token_cli(refresh_token):
    """Rafraîchit le jeton d'accès pour l'environnement CLI."""
    if not refresh_token:
        print("Erreur: Jeton de rafraîchissement manquant.")
        return None
        
    url = "https://www.strava.com/api/v3/oauth/token"
    payload = {
        'client_id': CLIENT_ID,
        'client_secret': CLIENT_SECRET,
        'grant_type': 'refresh_token',
        'refresh_token': refresh_token
    }
    
    try:
        response = requests.post(url, data=payload)
        response.raise_for_status()
        new_tokens = response.json()
        print("Jeton d'accès rafraîchi avec succès.")
        return new_tokens['access_token']
    except requests.exceptions.RequestException as e:
        print(f"Erreur lors du rafraîchissement du jeton: {e}")
        return None

def get_all_activity_ids_cli(access_token, max_activities=MAX_ACTIVITIES_LIST):
    """
    Récupère la liste complète des IDs d'activité en utilisant la pagination.
    """
    url = "https://www.strava.com/api/v3/athlete/activities"
    all_activities = []
    page = 1
    per_page = 200 # Maximum autorisé par page par Strava
    
    print(f"Début de la récupération de la liste d'activités (jusqu'à {max_activities})...")

    while len(all_activities) < max_activities:
        params = {
            'access_token': access_token, 
            'per_page': per_page,
            'page': page
        }
        
        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            activities = response.json()
            
            if not activities:
                # Aucune activité sur cette page, nous avons atteint la fin
                break
            
            # Stocke l'ID et le nom pour le prétraitement
            all_activities.extend([{'id': activity['id'], 'name': activity['name']} for activity in activities])
            page += 1
            time.sleep(0.5) 

        except requests.exceptions.RequestException as e:
            print(f"Erreur lors de la récupération des activités à la page {page}: {e}")
            break
            
    return all_activities[:max_activities]

def get_activity_data_and_save(activity_id, access_token):
    """
    Récupère les données d'une activité (y compris la puissance et la cadence) 
    depuis l'API et les sauvegarde dans la DB.
    """
    
    # 1. Récupération des détails (nom, type de sport et DATE DE DÉBUT)
    url_details = f"https://www.strava.com/api/v3/activities/{activity_id}"
    params_details = {'access_token': access_token}
    response_details = requests.get(url_details, params=params_details)
    
    activity_name = "Activité sans nom"
    sport_type = None
    activity_start_date = None

    if response_details.status_code == 200:
        activity_details = response_details.json()
        activity_name = activity_details.get('name', activity_name)
        activity_id = activity_details.get('id', activity_id)
        sport_type = activity_details.get('sport_type')
        activity_start_date = activity_details.get('start_date_local')
        has_streams = activity_details.get('has_heartrate', False) or activity_details.get('has_latlng', False)
    else:
        return False, activity_name 
    
    # Cas des activités sans stream de données détaillées (Yoga, Musculation, etc.)
    if sport_type in ['Workout', 'Yoga', 'VirtualRide', 'WeightTraining'] or not has_streams:
        empty_df = pd.DataFrame({'temps_relatif_sec': [0], 'distance_m': [0], 'altitude_m': [0]})
        # *** Appel corrigé avec DB_PATH ***
        save_activity_to_db(activity_id, activity_name, sport_type, empty_df, activity_start_date)
        return True, activity_name 
        
    # 2. Récupération des streams détaillés
    # Inclut la puissance, la cadence, et la vitesse lissée
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
        return False, activity_name
        
    data = response_streams.json()
    
    # Construction du DataFrame avec toutes les colonnes demandées
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
        'surface' : data.get('surface', {}).get('data') # Surface
    })
    
    if df.empty or 'temps_relatif_sec' not in df.columns or df['temps_relatif_sec'].isnull().all():
        return False, activity_name
    
    # # --- NOUVEAU: FILTRAGE DES POINTS OÙ RESTING EST TRUE ---
    # if 'resting' in df.columns:
    #     # On ne garde que les points où resting est False (ou n'est pas True)
    #     # Note: Si la colonne contient des NaN ou des valeurs nulles, ce filtre les conserve,
    #     # supposant que l'absence d'information signifie l'absence de repos confirmé.
    #     df = df[df['resting'] == False].copy()

    # if 'moving' in df.columns:
    #     # On ne garde que les points où resting est False (ou n'est pas True)
    #     # Note: Si la colonne contient des NaN ou des valeurs nulles, ce filtre les conserve,
    #     # supposant que l'absence d'information signifie l'absence de repos confirmé.
    #     df = df[df['moving'] == True].copy()
    
    # Vérification après filtrage (pour les activités courtes ou avec beaucoup d'arrêts)
    if df.empty:
        # Si après filtrage, le DF est vide, on renvoie un DF minimal pour éviter des erreurs
        empty_df = pd.DataFrame({'temps_relatif_sec': [0], 'distance_m': [0], 'altitude_m': [0]})
        save_activity_to_db(activity_id, activity_name, sport_type, empty_df, activity_start_date)
        return True, activity_name

    # 3. Sauvegarder dans la DB
    # *** Appel corrigé avec DB_PATH ***
    save_activity_to_db(activity_id, activity_name, sport_type, df, activity_start_date)
    return True, activity_name


# --- Fonction principale de synchronisation ---

def sync_cache_main():
    """Fonction principale exécutée en local."""
    print("--- Démarrage de la synchronisation du cache Strava ---")
    print(f"Fichier de base de données cible: {DB_PATH}") 
    
    # Vérification des variables après le chargement de load_dotenv()
    if not all([CLIENT_ID, CLIENT_SECRET, REFRESH_TOKEN]):
        print("Erreur: Les variables d'environnement (CLIENT_ID, CLIENT_SECRET, REFRESH_TOKEN) ne sont pas définies.")
        print("Veuillez vérifier votre fichier .env et vos jetons.")
        return

    # 1. Initialiser la DB si nécessaire
    # *** Appel corrigé avec DB_PATH ***
    init_db() 
    print("Base de données SQLite initialisée.")

    # 2. Obtenir le jeton d'accès
    access_token = get_access_token_cli(REFRESH_TOKEN)
    if not access_token:
        print("Échec de l'obtention du jeton d'accès. Arrêt.")
        return

    # 3. Récupérer la liste des activités complètes (avec pagination)
    activities_list = get_all_activity_ids_cli(access_token) 
    print(f"Liste d'activités récupérée. Total: {len(activities_list)}.")

    # 4. Prétraitement: Identifier les activités qui ne sont PAS déjà en cache
    activities_to_process = []
    
    print("Vérification du cache local pour déterminer les activités à télécharger...")
    for activity in activities_list:
        activity_id = activity['id']
        # load_activity_from_db retourne None si l'activité n'est pas trouvée
        # *** Appel corrigé avec DB_PATH ***
        df_cached, _, _, _ = load_activity_from_db(activity_id) 

        if df_cached is None: 
             activities_to_process.append(activity)
    
    
    # --- APPLICATION DE LA LIMITE ---
    if len(activities_to_process) > MAX_ACTIVITIES_TO_CACHE_PER_RUN:
        print(f"Limite fixée à {MAX_ACTIVITIES_TO_CACHE_PER_RUN} nouvelles activités pour cette exécution.")
        activities_to_process = activities_to_process[:MAX_ACTIVITIES_TO_CACHE_PER_RUN]
    
    print(f"Activités à mettre à jour/télécharger: {len(activities_to_process)}")
    
    if not activities_to_process:
        print("Cache déjà à jour. Aucune nouvelle activité à traiter.")
        return
        
    print("--- Début du téléchargement et du caching ---")
    
    # 5. Parcourir et mettre en cache chaque NOUVELLE activité avec barre de progression
    
    for activity in tqdm(activities_to_process, desc="Progression du Cache", unit=" activité"):
        activity_id = activity['id']
        
        success, activity_name = get_activity_data_and_save(activity_id, access_token)
        
        if success:
            tqdm.write(f"Cache OK pour : {activity_name[:40]}...") 
        else:
             tqdm.write(f"Cache ÉCHOUÉ pour : {activity_name[:40]}...")
        
        # API Rate Limit Strava: attendre 1 seconde
        time.sleep(1.0) 

    print("\n--- Synchronisation du cache terminée. Le fichier strava_cache.db est à jour. ---")

if __name__ == "__main__":
    sync_cache_main()
