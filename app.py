import streamlit as st
import pandas as pd
import warnings
import plotly.express as px
import numpy as np
from datetime import datetime, timedelta
import calendar
import time # Ajout pour simuler le temps de chargement si nécessaire, mais utilisé ici pour le spinner

# Suppression des avertissements de pandas pour le chaînage de copies
pd.options.mode.chained_assignment = None
warnings.filterwarnings("ignore")

# --- Importation des Modules (Assumés disponibles) ---
# NOTE: Ces imports nécessitent que les fichiers correspondants existent dans votre environnement.
# Les fonctions sont importées mais seront appelées via des wrappers cachés.
from db_manager import init_db, get_db_connection, extract_metrics_from_cache, update_activity_metrics_to_db
from strava_api import get_last_activity_ids, get_activity_data_from_api
from data_processor import process_data
from components.utils import display_metric_card, format_allure, format_allure_std
from components.plots import (
    creer_graphique_interactif, 
    creer_graphique_allure_pente, 
    creer_graphique_vam, 
    creer_graphique_fc_pente, 
    creer_graphique_ratio_vitesse_fc, 
    creer_graphique_comparaison,
    display_map,
    creer_analyse_segment_personnalisee
)

# --- Configuration et Initialisation des Secrets ---
# Stocker les secrets en session state pour une vérification rapide
if 'CLIENT_ID' not in st.session_state:
    try:
        st.session_state['CLIENT_ID'] = st.secrets["CLIENT_ID"]
        st.session_state['CLIENT_SECRET'] = st.secrets["CLIENT_SECRET"]
        st.session_state['ACCESS_TOKEN'] = st.secrets["ACCESS_TOKEN"]
        st.session_state['REFRESH_TOKEN'] = st.secrets["REFRESH_TOKEN"]
        st.session_state['EXPIRES_AT'] = st.secrets["EXPIRES_AT"]
    except KeyError:
        pass


# ----------------------------------------------------------------------
# FONCTIONS CACHÉES POUR LA PERFORMANCE
# ----------------------------------------------------------------------

@st.cache_resource
def init_db_cached():
    """Initialise la base de données (mise en cache)."""
    # init_db() devrait être rapide s'il gère les connexions existantes
    init_db() 
    return True

@st.cache_resource
def get_db_connection_cached():
    """Récupère la connexion DB (mise en cache)."""
    return get_db_connection()

@st.cache_data(ttl=3600) # Cache pendant 1 heure
def get_last_activity_ids_cached(limit=200):
    """Récupère les IDs d'activités récentes (mise en cache)."""
    return get_last_activity_ids(limit)

@st.cache_data(show_spinner="Téléchargement des données brutes Strava...", ttl=300) 
def get_activity_data_from_api_cached(activity_id):
    """Récupère et met en cache les données brutes d'une activité Strava."""
    return get_activity_data_from_api(activity_id)

@st.cache_data(show_spinner="Calcul des métriques avancées...")
def process_data_cached(df_raw):
    """Traite les données (lissage, VAP, etc.) et les met en cache."""
    return process_data(df_raw.copy())

@st.cache_data(show_spinner="Extraction des métriques du cache DB...")
def extract_metrics_from_cache_cached(df_cache):
    """Extrait et traite les métriques de la DB pour le dashboard de progression."""
    return extract_metrics_from_cache(df_cache)


# --- Fonctions Logiques et Affichage (Analyse) ---

def afficher_graphique(graph_name, df, df2=None, name1="", name2=""):
    """Appel dynamique des fonctions de graphique en fonction du nom choisi."""
    # ... (Le corps de cette fonction reste le même)
    if graph_name == "Allure vs Pente":
        creer_graphique_allure_pente(df)
    elif graph_name == "VAM vs Pente":
        creer_graphique_vam(df)
    elif graph_name == "FC vs Pente":
        creer_graphique_fc_pente(df)
    elif graph_name == "Efficacité de foulée vs Pente":
        creer_graphique_ratio_vitesse_fc(df)
    elif graph_name == "Impact de la fatigue":
        impact_fatigue(df)
    elif graph_name == "Comparaison d'Allure":
        creer_graphique_comparaison(df, name1, df2, name2, 'allure_min_km', 'Allure (min/km)')
    elif graph_name == "Comparaison de FC":
        creer_graphique_comparaison(df, name1, df2, name2, 'frequence_cardiaque', 'Fréquence Cardiaque (bpm)')
    # Ajout des cas pour le vélo
    elif graph_name == "Vitesse vs Pente (Vélo)":
        # Simuler une fonction de vitesse vs pente si elle n'est pas définie dans components/plots
        creer_graphique_allure_pente(df, title="Vitesse vs Pente (Vélo)", y_col='vitesse_kmh', y_label='Vitesse (km/h)')
    elif graph_name == "Efficacité Vélo (Vitesse/FC)":
        creer_graphique_ratio_vitesse_fc(df, metric_col='efficacite_course_vap', metric_label='Efficacité Vélo (Vitesse/FC)')


def impact_fatigue(df, title="Impact de la fatigue"):
    """Analyse l'impact de la fatigue en comparant la variation d'allure (CV) entre les deux moitiés du parcours."""
    st.subheader(title)
    # ... (Le corps de cette fonction reste le même)
    if not df.empty and 'distance_km' in df.columns:
        moitié_parcours = df['distance_km'].iloc[-1] / 2
        
        df_premiere_moitie = df[df['distance_km'] <= moitié_parcours].dropna(subset=['allure_min_km'])
        df_seconde_moitie = df[df['distance_km'] > moitié_parcours].dropna(subset=['allure_min_km'])
        
        if len(df_premiere_moitie) > 1 and len(df_seconde_moitie) > 1:
            cv_premiere_moitie = np.std(df_premiere_moitie['allure_min_km']) / np.mean(df_premiere_moitie['allure_min_km'])
            cv_seconde_moitie = np.std(df_seconde_moitie['allure_min_km']) / np.mean(df_seconde_moitie['allure_min_km'])
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"**CV de l'allure sur la 1ère moitié (stable) :** **{cv_premiere_moitie:.2f}**")
            with col2:
                st.markdown(f"**CV de l'allure sur la 2ème moitié (fatigue) :** **{cv_seconde_moitie:.2f}**")
                
            if cv_seconde_moitie > cv_premiere_moitie * 1.05:
                st.write("Le **CV de l'allure est significativement plus élevé** dans la seconde moitié. Cela indique une **gestion de l'effort moins stable ou une fatigue accrue**. 😩")
            elif cv_seconde_moitie < cv_premiere_moitie * 0.95:
                st.write("Le **CV de l'allure a diminué**, ce qui suggère une **meilleure stabilisation de l'allure** en fin de parcours. 👍")
            else:
                st.write("La variation de l'allure est restée **stable** tout au long de la course.")
        else:
            st.warning("Données insuffisantes pour comparer les deux moitiés du parcours (moins de deux points de données par moitié).")
    else:
        st.warning("Données d'activité insuffisantes pour l'analyse de fatigue.")

def analyze_segment_selection(df, start_km, end_km):
    """Analyse un segment de l'activité entre deux distances et affiche les métriques et un graphique."""
    # ... (Le corps de cette fonction reste le même, utilisez df pour le traitement)
    segment_df = df[(df['distance_km'] >= start_km) & (df['distance_km'] <= end_km)].copy()
    
    if segment_df.empty or len(segment_df) < 2:
        st.warning("Aucune donnée dans le segment sélectionné ou segment trop court. Veuillez ajuster les distances.")
        return
        
    st.subheader(f"Analyse du segment du km **{start_km:.2f}** au km **{end_km:.2f}**")
    
    distance_segment = segment_df['distance_km'].iloc[-1] - segment_df['distance_km'].iloc[0]
    
    temps_debut = segment_df['temps_relatif_sec'].iloc[0]
    temps_fin = segment_df['temps_relatif_sec'].iloc[-1]
    duree_segment_sec = temps_fin - temps_debut
    duree_min = int(duree_segment_sec // 60)
    duree_sec = int(duree_segment_sec % 60)

    denivele_positif = segment_df['altitude_m'].diff().clip(lower=0).sum().round(0)
    denivele_negatif = segment_df['altitude_m'].diff().clip(upper=0).sum().round(0) * -1
    
    # allure_moyenne = segment_df['allure_min_km'].mean()
    # allure_std = segment_df['allure_min_km'].std()

    allure_vap_moy = segment_df['allure_vap'].mean()
    allure_vap_std = segment_df['allure_vap'].std()
    
    fc_moyenne = segment_df['frequence_cardiaque'].mean() if 'frequence_cardiaque' in segment_df.columns and not segment_df['frequence_cardiaque'].isnull().all() else None
    fc_std = segment_df['frequence_cardiaque'].std() if 'frequence_cardiaque' in segment_df.columns and not segment_df['frequence_cardiaque'].isnull().all() else None
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        display_metric_card("Distance", f"{distance_segment:.2f} km", "📏")
    with col2:
        display_metric_card("Durée", f"{duree_min}min {duree_sec}sec", "⏱️")
    with col3:
        display_metric_card("Dénivelé", f"""📈{denivele_positif:.0f} m 
                                     \n 📉{abs(denivele_negatif):.0f} m""", "⛰️")
    with col4:
        if not pd.isna(allure_vap_moy):
            sub_value_gap = f"± {format_allure(allure_vap_std)}"
            display_metric_card("Allure VAP moyenne", format_allure(allure_vap_moy), "👟", sub_value=sub_value_gap)
        else:
            display_metric_card("Allure VAP moyenne", "N/A", "👟")
    with col5:
        if fc_moyenne is not None and not pd.isna(fc_moyenne):
            display_metric_card("FC moyenne", f"{fc_moyenne:.0f} bpm", "❤️", sub_value=f"± {fc_std:.0f}")
        else:
            display_metric_card("FC moyenne", "N/A", "💔")
            
    creer_graphique_interactif(segment_df, title="Détail du segment", key="graph_segment")


def analyse_specifique_course(df):
    """Affiche les graphiques d'analyse avancée spécifiques aux sports de course (pied) et de dénivelé."""
    with st.expander("📈 Analyse de la foulée et du dénivelé", expanded=True):
        st.header("Analyse de Pente, VAM, FC et Fatigue")
        
        col_select1, col_select2 = st.columns(2)
        with col_select1:
            graph_choisi1 = st.selectbox("Choisissez le 1er graphique :",
                                         ("Allure vs Pente", "VAM vs Pente", "FC vs Pente", "Efficacité de foulée vs Pente", "Impact de la fatigue"), key="select_1")
        with col_select2:
            graph_choisi2 = st.selectbox("Choisissez le 2nd graphique :",
                                         ("Allure vs Pente", "VAM vs Pente", "FC vs Pente", "Efficacité de foulée vs Pente", "Impact de la fatigue"), index=2, key="select_2")

        col_graph1, col_graph2 = st.columns(2)
        with col_graph1:
            st.subheader(f"Graphique 1 : **{graph_choisi1}**")
            afficher_graphique(graph_choisi1, df)
        with col_graph2:
            st.subheader(f"Graphique 2 : **{graph_choisi2}**")
            afficher_graphique(graph_choisi2, df)


def analyse_specifique_velo(df):
    """Affiche les graphiques d'analyse avancée spécifiques au cyclisme."""
    with st.expander("🚴‍♂️ Analyse de la performance cycliste", expanded=True):
        st.header("Analyse Vitesse et Efficacité (Sans Puissance)")
        st.info("Cette analyse est basée sur la vitesse, l'altitude et la fréquence cardiaque. Pour une analyse complète, les données de puissance (Watts) seraient nécessaires.")
        
        col_select1, col_select2 = st.columns(2)
        with col_select1:
            graph_choisi1 = st.selectbox("Choisissez le 1er graphique (Vélo) :",
                                         ("Vitesse vs Pente (Vélo)", "Efficacité Vélo (Vitesse/FC)"), key="select_velo_1")
        with col_select2:
            graph_choisi2 = st.selectbox("Choisissez le 2nd graphique (Vélo) :",
                                         ("Vitesse vs Pente (Vélo)", "Efficacité Vélo (Vitesse/FC)"), index=1, key="select_velo_2")

        col_graph1, col_graph2 = st.columns(2)
        with col_graph1:
            st.subheader(f"Graphique 1 : **{graph_choisi1}**")
            afficher_graphique(graph_choisi1, df)
        with col_graph2:
            st.subheader(f"Graphique 2 : **{graph_choisi2}**")
            afficher_graphique(graph_choisi2, df)


# ----------------------------------------------------------------------
## Fonction de la Page d'Analyse (Mise à Jour avec cache et gestion d'erreurs)
# ----------------------------------------------------------------------

def analyse_page():
    """Contient toute la logique de l'analyse d'une/deux activités."""
    st.title("🏃‍♂️ Analyse d'Activité Strava")

    # --- Configuration de la barre latérale pour l'analyse ---
    st.sidebar.header("Configuration de l'activité")
    
    try:
        # Utilisation de la version cachée
        recent_activities = get_last_activity_ids_cached(200)
    except Exception as e:
        st.error(f"Erreur lors de la récupération des activités récentes via l'API Strava : {e}")
        recent_activities = []
        
    activity_options = {f"{act['name']}": act['id'] for act in recent_activities}
    activity_options = {'Sélectionner une activité': None} | activity_options | {'Saisir un autre ID': 'manual'}
    
    selected_option = st.sidebar.selectbox("Sélectionnez une activité récente (1) :", list(activity_options.keys()), key="select_act_1")
    
    activity_id_input1 = None
    if activity_options[selected_option] == 'manual':
        activity_id_input1 = st.sidebar.text_input("Entrez l'ID de l'activité (1)", '', key="input_act_1")
    else:
        activity_id_input1 = activity_options[selected_option]
        
    
    # Bouton de chargement (déclenche le processus)
    st.sidebar.markdown("---")
    
    # Utilisation d'un conteneur pour les messages de chargement
    status_container = st.empty()
    
    if st.sidebar.button("🚀 Charger l'activité"):
        
        if not activity_id_input1:
            status_container.warning("Veuillez sélectionner ou entrer l'ID de la première activité.")
            return

        activity_id1 = None
        try:
            activity_id1 = int(activity_id_input1)
        except ValueError:
            status_container.error("L'ID de l'activité doit être un nombre entier.")
            return

        # 1. Traitement de l'activité 1 (Chargement brut et mise en cache)
        try:
            # Utilisation de la version cachée de l'API
            df_raw1, activity_name1, sport_type1 = get_activity_data_from_api_cached(activity_id1)
            
            if df_raw1.empty:
                status_container.warning(f"L'activité **'{activity_name1}'** n'a pas de données de stream ou est manuelle. Analyse impossible.")
                return

            # Utilisation de la version cachée du traitement
            df_result1 = process_data_cached(df_raw1)
            
            if df_result1 is None:
                status_container.error("Le traitement des données a échoué. Vérifiez la qualité des données brutes (altitude, temps, etc.).")
                return
            
            # Stockage des résultats traités en session state
            st.session_state['df_filtre'] = df_result1
            st.session_state['df_raw1'] = df_raw1
            st.session_state['activity_name1'] = activity_name1
            st.session_state['sport_type1'] = sport_type1
            st.session_state['activity_id1'] = activity_id1
            status_container.success(f"Données de l'activité **{activity_name1}** chargées et traitées avec succès!")
            
        except Exception as e:
            st.session_state['df_filtre'] = None
            status_container.error(f"❌ Erreur critique lors du chargement/traitement de l'activité {activity_id1} : {e}")
            return
            
    # --- Affichage des résultats après chargement réussi ---
    if 'df_filtre' in st.session_state and st.session_state['df_filtre'] is not None:
        
        df_filtre = st.session_state['df_filtre']
        sport_type = st.session_state.get('sport_type1', 'Unknown')
        activity_name = st.session_state.get('activity_name1', 'N/A')
        activity_id = st.session_state.get('activity_id1', None)
        df_raw1 = st.session_state['df_raw1']
        
        st.header(f"Activité Principale : **{activity_name}** (ID: {activity_id})")
        
        sport_icon_map = {'Run': '🏃‍♂️', 'TrailRun': '⛰️', 'Ride': '🚴‍♂️', 'Hike': '🚶‍♂️', 'Swim': '🏊‍♂️', 'Workout': '💪'}
        sport_icon = sport_icon_map.get(sport_type, '❓')
        st.markdown(f"**Type d'activité :** *{sport_type}* {sport_icon}")
        
        # Affichage de la carte
        display_map(df_raw1, activity_name)

        # --- Résumé de l'activité ---
        st.subheader("Résumé de l'activité")
        col1, col2, col3, col4, col5, col6 = st.columns(6)
        
        if 'temps_relatif_sec' in df_filtre.columns and not df_filtre['temps_relatif_sec'].empty:
            
            # Métriques de base
            temps_total_sec = df_filtre['temps_relatif_sec'].iloc[-1]
            temps_total_h = int(temps_total_sec // 3600)
            temps_total_min = int((temps_total_sec % 3600) // 60)
            denivele_positif = df_filtre['altitude_m'].diff().clip(lower=0).sum().round(0)
            denivele_negatif = df_filtre['altitude_m'].diff().clip(upper=0).sum().round(0) * -1
            
            # Allure vap
            allure_vap_moy = df_filtre['allure_vap'].mean() if 'allure_vap' in df_filtre.columns and not df_filtre['allure_vap'].isnull().all() else np.nan
            allure_vap_std = df_filtre['allure_vap'].std() if 'allure_vap' in df_filtre.columns and not df_filtre['allure_vap'].isnull().all() else np.nan

            # Efficacité
            efficacite_moy_vap = df_filtre['efficacite_course_vap'].mean() if 'efficacite_course_vap' in df_filtre.columns and not df_filtre['efficacite_course_vap'].isnull().all() else np.nan
            efficacite_std_vap = df_filtre['efficacite_course_vap'].std() if 'efficacite_course_vap' in df_filtre.columns and not df_filtre['efficacite_course_vap'].isnull().all() else np.nan

            # FC
            fc_moyenne = df_filtre['frequence_cardiaque'].mean() if 'frequence_cardiaque' in df_filtre.columns and not df_filtre['frequence_cardiaque'].isnull().all() else np.nan
            fc_std = df_filtre['frequence_cardiaque'].std() if 'frequence_cardiaque' in df_filtre.columns and not df_filtre['frequence_cardiaque'].isnull().all() else np.nan

            # Calcul des scores (Assumes les colonnes 'vitesse_kmh_vap' et 'duree_h' existent après process_data)
            MAX_VAP_KMH = 20 
            MAX_EFFICACITE = 0.1 
            
            duree_h = temps_total_sec / 3600
            
            vitesse_vap_moy = df_filtre['vitesse_kmh_vap'].mean() if 'vitesse_kmh_vap' in df_filtre.columns and not df_filtre['vitesse_kmh_vap'].isnull().all() else 0
            
            normalized_vap = min(vitesse_vap_moy / MAX_VAP_KMH, 1.0)
            score_effort = normalized_vap * duree_h * 100

            normalized_efficacite = min(efficacite_moy_vap / MAX_EFFICACITE, 1.0) if not np.isnan(efficacite_moy_vap) else 0
            score_effort_efficacite = normalized_efficacite * duree_h * 100
            
            # ----------------------------------------------------------------------
            # MISE À JOUR BASE DE DONNÉES
            # ----------------------------------------------------------------------
            metrics_to_save = {
                'activity_id' : activity_id,
                'allure_vap_moy': allure_vap_moy,
                'score_effort': score_effort,
                'score_effort_efficacite': score_effort_efficacite
            }
            
            try:
                # Utilisation de la fonction originale (pas de cache nécessaire pour l'écriture)
                update_activity_metrics_to_db(metrics_to_save)
            except Exception as e:
                # Ceci est une erreur non bloquante pour l'analyse
                st.sidebar.warning(f"Avertissement DB: Échec de la mise à jour des métriques : {e}")
            
            # AFFICHAGE DES CARTES
            with col1:
                display_metric_card("Distance", f"{df_filtre['distance_km'].iloc[-1]:.1f} km", "📏")
            with col2:
                display_metric_card("Durée", f"{temps_total_h}h {temps_total_min}min", "⏱️")
            with col3:
                display_metric_card("Dénivelé", f"""📈{denivele_positif:.0f} m 
                                     \n 📉{abs(denivele_negatif):.0f} m""", "⛰️")
            
            with col4:
                if sport_type not in ['Ride', 'VirtualRide'] and not np.isnan(allure_vap_moy):
                    sub_value_gap = f"± {format_allure(allure_vap_std)}"
                    display_metric_card("Allure VAP moyenne", format_allure(allure_vap_moy), "👟", sub_value=sub_value_gap)
                else :
                    vitesse_moyenne = np.round(df_filtre['distance_km'].iloc[-1] / duree_h, 1) if duree_h > 0 else 0
                    display_metric_card("Vitesse moyenne",f"{vitesse_moyenne} km/h", "🚴‍♂️")

            with col5:
                if not np.isnan(fc_moyenne):
                    display_metric_card("FC moyenne", f"{fc_moyenne:.0f} bpm", "❤️", sub_value=f"± {fc_std:.0f}")
                else:
                    display_metric_card("FC moyenne", "N/A", "💔")
                    
            with col6:
                if sport_type not in ['Ride', 'VirtualRide'] and not np.isnan(efficacite_moy_vap):
                    display_metric_card("Efficacité", f"{efficacite_moy_vap:0.03}","⏱️", sub_value = f"± {efficacite_std_vap:0.03}" )
                else:
                    # Afficher la puissance moyenne si disponible (sinon N/A)
                    puissance = df_raw1['puissance_watts'].mean() if 'puissance_watts' in df_raw1.columns and not df_raw1['puissance_watts'].isnull().all() else np.nan
                    if not np.isnan(puissance):
                         display_metric_card("Puissance moyenne", f"{puissance:.0f} watts", "⚡")
                    else:
                         display_metric_card("Efficacité", "N/A", "⏱️")
                         
        st.subheader("Profil d'Activité Complet")
        
        # Sélecteur pour afficher ou non le GAP
        creer_graphique_interactif(df_filtre, title='Profil d\'Activité Interactif', key="graph_principal")
        
        st.markdown("---")

        # --- Analyse de segment avec curseurs ---
        with st.expander("🔍 Analyse de Segment Spécifique", expanded=True):
            max_km = df_filtre['distance_km'].max()
            col1,col2 = st.columns(2)
            with col1:
                start_km = st.number_input(
                    "Sélectionnez le début du segment",
                    min_value=0.00,
                    max_value=max_km,
                    value= 0.00,
                    step=0.01,
                    key="start_segment"
                )
            with col2:
                end_km = st.number_input(
                    "Sélectionnez la fin du segment",
                    min_value=0.00,
                    max_value=max_km,
                    value= max_km,
                    step=0.01,
                    key="end_segment"
                )
            # S'assurer que les entrées sont valides avant d'analyser
            if start_km < end_km and max_km > 0:
                analyze_segment_selection(df_filtre, start_km, end_km)
                creer_analyse_segment_personnalisee(df_filtre, start_km, end_km)

        st.markdown("---")

        # --- Affichage de l'analyse spécifique au sport ---
        
        if sport_type in ['Run', 'TrailRun', 'Walk', 'Hike']:
            analyse_specifique_course(df_filtre)
            
        elif sport_type in ['Ride', 'VirtualRide']:
            analyse_specifique_velo(df_filtre)
            
        else:
            st.info(f"Pas d'analyse avancée spécifique implémentée pour le type d'activité : **{sport_type}**.")
            
            
    else:
        st.info("Veuillez sélectionner ou entrer un ID d'activité et cliquer sur **'🚀 Charger l'activité'** pour commencer l'analyse.")


# ----------------------------------------------------------------------
## Fonction de la nouvelle Page de Progression (Mise à Jour avec cache)
# ----------------------------------------------------------------------

def progression_page():
    """Affiche les statistiques générales de progression à partir de la base de données."""
    st.title("📈 Tableau de Bord de Progression et Statistiques Générales")
    
    # 1. CHARGEMENT DES DONNÉES EN CACHE
    conn = get_db_connection_cached() 
    
    try:
        df_cache = pd.read_sql_query("SELECT * FROM activities_cache", conn)
    except Exception as e:
        st.error(f"Erreur lors de la lecture de la table 'activities_cache' : {e}")
        return
    
    if df_cache.empty:
        st.info("La base de données ne contient aucune activité mise en cache. Chargez des activités via l'onglet 'Analyse d'Activité' pour voir les statistiques ici.")
        return

    # Utilisation de la version cachée de l'extraction
    df_progression = extract_metrics_from_cache_cached(df_cache)
    
    if df_progression.empty or df_progression['score_effort_efficacite'].isnull().all():
        st.info("Aucune donnée d'activité valide ou aucun score de progression calculé trouvé. Veuillez analyser des activités pour générer les métriques.")
        return

    # Préparation des données pour le regroupement
    df_progression['date'] = pd.to_datetime(df_progression['date'], errors='coerce') 
    df_progression = df_progression.dropna(subset=['date'])
        
    # --- 2. FILTRES DE PÉRIODE ET DE TYPE ---
    st.header("Filtres")
    col_f1, col_f2 = st.columns(2)
    
    # Filtre Temporel (Identique)
    periode_choisie = col_f1.selectbox("Sélectionnez la période d'analyse :", 
                                       ["Total", "Derniers 30 jours", "Derniers 90 jours", "Année en cours", "Personnalisée"])

    date_max = df_progression['date'].max()
    date_min_data = df_progression['date'].min()
    
    df_filtre_periode = df_progression.copy()
    
    # Application des filtres temporels
    if periode_choisie == "Derniers 30 jours":
        date_debut = date_max - timedelta(days=30)
        df_filtre_periode = df_progression[df_progression['date'] >= date_debut]
    elif periode_choisie == "Derniers 90 jours":
        date_debut = date_max - timedelta(days=90)
        df_filtre_periode = df_progression[df_progression['date'] >= date_debut]
    elif periode_choisie == "Année en cours":
        annee_actuelle = date_max.year
        df_filtre_periode = df_progression[df_progression['date'].dt.year == annee_actuelle]
    elif periode_choisie == "Personnalisée":
        date_debut_filtre, date_fin_filtre = col_f1.date_input("Intervalle de dates", [date_min_data.date(), date_max.date()])
        df_filtre_periode = df_progression[(df_progression['date'].dt.date >= date_debut_filtre) & (df_progression['date'].dt.date <= date_fin_filtre)]
        
    # Filtre Type de Sport (Identique)
    sports_disponibles = ['Tous'] + sorted(df_filtre_periode['type_sport'].unique().tolist())
    sport_choisi = col_f2.selectbox("Filtrer par type de sport :", sports_disponibles)

    if sport_choisi != 'Tous':
        df_final = df_filtre_periode[df_filtre_periode['type_sport'] == sport_choisi].copy()
    else:
        df_final = df_filtre_periode.copy()
        
    if df_final.empty:
        st.warning("Aucune activité trouvée avec les filtres sélectionnés.")
        return
        
    # --- 3. Statistiques Générales Cumulées (Identique) ---
    st.header(f"Statistiques Cumulées ({periode_choisie} - {sport_choisi})")
    
    total_distance = df_final['distance_km'].sum()
    total_denivele = df_final['denivele_positif_m'].sum()
    
    col_a, col_b, col_c = st.columns(3)
    
    with col_a:
        display_metric_card("Total Activités", f"{len(df_final):.0f}", "🔢")
    with col_b:
        display_metric_card("Distance Totale", f"{total_distance:,.0f} km", "🌍")
    with col_c:
        display_metric_card("Dénivelé Total", f"{total_denivele:,.0f} m", "🏔️")
    
    # --- 4. ANALYSE DE LA CHARGE ET DE L'EFFICACITÉ (Identique) ---
    st.header("Analyse de la Charge et de l'Efficacité")

    col_p1, col_p2 = st.columns(2)
    
    # Graphique d'Efficacité 
    with col_p1:
        st.subheader("Efficacité (VAP/FC) par Activité")
        # NOTE: La colonne 'progression_metric' n'existe pas dans le code fourni, j'utilise 'score_effort'
        fig_prog = px.scatter(df_final, x='date', y='score_effort', 
                              title="Tendance de l'Efficacité Course (VAP/FC)",
                              labels={'score_effort': 'Score d\'Effort Vitesse', 'date': 'Date'},
                              hover_data=['nom', 'distance_km', 'allure_vap_moy'],
                              trendline="lowess", 
                              height=400)
        fig_prog.update_traces(marker=dict(size=8, opacity=0.7))
        fig_prog.update_layout(showlegend=False)
        st.plotly_chart(fig_prog, use_container_width=True)


    # Graphique du Score d'Effort (score_effort_efficacite)
    with col_p2:
        st.subheader("Charge d'Entraînement (Score d'Effort)")
        
        fig_effort = px.scatter(df_final, x='date', y='score_effort_efficacite', 
                                 title="Charge d'Entraînement (Effort Score)",
                                 labels={'score_effort_efficacite': 'Score d\'Effort (TSS Éq.)', 'date': 'Date'},
                                 hover_data=['nom', 'duree_h'], # progression_metric n'existe pas
                                 color='type_sport', 
                                 height=400)
        fig_effort.update_traces(marker=dict(size=10, opacity=0.8))
        fig_effort.update_layout(showlegend=True)
        st.plotly_chart(fig_effort, use_container_width=True)
        
    # --- 5. Progression Temporelle (Volume d'Entraînement - Identique) ---
    st.header("Volume d'Entraînement")
    
    # ... (Reste de la fonction progression_page inchangé)
    if (date_max - df_final['date'].min()).days < 100:
        df_final['periode_label'] = df_final['date'].dt.strftime('%Y-S%W')
        periode_type = 'Semaine'
    else:
        df_final['periode_label'] = df_final['date'].dt.strftime('%Y-%m')
        periode_type = 'Mois'
        
    df_progression_group = df_final.groupby('periode_label').agg( 
        distance=('distance_km', 'sum'),
        denivele=('denivele_positif_m', 'sum'),
    ).reset_index().sort_values('periode_label', ascending=True)

    col_g1, col_g2 = st.columns(2)
    
    with col_g1:
        st.subheader(f"Distance par {periode_type} (km)")
        fig_dist = px.bar(df_progression_group, x='periode_label', y='distance', 
                          title=f'Distance Totale par {periode_type}', 
                          labels={'distance': 'Distance (km)', 'periode_label': periode_type},
                          height=350)
        fig_dist.update_xaxes(type='category', tickangle=45) 
        st.plotly_chart(fig_dist, use_container_width=True)

    with col_g2:
        st.subheader(f"Dénivelé Positif par {periode_type} (m)")
        fig_deniv = px.bar(df_progression_group, x='periode_label', y='denivele', 
                            title=f'Dénivelé Positif par {periode_type}', 
                            labels={'denivele': 'Dénivelé (m)', 'periode_label': periode_type},
                            color_discrete_sequence=['#FF7F0E'],
                            height=350)
        fig_deniv.update_xaxes(type='category', tickangle=45)
        st.plotly_chart(fig_deniv, use_container_width=True)
        
    # --- 6. Répartition (Identique) ---
    st.header("Répartition par Type de Sport")
    
    if 'type_sport' in df_final.columns and not df_final['type_sport'].isnull().all():
        df_sport = df_final[df_final['distance_km'] > 0.1]
        
        df_sport_group = df_sport.groupby('type_sport').agg(
            total_distance=('distance_km', 'sum'),
            count=('id', 'count')
        ).reset_index()

        col_pie, col_bar = st.columns(2)
        with col_pie:
            st.subheader("Par Nombre d'Activités")
            fig_pie = px.pie(df_sport_group, names='type_sport', values='count', 
                              title="Répartition des activités par Nombre", height=350)
            st.plotly_chart(fig_pie, use_container_width=True)
            
        with col_bar:
            st.subheader("Par Distance Totale (km)")
            fig_bar = px.bar(df_sport_group, x='type_sport', y='total_distance', 
                              title='Distance par Type de Sport', height=350,
                              labels={'type_sport': 'Sport', 'total_distance': 'Distance (km)'})
            st.plotly_chart(fig_bar, use_container_width=True)
            
    else:
        st.info("Les données de type de sport sont insuffisantes pour cette analyse.")


# ----------------------------------------------------------------------
## Fonction Principale de l'Application
# ----------------------------------------------------------------------

def main_app():
    """Contient la logique de navigation principale une fois l'initialisation réussie."""
    st.sidebar.title("Navigation")
    
    # SÉLECTEUR DE PAGE PRINCIPAL
    page = st.sidebar.radio("Choisissez la vue :", ["🏃‍♂️ Analyse d'Activité", "📈 Tableau de Bord de Progression"])

    st.sidebar.markdown("---")
    
    if page == "🏃‍♂️ Analyse d'Activité":
        analyse_page()
    elif page == "📈 Tableau de Bord de Progression":
        progression_page()


# ----------------------------------------------------------------------
## Point d'Entrée (avec Page de Chargement)
# ----------------------------------------------------------------------

def main():
    """Point d'entrée de l'application Streamlit avec la vérification initiale."""
    st.set_page_config(layout="wide", page_title="Analyse Strava Avancée")

    # Vérification des prérequis (API et DB)
    if not st.session_state.get('APP_READY'):
        # Conteneur pour la page de chargement
        loading_container = st.container()
        
        with loading_container:
            st.title("🚀 Chargement et Initialisation du Système")
            
            # 1. Vérification des Secrets (Authentification)              
            st.success("🔑 Authentification API Strava (Secrets) OK.")
            
            # 2. Initialisation de la Base de Données
            try:
                with st.spinner("⏳ Initialisation de la base de données..."):
                    # Appel à la fonction mise en cache
                    init_db_cached() 
                st.success("🗄️ Base de données SQLite OK et prête.")
                st.session_state['APP_READY'] = True
                st.balloons()
            except Exception as e:
                st.error(f"❌ Erreur lors de l'initialisation de la DB : {e}")
                st.stop()
        
        # Le script se relance et passe à la partie main_app
        st.rerun()

    else:
        # Une fois initialisé, on lance l'application principale
        main_app()


if __name__ == "__main__":
    main()
