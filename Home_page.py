# Home_page.py

from datetime import datetime, timezone

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st

from strava_api import get_activity_data_from_api, get_last_activity_ids
from utils.data_processing import (
    allure_format,
    calculate_tss,
    process_activity,
    time_formatter,
)
from utils.db_manager import get_db_connection, init_db, sql_df
from utils.plotting import agg_sql_df_period

st.set_page_config(layout='wide')

st.markdown(f"""
    <style>
    /* Titre principal en Orange Strava */
    h1 {{
        color: #FC4C02;
    }}

    /* Bouton 'Charger l'activité' personnalisé */
    div.stButton > button:first-child {{
        background-color: #FC4C02;
        color: white;
        border: none;
        border-radius: 8px;
        font-weight: bold;
    }}

    /* Fond légèrement grisé pour les containers */
    [data-testid="stVerticalBlockBorderWrapper"] {{
        background-color: #fcfcfc;
    }}

    /* Style pour les métriques */
    [data-testid="stMetricValue"] {{
        color: #FC4C02;
    }}

    .stProgress > div > div > div > div {{
        background-color: #FC4C02;
    }}
    </style>
    """, unsafe_allow_html=True)

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

activity_id_input = None
if activity_options[selected_option] == 'manual':
    activity_id_input = st.sidebar.text_input("Entrez l'ID de l'activité (1)", '', key="input_act")
else:
    activity_id_input = activity_options[selected_option]

st.sidebar.markdown("---")

# Utilisation d'un conteneur pour les messages de chargement
status_container = st.empty()

if st.sidebar.button("🚀 Charger l'activité"):

    if not activity_id_input:
        status_container.warning("Veuillez sélectionner ou entrer l'ID de la première activité.")

    activity_id = None
    try:
        activity_id = int(activity_id_input)
    except ValueError:
        status_container.error("L'ID de l'activité doit être un nombre entier.")

    # 1. Traitement de l'activité 1 (Chargement brut et mise en cache)
    try:
        # Utilisation de la version cachée de l'API
        df_raw, activity_name, sport_type, activity_date = get_activity_data_from_api_cached(activity_id)

        df_raw, km_effort_itra, km_effort_611, temps_total_formatte, ratio_denivele_distance = process_activity(df_raw)

        if df_raw.empty:
            status_container.warning(f"L'activité **'{activity_name}'** n'a pas de données de stream ou est manuelle. Analyse impossible.")

        # Stockage des résultats traités en session state
        st.session_state['df_raw'] = df_raw
        st.session_state['activity_name'] = activity_name
        st.session_state['sport_type'] = sport_type
        st.session_state['activity_id'] = activity_id
        st.session_state['activity_date'] = activity_date
        status_container.success(f"Données de l'activité **{activity_name}** chargées et traitées avec succès!")

    except Exception as e:
        status_container.error(f"❌ Erreur critique lors du chargement/traitement de l'activité {activity_id} : {e}")

if 'df_raw' in st.session_state:
    df_raw = st.session_state['df_raw']
    activity_name = st.session_state['activity_name']
    sport_type = st.session_state['sport_type']
    activity_date = st.session_state['activity_date']
    st.header(f"Activité Principale : **{activity_name}**")

    sport_icon_map = {'Run': '🏃‍♂️', 'TrailRun': '⛰️', 'Ride': '🚴‍♂️', 'Hike': '🚶‍♂️'}
    sport_icon = sport_icon_map.get(sport_type, '❓')
    st.markdown(f"**Type d'activité :** *{sport_type}* {sport_icon}")

    with st.container(border=True):
        st.subheader("Indicateurs généraux de l'activité")

        col1, col2, col3, col4, col5, col6 = st.columns(6)
        with col1:
            st.metric("📏 Distance", value=f"{np.round(df_raw['distance_km'].max(),1)} km")
        with col2:
            st.metric("⛰️ Dénivelé +", value=f"{int(df_raw['d_pos_cum'].max())}m")
        with col3:
            st.metric("⏱️ Temps", value=f"{time_formatter(df_raw['temps_min'].max())}")
        with col4:
            st.metric("🏃 Allure Moyenne", value=f"{allure_format(df_raw['allure_min_km'].mean())}")
        with col5:
            st.metric("📈 VAP Moyenne", value=f"{allure_format(df_raw['vap_allure'].mean())}", help="Allure Ajustée à la Pente")
        with col6:
            st.metric("❤️ FC Moyenne", value=f"{int(df_raw['frequence_cardiaque'].mean())} bpm")

    with st.container(border=True):
        st.subheader("Charge d'entraînement")
        c1, c2 = st.columns([1,3])
        with c1:
            tss = calculate_tss(df=df_raw, FTP=5)
            st.metric(label="Training Stress Score", value=tss)
        with c2:
            level = "Faible" if tss < 50 else "Modérée" if tss < 150 else "Elevée" if tss < 300 else "Intense"
            st.progress(min(tss/600, 1.0), text=f"Charge d'entrainement: {level}")

    df_sql = sql_df()
    df_sql['year'] = pd.to_datetime(df_sql['activity_start_date']).dt.year
    df_sql['month'] = pd.to_datetime(df_sql['activity_start_date']).dt.month
    df_sql['week'] = pd.to_datetime(df_sql['activity_start_date']).dt.isocalendar().week
    df_sql['delta_day'] = datetime.now(timezone.utc) - pd.to_datetime(df_sql['activity_start_date'])
    df_sql['delta_day'] = df_sql['delta_day'].dt.days

    list_of_sport_type = df_sql['sport_type'].unique().tolist()
    sport_type_options = st.multiselect("Sélectionne le ou les sports que tu souhaites voir l'évolution?",
                                        options=list_of_sport_type, default=['Hike','Run','TrailRun'])



    with st.container(border=True):
        st.subheader("📊 Statistiques hebdomadaires")
        col_dist_week, col_d_pos_week = st.columns(2)
        with col_dist_week:
            fig = agg_sql_df_period(df_sql, period='week', feature='total_distance_km', sport_type_list=sport_type_options)
            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
        with col_d_pos_week:
            fig = agg_sql_df_period(df_sql, period='week', feature='total_elevation_gain_m', sport_type_list=sport_type_options)
            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

    with st.container(border=True):
        st.subheader("📊 Statistiques mensuelles")
        col_dist_month, col_d_pos_month = st.columns(2)
        with col_dist_month:
            fig = agg_sql_df_period(df_sql, period='month', feature='total_distance_km', sport_type_list=sport_type_options)
            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
        with col_d_pos_month:
            fig = agg_sql_df_period(df_sql, period='month', feature='total_elevation_gain_m', sport_type_list=sport_type_options)
            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

else:
    st.info("Veuillez sélectionner ou entrer un ID d'activité et cliquer sur **'🚀 Charger l'activité'** pour  \
            commencer l'analyse.")
