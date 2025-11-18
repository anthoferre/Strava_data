# Home_page.py

import streamlit as st
from datetime import datetime, timezone
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import matplotlib.colors as mcolors
from db_manager import init_db, get_db_connection, init_db, sql_df
from strava_api import get_last_activity_ids, get_activity_data_from_api

from utils.data_processing import process_activity, time_formatter, allure_format

st.set_page_config(layout='wide')
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
    

# Bouton de chargement (déclenche le processus)
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
    
    
    ####################################################
    st.subheader("Indicateurs généraux de l'activité")

    col1, col2, col3, col4, col5, col6 = st.columns(6)
    with col1:
        st.metric("Distance", value=f"{np.round(df_raw['distance_km'].max(),1)} km", border=True)
    with col2:
        st.metric("Dénivelé +", value=f"{int(df_raw['d_pos_cum'].max())}m", border=True)
    with col3:
        st.metric("Temps", value=f"{time_formatter(df_raw['temps_min'].max())}", border=True)
    with col4:
        st.metric("Allure Moyenne", value=f"{allure_format(df_raw['allure_min_km'].mean())}", border=True)
    with col5:
        st.metric("VAP Moyenne", value=f"{allure_format(df_raw['vap_allure'].mean())}", border=True, help="Allure Ajustée à la Pente")
    with col6:
        st.metric("FC Moyenne", value=f"{int(df_raw['frequence_cardiaque'].mean())} bpm", border=True)

    st.divider()
    st.subheader("Evolution de la charge d'entraînement")

    df_sql = sql_df()
    df_sql['year'] = pd.to_datetime(df_sql['activity_start_date']).dt.year
    df_sql['month'] = pd.to_datetime(df_sql['activity_start_date']).dt.month
    df_sql['week'] = pd.to_datetime(df_sql['activity_start_date']).dt.isocalendar().week
    df_sql['delta_day'] = datetime.now(timezone.utc) - pd.to_datetime(df_sql['activity_start_date'])
    df_sql['delta_day'] = df_sql['delta_day'].dt.days

    def agg_sql_df_period(df, period, feature, sport_type_list):

        df_sport_type = df[df['sport_type'].isin(sport_type_list)]

        df_agg = df_sport_type.groupby(by=period)[feature].sum().reset_index()

        cmap = plt.cm.viridis # Vous pouvez choisir 'viridis', 'plasma', 'magma', etc.

        # 2. Normaliser les données de distance pour les faire correspondre aux couleurs
        norm = mcolors.Normalize(
            vmin=df_agg[feature].min(),
            vmax=df_agg[feature].max()
        )
        scalar_mappable = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
        scalar_mappable.set_array(df_agg[feature])

        st.caption(f"Evolution de {feature} par {period}")

        fig, ax = plt.subplots()
        bars = sns.barplot(data=df_agg, x=period, y=feature, ax=ax)
        for i, bar in enumerate(bars.patches):
            # Récupérer la valeur de distance correspondante
            distance_value = df_agg[feature].iloc[i]
            # Appliquer la couleur basée sur la normalisation
            bar.set_color(scalar_mappable.to_rgba(distance_value))
        dict_month = {1: 'Janv.', 2: 'Févr.', 3: 'Mars', 4: 'Avril', 5: 'Mai', 6: 'Juin', 7: 'Juil.', 8: 'Août', 
                     9: 'Sept.', 10: 'Oct.', 11: 'Nov.', 12: 'Déc.'}
        month_number = df['month'].unique().tolist()
        month_labels = [dict_month[m] for m in month_number]
        
        if period == 'week':
            ax.xaxis.set_major_locator(MultipleLocator(3))
        elif period =='month':
            ax.set_xticklabels(month_labels, rotation=45, ha='right')


        # 5. Ajouter la barre de couleur (Colorbar)
        cbar = fig.colorbar(scalar_mappable, ax=ax, orientation='vertical', pad=0.03)
        return fig
    
    list_of_sport_type = df_sql['sport_type'].unique().tolist()
    sport_type_options = st.multiselect("Sélectionne le ou les sports que tu souhaites voir l'évolution?", 
                                        options=list_of_sport_type, default=['Hike','Run','TrailRun'])
    
    
    col_dist_week, col_d_pos_week = st.columns(2)
    with col_dist_week:
        fig = agg_sql_df_period(df_sql, period='week', feature='total_distance_km', sport_type_list=sport_type_options)
        st.pyplot(fig)
        plt.close(fig)
    with col_d_pos_week:
        fig = agg_sql_df_period(df_sql, period='week', feature='total_elevation_gain_m', sport_type_list=sport_type_options)
        st.pyplot(fig)
        plt.close(fig)

    col_dist_month, col_d_pos_month = st.columns(2)
    with col_dist_month:
        fig = agg_sql_df_period(df_sql, period='month', feature='total_distance_km', sport_type_list=sport_type_options)
        st.pyplot(fig)
        plt.close(fig)
    with col_d_pos_month:
        fig = agg_sql_df_period(df_sql, period='month', feature='total_elevation_gain_m', sport_type_list=sport_type_options)
        st.pyplot(fig)
        plt.close(fig)

        
else:
    st.info("Veuillez sélectionner ou entrer un ID d'activité et cliquer sur **'🚀 Charger l'activité'** pour commencer l'analyse.")