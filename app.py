import streamlit as st
import pandas as pd
import warnings
import plotly.express as px
import numpy as np
from datetime import datetime, timedelta
import calendar

# Suppression des avertissements de pandas pour le chaînage de copies
pd.options.mode.chained_assignment = None
warnings.filterwarnings("ignore")

# --- Importation des Modules (Assumés disponibles) ---
# NOTE: Ces imports nécessitent que les fichiers correspondants existent dans votre environnement.
from db_manager import init_db, get_db_connection, extract_metrics_from_cache
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

# Utiliser les secrets Streamlit pour l'authentification et les stocker en session
if 'CLIENT_ID' not in st.session_state:
    try:
        # NOTE: Les secrets doivent être configurés dans .streamlit/secrets.toml
        st.session_state['CLIENT_ID'] = st.secrets["CLIENT_ID"]
        st.session_state['CLIENT_SECRET'] = st.secrets["CLIENT_SECRET"]
        st.session_state['ACCESS_TOKEN'] = st.secrets["ACCESS_TOKEN"]
        st.session_state['REFRESH_TOKEN'] = st.secrets["REFRESH_TOKEN"]
        st.session_state['EXPIRES_AT'] = st.secrets["EXPIRES_AT"]
    except KeyError:
        # Permet de lancer l'app même sans secrets pour tester le code/la DB
        pass 


# --- Fonctions Logiques et Affichage (Analyse) ---

# Fonction utilitaire pour appeler le bon graphique (nécessaire pour les selectbox)
def afficher_graphique(graph_name, df, df2=None, name1="", name2=""):
    """Appel dynamique des fonctions de graphique en fonction du nom choisi."""
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


def impact_fatigue(df, title="Impact de la fatigue"):
    """Analyse l'impact de la fatigue en comparant la variation d'allure (CV) entre les deux moitiés du parcours."""
    st.subheader(title)
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
    
    allure_moyenne = segment_df['allure_min_km'].mean()
    allure_std = segment_df['allure_min_km'].std()
    
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
        if not pd.isna(allure_moyenne):
            sub_value_gap = f"± {format_allure(allure_std)}"
            display_metric_card("Allure moyenne", format_allure(allure_moyenne), "👟", sub_value=sub_value_gap)
        else:
            display_metric_card("Allure moyenne", "N/A", "👟")
    with col5:
        if fc_moyenne is not None and not pd.isna(fc_moyenne):
            display_metric_card("FC moyenne", f"{fc_moyenne:.0f} bpm", "❤️", sub_value=f"± {fc_std:.0f}")
        else:
            display_metric_card("FC moyenne", "N/A", "💔")
            
    creer_graphique_interactif(segment_df, title="Détail du segment", key="graph_segment")


# NOUVELLE FONCTION : Analyse spécifique pour Course, Trail et Marche
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


# NOUVELLE FONCTION : Analyse spécifique pour le Vélo
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
## Fonction de la Page d'Analyse (Mise à Jour)
# ----------------------------------------------------------------------

def analyse_page():
    """Contient toute la logique de l'analyse d'une/deux activités."""
    st.title("🏃‍♂️ Analyse d'Activité Strava")

    # --- Configuration de la barre latérale pour l'analyse (Inchangée) ---
    st.sidebar.header("Configuration de l'activité")
    
    recent_activities = get_last_activity_ids(200)
    activity_options = {f"{act['name']}": act['id'] for act in recent_activities}
    activity_options = {'Sélectionner une activité': None} | activity_options | {'Saisir un autre ID': 'manual'}
    
    selected_option = st.sidebar.selectbox("Sélectionnez une activité récente (1) :", list(activity_options.keys()), key="select_act_1")
    
    activity_id_input1 = None
    if activity_options[selected_option] == 'manual':
        activity_id_input1 = st.sidebar.text_input("Entrez l'ID de l'activité (1)", '', key="input_act_1")
    else:
        activity_id_input1 = activity_options[selected_option]
        
    st.sidebar.markdown("---")
    st.sidebar.subheader("Optionnel : 2ème Activité (Comparaison)")
    
    activity_options2 = {f"{act['name']} ({act['id']})": act['id'] for act in recent_activities if act['id'] != activity_id_input1}
    activity_options2 = {'Ne pas comparer': None, 'Saisir un autre ID': 'manual'} | activity_options2
    
    selected_option2 = st.sidebar.selectbox("Sélectionnez une activité récente (2) :", list(activity_options2.keys()), key="select_act_2")
    
    activity_id_input2 = None
    if selected_option2 == 'Saisir un autre ID':
        activity_id_input2 = st.sidebar.text_input("Entrez l'ID de l'activité (2)", '', key="input_act_2")
    elif selected_option2 != 'Ne pas comparer' and activity_options2[selected_option2] is not None:
        activity_id_input2 = activity_options2[selected_option2]

    st.sidebar.markdown("---")   
 
    
    # Bouton de chargement (déclenche le processus)
    st.sidebar.markdown("---")
    if st.sidebar.button("🚀 Charger / Comparer les activités"):
        
        if not activity_id_input1:
             st.warning("Veuillez sélectionner ou entrer l'ID de la première activité.")
             return
        
        # 1. Traitement de l'activité 1
        try:
            activity_id1 = int(activity_id_input1)
            with st.spinner(f"Chargement de l'activité **{activity_id1}**..."):
                df_raw1, activity_name1, sport_type1 = get_activity_data_from_api(activity_id1)
                st.session_state['df_raw1'] = df_raw1
                st.session_state['activity_name1'] = activity_name1
                st.session_state['sport_type1'] = sport_type1
        except ValueError:
            st.error("L'ID de la première activité doit être un nombre entier.")
            return

        # 2. Traitement de l'activité 2 (si fournie)
        st.session_state['df_raw2'] = None
        st.session_state['activity_name2'] = None
        
        if activity_id_input2 and activity_id_input2 != 'manual':
            try:
                activity_id2 = int(activity_id_input2)
                with st.spinner(f"Chargement de l'activité **{activity_id2}** pour comparaison..."):
                    df_raw2, activity_name2, sport_type2 = get_activity_data_from_api(activity_id2)
                    st.session_state['df_raw2'] = df_raw2
                    st.session_state['activity_name2'] = activity_name2
                    st.session_state['sport_type2'] = sport_type2
            except ValueError:
                st.error("L'ID de la deuxième activité doit être un nombre entier.")
                return
        
        st.success("Chargement terminé. Analyse des données en cours...")
        st.rerun()

    # --- Affichage des résultats ---
    if 'df_raw1' in st.session_state and st.session_state['df_raw1'] is not None:
        
        if st.session_state['df_raw1'].empty:
            st.warning(f"L'activité **'{st.session_state.get('activity_name1', 'N/A')}'** n'a pas de données de stream ou est manuelle. Elle ne peut pas être analysée.")
            return

        df_result1 = process_data(st.session_state['df_raw1'].copy())
        if df_result1 is None:
            st.warning("Le traitement des données de l'activité 1 a échoué. Veuillez vérifier les données de l'activité ou les paramètres de lissage.")
            return

        df_filtre = df_result1.copy()
        sport_type = st.session_state.get('sport_type1', 'Unknown')
        
        st.header(f"Activité Principale : **{st.session_state['activity_name1']}**")
        
        sport_icon_map = {'Run': '🏃‍♂️', 'TrailRun': '⛰️', 'Ride': '🚴‍♂️', 'Hike': '🚶‍♂️', 'Swim': '🏊‍♂️', 'Workout': '💪'}
        sport_icon = sport_icon_map.get(sport_type, '❓')
        st.markdown(f"**Type d'activité :** *{sport_type}* {sport_icon}")
        
        # Affichage de la carte
        display_map(st.session_state['df_raw1'], st.session_state['activity_name1'])

        # --- Résumé de l'activité (Inchangé) ---
        st.subheader("Résumé de l'activité")
        col1, col2, col3, col4, col5 = st.columns(5)
        
        if 'temps_relatif_sec' in df_filtre.columns and not df_filtre['temps_relatif_sec'].empty:
            
            # Métriques de base
            temps_total_sec = df_filtre['temps_relatif_sec'].iloc[-1]
            temps_total_h = int(temps_total_sec // 3600)
            temps_total_min = int((temps_total_sec % 3600) // 60)
            denivele_positif = df_filtre['altitude_m'].diff().clip(lower=0).sum().round(0)
            denivele_negatif = df_filtre['altitude_m'].diff().clip(upper=0).sum().round(0) * -1
            
            # Allure moyenne (brute et GAP)
            allure_moyenne = df_filtre['allure_min_km'].mean()
            allure_std = df_filtre['allure_min_km'].std()
            
            with col1:
                display_metric_card("Distance", f"{df_filtre['distance_km'].iloc[-1]:.1f} km", "📏")
            with col2:
                display_metric_card("Durée", f"{temps_total_h}h {temps_total_min}min", "⏱️")
            with col3:
                display_metric_card("Dénivelé", f"""📈{denivele_positif:.0f} m 
                                     \n 📉{abs(denivele_negatif):.0f} m""", "⛰️")
            with col4:
                if sport_type != 'Ride':
                    # Affichage de l'allure moyenne et du GAP juste en dessous
                    sub_value_gap = f"± {format_allure(allure_std)}"
                    display_metric_card("Allure moyenne", format_allure(allure_moyenne), "👟", sub_value=sub_value_gap)
                else :
                    vitesse_moyenne = np.round(df_filtre['distance_km'].iloc[-1] / temps_total_h,1)
                    display_metric_card("Vitesse moyenne",f"{vitesse_moyenne} km/h", "🚴‍♂️")

            if sport_type != 'Ride':
                if 'frequence_cardiaque' in df_filtre.columns and not df_filtre['frequence_cardiaque'].isnull().all():
                    fc_moyenne = df_filtre['frequence_cardiaque'].mean()
                    fc_std = df_filtre['frequence_cardiaque'].std()
                    with col5:
                        display_metric_card("FC moyenne", f"{fc_moyenne:.0f} bpm", "❤️", sub_value=f"± {fc_std:.0f}")
                else:
                    with col5:
                        display_metric_card("FC moyenne", "N/A", "💔")

            else:
                with col5:
                    display_metric_card("Puissance moyenne", f"{np.mean(st.session_state['df_raw1']['puissance_watts']):.0f} watts", "❤️")


        st.subheader("Profil d'Activité Complet")
        
        # Sélecteur pour afficher ou non le GAP
        creer_graphique_interactif(df_filtre, title='Profil d\'Activité Interactif', key="graph_principal")
        
        st.markdown("---")

        # --- Analyse de segment avec curseurs (Inchangée) ---
        with st.expander("🔍 Analyse de Segment Spécifique", expanded=False):
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
            analyze_segment_selection(df_filtre, start_km, end_km)
            creer_analyse_segment_personnalisee(df_filtre, start_km, end_km)

        st.markdown("---")
        
        # --- Logique de comparaison (Inchangée) ---
        if 'df_raw2' in st.session_state and st.session_state['df_raw2'] is not None and not st.session_state['df_raw2'].empty:
            
            df_result2 = process_data(st.session_state['df_raw2'].copy())
            
            if df_result2 is not None:
                with st.expander("📊 Comparaison d'Activités", expanded=True):
                    st.header("Comparaison : Allure et FC")
                    st.info(f"Comparaison entre **{st.session_state['activity_name1']}** ({sport_type}) et **{st.session_state['activity_name2']}** ({st.session_state.get('sport_type2', 'N/A')})")
                    
                    comparaison_type = st.selectbox("Type de comparaison :", ("Comparaison d'Allure", "Comparaison de FC"), key="comp_type_select")
                    
                    afficher_graphique(comparaison_type, df_filtre, df_result2, st.session_state['activity_name1'], st.session_state['activity_name2'])

        st.markdown("---")

        # --- NOUVEAU: Affichage de l'analyse spécifique au sport ---
        
        # Sports de course à pied, trail, marche
        if sport_type in ['Run', 'TrailRun', 'Walk', 'Hike']:
            analyse_specifique_course(df_filtre)
            
        # Sports de vélo
        elif sport_type in ['Ride', 'VirtualRide']:
            analyse_specifique_velo(df_filtre)
            
        else:
            # Autres sports (Natation, Workout, etc.)
            st.info(f"Pas d'analyse avancée spécifique implémentée pour le type d'activité : **{sport_type}**.")
            
            
    else:
        st.info("Veuillez sélectionner ou entrer un ID d'activité et cliquer sur **'🚀 Charger / Comparer les activités'** pour commencer l'analyse.")


# ----------------------------------------------------------------------
## Fonction de la nouvelle Page de Progression (UX améliorée - Inchangée)
# ----------------------------------------------------------------------

def progression_page():
    """Affiche les statistiques générales de progression à partir de la base de données."""
    st.title("📈 Tableau de Bord de Progression et Statistiques Générales")
    
    conn = get_db_connection() 
    df_cache = pd.read_sql_query("SELECT * FROM activities_cache", conn)
    
    if df_cache.empty:
        st.info("La base de données ne contient aucune activité mise en cache. Chargez des activités via l'onglet 'Analyse d'Activité' pour voir les statistiques ici.")
        return

    df_progression = extract_metrics_from_cache(df_cache)
    
    if df_progression.empty:
        st.info("Aucune donnée d'activité valide n'a pu être extraite du cache. Veuillez vous assurer que les activités analysées ne sont pas manuelles.")
        return

    # Préparation des données pour le regroupement
    df_progression['date'] = pd.to_datetime(df_progression['date'], errors='coerce') 
    df_progression = df_progression.dropna(subset=['date'])
      
    # --- FILTRES DE PÉRIODE ET DE TYPE ---
    st.header("Filtres")
    col_f1, col_f2 = st.columns(2)
    
    # 1. Filtre Temporel
    periode_choisie = col_f1.selectbox("Sélectionnez la période d'analyse :", 
                                       ["Total", "Derniers 30 jours", "Derniers 90 jours", "Année en cours", "Personnalisée"])

    date_max = df_progression['date'].max()
    date_min_data = df_progression['date'].min()
    
    df_filtre_periode = df_progression.copy()
    
    if periode_choisie == "Derniers 30 jours":
        date_debut = date_max - timedelta(days=30)
        df_filtre_periode = df_progression[df_progression['date'] >= date_debut]
    elif periode_choisie == "Derniers 90 jours":
        date_debut = date_max - timedelta(days=90)
        df_filtre_periode = df_progression[df_progression['date'] >= date_debut]
    elif periode_choisie == "Année en cours":
        annee_actuelle = date_max.year
        date_debut = datetime(annee_actuelle, 1, 1).date()
        df_filtre_periode = df_progression[df_progression['date'].dt.year == annee_actuelle]
    elif periode_choisie == "Personnalisée":
        date_debut_filtre, date_fin_filtre = col_f1.date_input("Intervalle de dates", [date_min_data, date_max.date()])
        df_filtre_periode = df_progression[(df_progression['date'].dt.date >= date_debut_filtre) & (df_progression['date'].dt.date <= date_fin_filtre)]
        
    # 2. Filtre Type de Sport
    sports_disponibles = ['Tous'] + sorted(df_filtre_periode['type_sport'].unique().tolist())
    sport_choisi = col_f2.selectbox("Filtrer par type de sport :", sports_disponibles)

    if sport_choisi != 'Tous':
        df_final = df_filtre_periode[df_filtre_periode['type_sport'] == sport_choisi].copy()
    else:
        df_final = df_filtre_periode.copy()
        
    if df_final.empty:
        st.warning("Aucune activité trouvée avec les filtres sélectionnés.")
        return
        
    
    # --- 1. Statistiques Générales Cumulées (Période Filtrée) ---
    st.header(f"Statistiques Cumulées ({periode_choisie} - {sport_choisi})")
    
    total_distance = df_final['distance_km'].sum()
    total_denivele = df_final['denivele_positif_m'].sum()
    
    # Calcul de la tendance (exemple simple : comparaison à la première moitié de la période)
    # Plus complexe à coder proprement ici sans date de début/fin claire, on se concentre sur les métriques clés
    
    col_a, col_b, col_c = st.columns(3)
    
    with col_a:
        display_metric_card("Total Activités", f"{len(df_final):.0f}", "🔢")
    with col_b:
        display_metric_card("Distance Totale", f"{total_distance:,.1f} km", "🌍")
    with col_c:
        display_metric_card("Dénivelé Total", f"{total_denivele:,.0f} m", "🏔️")

    
    # --- 2. Progression Temporelle (Adaptation à la Période) ---
    st.header("Progression Mensuelle/Hebdomadaire")
    
    # Adapter le regroupement à la taille de la période
    if (date_max - df_final['date'].min()).days < 100:
        # Période courte : affichage hebdomadaire
        df_final['periode_label'] = df_final['date'].dt.strftime('%Y-S%W')
        periode_type = 'Semaine'
    else:
        # Période longue : affichage mensuel
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
        
    # --- 3. Répartition (Inchangée) ---
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
## Boucle principale (Sélecteur de Page - Inchangée)
# ----------------------------------------------------------------------

def main():
    
    # --- Configuration générale ---
    st.set_page_config(layout="wide", page_title="Analyse Strava Avancée")
    init_db() 
    
    st.sidebar.title("Navigation")
    
    # SÉLECTEUR DE PAGE PRINCIPAL
    page = st.sidebar.radio("Choisissez la vue :", ["🏃‍♂️ Analyse d'Activité", "📈 Tableau de Bord de Progression"])

    st.sidebar.markdown("---")
    
    if page == "🏃‍♂️ Analyse d'Activité":
        analyse_page()
    elif page == "📈 Tableau de Bord de Progression":
        progression_page()


if __name__ == "__main__":
    main()
