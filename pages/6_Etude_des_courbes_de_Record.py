import pandas as pd
import streamlit as st

# Assurez-vous d'avoir plot_vap_curve_comparative et plot_vap_curve (ou la renommer)
from utils.db_manager import init_db, load_curve_records, save_curve_record
from utils.plotting import (calculate_vap_curve, plot_vap_curve,
                            plot_vap_curve_comparative)
from utils.style_css import inject_custom_css

st.set_page_config(layout="wide")
inject_custom_css()
st.title("🏃‍♂️ Profil de Performance : Courbe de Record VAP")

# Intervalles standards pour la courbe VAP (en secondes)
INTERVAL_SEC = [
    1, 5, 10, 30, 60,
    120, 300, 600, 1200, 1800,
    3600, 5400, 7200, 10800, 14400, 18000, 21600, 25200
]

if 'df_raw' in st.session_state:
    df_raw = st.session_state['df_raw']
    activity_name = st.session_state['activity_name']
    sport_type = st.session_state['sport_type']
    activity_date = st.session_state['activity_date']

    date_activity = pd.to_datetime(activity_date)
    date_fr = date_activity.strftime("%d/%m/%Y - %Hh%M")

    header_container = st.container(border=True)
    with header_container:
        st.subheader("Résumé de l'Activité", divider="rainbow")
        m1, m2, m3 = st.columns(3)
        m1.metric("🏃 Activité", value=activity_name)
        m2.metric("📅 Date", value=date_fr)
        m3.metric("📍 Sport", value=sport_type)

    init_db()

    # --- 1. CALCUL & SAUVEGARDE de l'activité actuelle ---

    if sport_type != "Bike" and 'vap_allure' in df_raw.columns:
        # ... (Logique de calcul et de sauvegarde de l'activité actuelle) ...

        vap_records_dict = calculate_vap_curve(df_raw, INTERVAL_SEC)

        if vap_records_dict:
            df_current_curve_records = pd.DataFrame(
                list(vap_records_dict.items()),
                columns=['duration', 'record']
            ).dropna(subset=['record'])

            if not df_current_curve_records.empty:
                # 🌟 SAUVEGARDE 🌟
                try:
                    save_curve_record(
                        df_current_curve_records,
                        sport_type,
                        activity_date
                    )
                    st.sidebar.success(f"Records VAP sauvegardés pour {activity_name}. Rechargement de l'historique...")
                except Exception as e:
                    st.error(f"Erreur lors de la sauvegarde des records de courbe : {e}")

    else:
        st.warning(f"L'étude de la courbe VAP est limitée aux activités de course/trail. Sport actuel : {sport_type}")

    # --- 2. ANALYSE HISTORIQUE et GRAPHIQUES ---
    with st.container(border=True):
        st.subheader("🥇 Record de Performance Absolu (Tous les Temps)")

        df_historical_curves = load_curve_records()

        if not df_historical_curves.empty:

            # Filtrer uniquement les sports pertinents et l'activité actuelle
            df_vap_history = df_historical_curves[
                df_historical_curves['sport_type'] == sport_type
            ].copy()

            if not df_vap_history.empty:

                # Conversion de la date (essentielle)
                df_vap_history['activity_date'] = pd.to_datetime(df_vap_history['activity_date'])
                df_vap_history['month_year'] = df_vap_history['activity_date'].dt.to_period('M')

                # Trouver le record le plus rapide (MIN) pour chaque durée, sur TOUTE l'histoire.
                df_absolute_best = df_vap_history.loc[
                    df_vap_history.groupby('duration')['record'].idxmin()
                ].reset_index(drop=True)

                absolute_best_dict = pd.Series(
                    df_absolute_best['record'].values,
                    index=df_absolute_best['duration']
                ).to_dict()

                # Tracé du Record Absolu (Utilisation de la fonction plot_vap_curve simple, ou la comparative avec une seule entrée)
                # J'utilise la version simple si elle existe encore, sinon la comparative
                try:
                    plot_vap_curve(absolute_best_dict)  # Si cette fonction trace une courbe simple
                except NameError:
                    plot_vap_curve_comparative({
                        f"Record Absolu ({sport_type})": absolute_best_dict
                        }, title="Profil de Performance Absolu", sport_type=sport_type
                    )
            else:
                st.info(f"Aucun record VAP trouvé pour le sport '{sport_type}' dans l'historique.")

        else:
            st.info("La table de records de courbe est vide.")

    with st.container(border=True):

        # --- B. GRAPHIQUE 2 : Évolution Mensuelle (Toutes les courbes mensuelles) ---
        st.subheader(f"📈 Évolution des Records : Comparaison Mensuelle pour le sport '{sport_type}'", divider="rainbow")

        # 1. Grouper par Mois/Année et Durée pour trouver le meilleur record mensuel
        df_monthly_best_records = df_vap_history.loc[
            df_vap_history.groupby(['month_year', 'duration'])['record'].idxmin()
        ].reset_index(drop=True)

        # 2. Préparer les options de sélection
        monthly_options = sorted(df_monthly_best_records['month_year'].astype(str).unique(), reverse=True)

        # 3. Widget de sélection
        selected_months = st.sidebar.multiselect(
            "Mois à afficher :",
            options=monthly_options,
            default=monthly_options[:min(3, len(monthly_options))]  # Sélectionne les 3 derniers mois par défaut
        )

        if selected_months:
            courbes_a_tracer = {}

            for month_str in selected_months:
                # Filtrer les records du mois sélectionné
                df_month = df_monthly_best_records[df_monthly_best_records['month_year'].astype(str) == month_str].copy()

                # Convertir en dictionnaire {duration: record}
                dict_curve = pd.Series(
                    df_month['record'].values,
                    index=df_month['duration']
                ).to_dict()

                courbes_a_tracer[month_str] = dict_curve

            # 4. Tracé des courbes mensuelles
            plot_vap_curve_comparative(
                courbes_a_tracer,
                sport_type=sport_type
            )

        else:
            st.info("Veuillez sélectionner au moins un mois.")

else:
    st.warning("⚠️ Veuillez charger une activité via la page principale pour étudier ses courbes de record.")
