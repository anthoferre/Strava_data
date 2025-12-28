#Prédiction.py

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from utils.data_processing import (calculate_all_records, fit_and_predict_time,
                                   time_formatter)
from utils.db_manager import (init_db, load_activity_records_by_key,
                              load_performance_records,
                              save_performance_records)
from utils.plotting import plot_record_regression
from utils.style_css import inject_custom_css

st.set_page_config(layout="wide")
inject_custom_css()


st.title("Prédiction des prochains temps de course")

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

    # Enregistrement des performances tous les km dans la database
    init_db()
    df_existing_record = load_activity_records_by_key(activity_date, sport_type)

    if df_existing_record is not None and not df_existing_record.empty:
        df_records = load_performance_records()

    else:
        max_distance_floor = int(np.floor(df_raw['distance_effort_itra'].max()))
        distances_list = [i for i in range(1, max_distance_floor + 1)]
        df_results = calculate_all_records(df_raw, 'distance_effort_itra', 'temps_min', distances_a_calculer = distances_list)

        save_performance_records(df_results, sport_type, activity_date)
        df_records = load_performance_records()

    df_record_per_distance = df_records.groupby(by='distance_km')['best_time_min'].min().reset_index()
    with st.container(border=True):
        with st.container(border=True):
            st.subheader("Paramètres de la prochaine course", divider="rainbow")
            col_distance, col_denivele = st.columns(2)
            with col_distance:
                new_distance = st.number_input("Nouvelle distance à prédire", key="New_distance", value=None)
            with col_denivele:
                new_denivele = st.number_input("Quel dénivelé positif sur la course à prédire", key="New_dénivelé_pos", value=None)
        with st.container(border=True):
            st.subheader("Estimation", divider="rainbow")
            if new_distance is None:
                st.info("Rentre la nouvelle distance à prédire")
            else:
                new_time, fig_reg = fit_and_predict_time(df_record_per_distance, new_distance, new_denivele)
                st.metric(label="Temps prédit", value=time_formatter(new_time * 60))
                with st.expander("Veux tu regarder la courbe de Puissance?"):
                    st.pyplot(fig_reg)
                    plt.close(fig_reg)

    with st.container(border=True):
        st.subheader('Meilleure performance historique par distance', divider="rainbow")
        plot_record_regression(df_record=df_record_per_distance)
