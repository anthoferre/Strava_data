# Etude_Graphique.py

import numpy as np
import pandas as pd
import streamlit as st

from utils.plotting import (calculate_vap_curve, coefficient_variation,
                            crosstab, plot_boxplot, plot_jointplot,
                            plot_vap_curve)
from utils.style_css import inject_custom_css

st.set_page_config(layout="wide")
inject_custom_css()

st.title("Visualisation des indicateurs de la performance")

if 'df_raw' in st.session_state:
    df_raw = st.session_state['df_raw']
    activity_name = st.session_state['activity_name']
    sport_type = st.session_state['sport_type']
    activity_date = st.session_state['activity_date']

    list_col_all = df_raw.columns.tolist()
    list_col_num = df_raw.select_dtypes([int, float]).columns.tolist()
    list_col_cat = df_raw.select_dtypes([object, 'category']).columns.tolist()

    date_activity = pd.to_datetime(activity_date)
    date_fr = date_activity.strftime("%d/%m/%Y - %Hh%M")

    header_container = st.container(border=True)
    with header_container:
        st.subheader("Résumé de l'Activité", divider="rainbow")
        m1, m2, m3 = st.columns(3)
        m1.metric("🏃 Activité", value=activity_name)
        m2.metric("📅 Date", value=date_fr)
        m3.metric("📍 Sport", value=sport_type)

    with st.expander("🔥 Etude heatmap", expanded=True):
        with st.container(border=True):
            st.subheader("Paramètres", divider="rainbow")

            aggfunc_dict = {
                'Moyenne' : np.mean,
                'Ecart_Type' : np.std,
                'Somme': np.sum,
                'Median': np.median,
                'Minimum' : np.min,
                'Maximum': np.max,
                'Coefficient_de_Variation': coefficient_variation
            }

            col1, col2 = st.columns(2)
            feature_option = col1.selectbox('Indicateur à étudier', options=[*list_col_num])
            aggfunc_option = col2.radio("Quelle fonction veux tu exécuter", options=list(aggfunc_dict.keys()))

            aggfunc = aggfunc_dict[aggfunc_option]

            ct_temp = pd.crosstab(
                index=df_raw['tranche_distance'],
                columns=df_raw['tranche_pente'],
                values=df_raw[feature_option],
                aggfunc=aggfunc
            ).fillna(0)

            valeurs_positives = ct_temp.values[ct_temp.values > 0]
            actual_min = float(valeurs_positives.min())
            actual_max = float(valeurs_positives.max())

            vmin, vmax = st.slider(
                label=f"Ajuster l'échelle de couleur pour {aggfunc_option}",
                min_value=actual_min,
                max_value=actual_max,
                value=(actual_min, actual_max)
            )
        with st.container(border=True):
            st.subheader(f"Heatmap : Etude de la fonction '{aggfunc_option}' sur la variable '{feature_option}'" , divider="rainbow")
            crosstab(df_raw, feature_option, aggfunc=aggfunc, vmin=vmin, vmax=vmax)

    st.divider()
    with st.expander("📊 BoxPlot"):
        with st.container(border=True):
            st.subheader("Paramètres", divider="rainbow")
            col_var_x, col_var_y = st.columns(2)
            var_x = col_var_x.selectbox("Variable en abscisse", options=list_col_cat)
            var_y = col_var_y.selectbox("Variable en ordonnée", options=list_col_num)

            var_hue = None
            if st.checkbox("Souhaites tu une troisième variable pour différentes couleurs?", key="var_hue_boxplot_option"):
                var_hue = st.selectbox("Variable couleur", options=[None] + list_col_cat, key="var_hue_boxplot")

        with st.container(border=True):
            st.subheader(f"Boxplot - Etude de la variable '{var_y}' en fonction de la catégorie '{var_x}'", divider="rainbow")
            if var_x is None or var_y is None:
                st.info("Sélectionne les deux variables pour le graphique")
            else:
                plot_boxplot(df_raw, var_x, var_y, var_hue)

    st.divider()

    # Joint Plot

    with st.expander("📈 Corrélation (Joint Plot)"):
        with st.container(border=True):
            st.subheader("Paramètres", divider="rainbow")
            col_var_x, col_var_y = st.columns(2)
            with col_var_x:
                var_x = st.selectbox("Variable en abscisse", options=list_col_num, key='var_x_joint_plot')
                list_col_num_var_y = [col for col in list_col_num if col != var_x]
            with col_var_y:
                var_y = st.selectbox("Variable en ordonnée", options=list_col_num_var_y, key='var_y_joint_plot')
            var_hue = None
            if st.checkbox("Souhaites tu une troisième variable pour différentes couleurs?", key="var_hue_jointplot_option"):
                var_hue = st.selectbox("Variable couleur", options=[None] + list_col_cat, key="var_hue_jointplot")

        with st.container(border=True):
            st.subheader(f"Joint Plot - Etude de la variable {var_y} en fonction de la catégorie {var_x}", divider="rainbow")
            if var_x is None or var_y is None:
                st.info("Sélectionne les deux variables pour le graphique")
            else:
                plot_jointplot(df_raw, var_x, var_y, var_hue)

    st.divider()

    with st.expander("🏆 Record d'Allure Moyenne Ajustée (VAP)"):
        interval_sec = [
            1, 5, 10, 30, 60,
            120, 300, 600, 1200, 1800,
            3600, 5400, 7200, 10800, 14400, 18000, 21600, 25200
        ]

        max_duration = len(df_raw)
        interval_sec = [i for i in interval_sec if i <= max_duration]
        if max_duration not in interval_sec:
            interval_sec.append(max_duration)

        vap_curve = calculate_vap_curve(df=df_raw, intervals=interval_sec)
        plot_vap_curve(vap_curve)


else:
    st.info("Veuillez sélectionner ou entrer un ID d'activité et cliquer sur **'🚀 Charger l'activité'** pour commencer l'analyse.")
