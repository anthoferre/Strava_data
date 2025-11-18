# Puissance.py

import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

st.title("Etude de la Puissance")

if 'df_raw' in st.session_state:
    df_raw = st.session_state['df_raw']
    activity_name = st.session_state['activity_name']
    sport_type = st.session_state['sport_type']
    activity_date = st.session_state['activity_date']

    col1, col2 = st.columns([1.25,2])

    with col1:
        compte_par_zone = df_raw.groupby(by='zone_puissance').size().reset_index(name='temps_en_echantillons')
        temps_total = compte_par_zone['temps_en_echantillons'].sum()
        compte_par_zone['proportion'] = (compte_par_zone['temps_en_echantillons'] / temps_total) * 100
        fig_p_col1, ax = plt.subplots(figsize=(7,5))
        sns.barplot(data=compte_par_zone, x='zone_puissance', y='proportion', ax=ax, palette='viridis')
        plt.xticks(rotation=45, ha='right')
        plt.ylabel('Proportion en %')
        st.pyplot(fig_p_col1)
        plt.close(fig_p_col1)
    
    ordres_zones_p = [
        '(0 to 55% FTP) Récup',
        '(55 to 75% FTP) Endurance',
        '(75 to 90% FTP) Tempo',
        '(90 to 105% FTP) Seuil',
        '(105 to 120% FTP) VO2 Max',
        '(120 to 150% FTP) Capacité Anaérobie',
        '(150% to Max FTP) Effort Max'
    ]
    df_raw['zone_puissance'] = pd.Categorical(df_raw['zone_puissance'], categories=ordres_zones_p, ordered=True)
    df_agg_zones_p = df_raw.groupby(by=['temps_bin','zone_puissance']).size().unstack(fill_value=0)
    df_agg_zones_p = df_agg_zones_p[ordres_zones_p].reset_index()
    df_agg_zones_p['total_echantillons_bin'] = df_agg_zones_p[ordres_zones_p].sum(axis=1)
    df_agg_zones_p['total_echantillons_bin'] = df_agg_zones_p['total_echantillons_bin'].replace(0, 1)

    with col2:
        fig_p_col2,ax = plt.subplots()
        x_data = df_agg_zones_p['temps_bin'] * 100
        y_data = [(df_agg_zones_p[zone] / df_agg_zones_p['total_echantillons_bin']) *100 for zone in ordres_zones_p]
        cmap = plt.cm.viridis
        colors = cmap(np.linspace(0.1,1.0, len(ordres_zones_p)))
        ax.stackplot(x_data, y_data, labels=ordres_zones_p, colors=colors)
        plt.xlabel("Temps normalisée en % de l'activité")
        plt.ylabel("Proportion dans les différentes zones de Puissance")
        plt.legend(bbox_to_anchor=(1.42, 1), fontsize='small')
        st.pyplot(fig_p_col2)
        plt.close(fig_p_col2)

else:
    st.info("Veuillez sélectionner ou entrer un ID d'activité et cliquer sur **'🚀 Charger l'activité'** pour commencer l'analyse.")