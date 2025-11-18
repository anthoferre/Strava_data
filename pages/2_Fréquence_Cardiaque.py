import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np  

st.title("Etude de la Fréquence Cardiaque")

if 'df_raw' in st.session_state:
    df_raw = st.session_state['df_raw']
    activity_name = st.session_state['activity_name']
    sport_type = st.session_state['sport_type']
    activity_date = st.session_state['activity_date']
    col1, col2 = st.columns([1.25,2])

    with col1:
        compte_par_zone = df_raw.groupby(by='zone_fc').size().reset_index(name='temps_en_echantillons')
        temps_total = compte_par_zone['temps_en_echantillons'].sum()
        compte_par_zone['proportion'] = (compte_par_zone['temps_en_echantillons'] / temps_total) * 100
        fig_fc_col1, ax = plt.subplots(figsize=(7,5))
        sns.barplot(data=compte_par_zone, x='zone_fc', y='proportion', ax=ax, palette='viridis')
        plt.xticks(rotation=45, ha='right')
        plt.ylabel('Proportion en %')
        st.pyplot(fig_fc_col1)
        plt.close(fig_fc_col1)

    ordres_zones = [
        '(0 - 60% FC Max) Récup',
        '(60 - 68% FC Max) End. Base',
        '(68 - 75% FC Max) End. Fond.',
        '(75 - 82% FC Max) Tempo',
        '(82 - 89% FC Max) Seuil',
        '(89 - 94% FC Max) VO2 Max',
        '(94 - 100% FC Max) Effort Max'
    ]
    df_raw['zone_fc'] = pd.Categorical(df_raw['zone_fc'], categories=ordres_zones, ordered=True)
    df_raw['temps_bin'] = (df_raw['temps_normalisee'] // 0.05) * 0.05
    df_agg_zones_fc = df_raw.groupby(by=['temps_bin','zone_fc']).size().unstack(fill_value=0)
    df_agg_zones_fc = df_agg_zones_fc[ordres_zones].reset_index()
    df_agg_zones_fc['total_echantillons_bin'] = df_agg_zones_fc[ordres_zones].sum(axis=1)
    df_agg_zones_fc['total_echantillons_bin'] = df_agg_zones_fc['total_echantillons_bin'].replace(0, 1)

    with col2:
        fig_fc_col2,ax = plt.subplots()
        x_data = df_agg_zones_fc['temps_bin'] * 100
        y_data = [(df_agg_zones_fc[zone] / df_agg_zones_fc['total_echantillons_bin']) *100 for zone in ordres_zones]
        cmap = plt.cm.viridis
        colors = cmap(np.linspace(0.1,1.0, len(ordres_zones)))
        ax.stackplot(x_data, y_data, labels=ordres_zones, colors=colors)
        plt.xlabel("Temps normalisée en % de l'activité")
        plt.ylabel("Proportion dans les différentes zones de FC")
        plt.legend(bbox_to_anchor=(1.42, 1), fontsize='small')
        st.pyplot(fig_fc_col2)
        plt.close(fig_fc_col2)


else:
    st.info("Veuillez sélectionner ou entrer un ID d'activité et cliquer sur **'🚀 Charger l'activité'** pour commencer l'analyse.")