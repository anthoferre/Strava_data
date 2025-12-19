# Puissance.py

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly_express as px
import seaborn as sns
import streamlit as st

from utils.style_css import inject_custom_css

st.set_page_config(layout="wide")
inject_custom_css()


st.title("Etude de la Puissance")

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

    with st.container(border=True):
        st.subheader("Etude de la Distribution de la Puissance", divider="rainbow")
        col1, col2 = st.columns([1, 2])
        with col1:
            # Barplot de répartition
            compte_par_zone = df_raw.groupby(by='zone_puissance').size().reset_index(name='count')
            compte_par_zone_filtre = compte_par_zone[compte_par_zone['count'] > 0]
            fig_pie = px.pie(compte_par_zone_filtre, values='count', names='zone_puissance',
                             title="Répartition par zone de Puissance",
                             color='zone_puissance',
                             color_discrete_sequence=px.colors.sequential.Oranges,
                             hole=0.2)
            fig_pie.update_layout(showlegend=False, height=350)
            st.plotly_chart(fig_pie, use_container_width=True)

        with col2:
            fig_hist = px.histogram(df_raw, x='puissance_watts', color='zone_puissance',
                                    title="Distribution de la Puissance",
                                    color_discrete_sequence=px.colors.sequential.Oranges,
                                    barmode='overlay',
                                    histnorm='percent',
                                    )
            fig_hist.update_layout(height=350, showlegend=False)
            st.plotly_chart(fig_hist, use_container_width=True)

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
    df_raw['temps_normalisee'] = (df_raw['temps_normalisee'] // 0.05) * 0.05
    df_agg_zones_p = df_raw.groupby(by=['temps_normalisee', 'zone_puissance']).size().unstack(fill_value=0)
    df_agg_zones_p = df_agg_zones_p.rolling(window=3, center=True).mean().fillna(0)

    # --- CHANGEMENT 2 : Supprimer les colonnes (zones) qui sont à zéro ---
    # On ne garde que les zones où la somme des échantillons est > 0
    zones_actives = [z for z in ordres_zones_p if z in df_agg_zones_p.columns and df_agg_zones_p[z].sum() > 0]
    df_agg_zones_p = df_agg_zones_p[zones_actives].reset_index()

    # 3. Calcul des proportions (pour que le total fasse 100% à chaque point X)
    df_agg_zones_p['total_bin'] = df_agg_zones_p[zones_actives].sum(axis=1).replace(0, 1)

    for zone in zones_actives:
        df_agg_zones_p[zone] = (df_agg_zones_p[zone] / df_agg_zones_p['total_bin']) * 100

    with st.container(border=True):
        st.subheader("📈 Évolution des zones de Puissance au fil de l'effort", divider="rainbow")

        fig_area = px.area(
            df_agg_zones_p,
            x='temps_normalisee',
            y=zones_actives,
            color_discrete_sequence=px.colors.sequential.Oranges,
            line_shape='spline',
            labels={
                "value": "Proportion dans les zones de Puissance (%)",
                "temps_normalisee": "Progression de l'activité (%)"
            }
        )

        # Supprimer les bordures de lignes pour un effet "bloc de couleur" pur
        fig_area.update_traces(line=dict(width=0))
        fig_area.update_layout(hovermode="x unified", yaxis_range=[0, 100])

        st.plotly_chart(fig_area, use_container_width=True)

    with st.container(border=True):
        st.subheader("⛰️ Impact du profil sur la Puissance", divider="rainbow")
        # Création d'un graphique à deux axes (Altitude et Puissance)
        fig_dual = go.Figure()

        # Courbe Altitude (en fond rempli)
        fig_dual.add_trace(go.Scatter(x=df_raw['distance_normalisee'], y=df_raw['altitude_m'],
                                      name='Altitude', fill='tozeroy', line_color='lightgrey', yaxis='y1'))

        # Courbe FC (par dessus)
        fig_dual.add_trace(go.Scatter(x=df_raw['distance_normalisee'], y=df_raw['puissance_watts'],
                                      name='FC', line_color='#FC4C02', yaxis='y2'))

        fig_dual.update_layout(
            yaxis=dict(title="Altitude (m)", showgrid=False),
            yaxis2=dict(title="Puissance (watts)", overlaying='y', side='right', showgrid=False),
            hovermode="x unified",
            height=400,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig_dual, use_container_width=True)

else:
    st.info("Veuillez sélectionner ou entrer un ID d'activité et cliquer sur **'🚀 "
            "Charger l'activité'** pour commencer l'analyse.")

