import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly_express as px
import seaborn as sns
import streamlit as st

st.markdown("""
    <style>
    /* Titre principal en Orange Strava */
    h1 {{
        color: #FC4C02;
    }}

    [data-testid="stSubheader"] {{
    color: #FC4C02 !important;
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

st.title("💓 Etude de la Fréquence Cardiaque")

# Mapping fixe : Zone -> Couleur
fc_color_map = {
    '(0 - 60% FC Max) Récup': '#3498db',         # Bleu
    '(60 - 68% FC Max) End. Base': '#2ecc71',    # Vert
    '(68 - 75% FC Max) End. Fond.': '#f1c40f',   # Jaune
    '(75 - 82% FC Max) Tempo': '#e67e22',        # Orange
    '(82 - 89% FC Max) Seuil': '#e74c3c',        # Rouge clair
    '(89 - 94% FC Max) VO2 Max': '#c0392b',      # Rouge foncé
    '(94 - 100% FC Max) Effort Max': '#8e44ad'   # Violet
}

if 'df_raw' in st.session_state:
    df_raw = st.session_state['df_raw']
    activity_name = st.session_state['activity_name']
    sport_type = st.session_state['sport_type']
    activity_date = st.session_state['activity_date']

    with st.container(border=True):
        col1, col2 = st.columns([1, 2])
        with col1:
            # Barplot de répartition
            compte_par_zone = df_raw.groupby(by='zone_fc').size().reset_index(name='count')
            compte_par_zone_filtre = compte_par_zone[compte_par_zone['count'] > 0]
            fig_pie = px.pie(compte_par_zone_filtre, values='count', names='zone_fc',
                             title="Répartition par zone de FC",
                             color='zone_fc',
                             color_discrete_map=fc_color_map,
                             hole=0.2)
            fig_pie.update_layout(showlegend=False, height=350)
            st.plotly_chart(fig_pie, use_container_width=True)

        with col2:
            fig_hist = px.histogram(df_raw, x='frequence_cardiaque', color='zone_fc',
                                    title="Distribution de la Fréquence Cardiaque",
                                    color_discrete_map=fc_color_map,
                                    barmode='overlay',
                                    histnorm='percent',
                                    )
            fig_hist.update_layout(height=350, showlegend=False)
            st.plotly_chart(fig_hist, use_container_width=True)

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
    df_raw['temps_normalisee'] = (df_raw['temps_normalisee'] // 0.05) * 0.05
    df_agg_zones_fc = df_raw.groupby(by=['temps_normalisee', 'zone_fc']).size().unstack(fill_value=0)
    df_agg_zones_fc = df_agg_zones_fc.rolling(window=3, center=True).mean().fillna(0)

    # --- CHANGEMENT 2 : Supprimer les colonnes (zones) qui sont à zéro ---
    # On ne garde que les zones où la somme des échantillons est > 0
    zones_actives = [z for z in ordres_zones if z in df_agg_zones_fc.columns and df_agg_zones_fc[z].sum() > 0]
    df_agg_zones_fc = df_agg_zones_fc[zones_actives].reset_index()

    # 3. Calcul des proportions (pour que le total fasse 100% à chaque point X)
    df_agg_zones_fc['total_bin'] = df_agg_zones_fc[zones_actives].sum(axis=1).replace(0, 1)

    for zone in zones_actives:
        df_agg_zones_fc[zone] = (df_agg_zones_fc[zone] / df_agg_zones_fc['total_bin']) * 100

    with st.container(border=True):
        st.subheader("📈 Évolution des zones au fil de l'effort")

        fig_area = px.area(
            df_agg_zones_fc,
            x='temps_normalisee',
            y=zones_actives,
            color_discrete_map=fc_color_map,
            line_shape='spline', # Pour des courbes douces
            title="Répartition dynamique des zones de FC",
            labels={
                "value": "Proportion dans les zones de FC (%)",
                "temps_normalisee": "Progression de l'activité (%)"
            }
        )

        # Supprimer les bordures de lignes pour un effet "bloc de couleur" pur
        fig_area.update_traces(line=dict(width=0))
        fig_area.update_layout(hovermode="x unified", yaxis_range=[0, 100])

        st.plotly_chart(fig_area, use_container_width=True)

    with st.container(border=True):
        st.subheader("⛰️ Impact du profil sur la Fréquence Cardiaque")
        # Création d'un graphique à deux axes (Altitude et FC)
        fig_dual = go.Figure()

        # Courbe Altitude (en fond rempli)
        fig_dual.add_trace(go.Scatter(x=df_raw['distance_normalisee'], y=df_raw['altitude_m'],
                                      name='Altitude', fill='tozeroy', line_color='lightgrey', yaxis='y1'))

        # Courbe FC (par dessus)
        fig_dual.add_trace(go.Scatter(x=df_raw['distance_normalisee'], y=df_raw['frequence_cardiaque'],
                                      name='FC', line_color='#FC4C02', yaxis='y2'))

        fig_dual.update_layout(
            yaxis=dict(title="Altitude (m)", showgrid=False),
            yaxis2=dict(title="FC (bpm)", overlaying='y', side='right', showgrid=False),
            hovermode="x unified",
            height=400,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig_dual, use_container_width=True)

else:
    st.info("Veuillez sélectionner ou entrer un ID d'activité et cliquer sur **'🚀 "
            "Charger l'activité'** pour commencer l'analyse.")


