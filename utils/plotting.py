# plotting.py

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import streamlit as st
from matplotlib.ticker import FuncFormatter
from plotly.subplots import make_subplots


def coefficient_variation(feature):
    """Calcule le Coefficient de Variation (CV) : écart-type / moyenne."""
    mean = feature.mean()
    std = feature.std()

    # Évite la division par zéro si la moyenne est 0
    if mean == 0:
        return 0

    # Retourne le CV (souvent affiché en pourcentage si besoin, ici juste le ratio)
    return std / mean


def get_format(feature, aggfunc):
    """Retourne le format pour le crosstab"""
    if feature in ['vitesse_km_h_lissee', 'vap_vitesse', 'diff_vitesse']:
        return '.1f'
    elif feature in ['allure_min_km', 'efficacite_course', 'vap_allure']:
        return '.2f'
    elif aggfunc is coefficient_variation:
        return '.2f'
    return 'd'


def crosstab(df, feature, aggfunc, vmin=None, vmax=None):

    fmt_heatmap = get_format(feature, aggfunc)
    dtype_final = float if 'f' in fmt_heatmap else int

    ct = pd.crosstab(
        index=df['tranche_distance'],
        columns=df['tranche_pente'],
        values=df[feature],
        aggfunc=aggfunc
    ).fillna(0).astype(dtype_final)

    actual_min = float(ct.values.min())
    actual_max = float(ct.values.max())

    zmin = vmin if vmin is not None else actual_min
    zmax = vmax if vmax is not None else actual_max

    fig = go.Figure(data=go.Heatmap(
        z=ct.values.round(2),
        x=ct.columns.astype(str),
        y=ct.index.astype(str),
        colorscale='Oranges',
        zmin=zmin,
        zmax=zmax,
        text=ct.values.round(2),
        texttemplate="%{text}",
        hovertemplate="<b>Distance:</b> %{y}<br><b>Pente:</b> %{x}<br><b>Valeur:</b> %{z}<extra></extra>"
    ))

    # 3. Mise en forme esthétique (Le look "Strava")
    fig.update_layout(
        xaxis_title="Tranches de Pente (%)",
        yaxis_title="Tranches de Distance (km)",
        height=500,
        margin=dict(l=20, r=20, t=40, b=20),
        paper_bgcolor='rgba(0,0,0,0)',  # Fond transparent
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color="#FC4C02")    # On applique votre orange !
    )

    # 4. Affichage dans Streamlit
    st.plotly_chart(fig, use_container_width=True)


def plot_jointplot(df, x_var, y_var, hue_var=None):

    df_plot = df.dropna(subset=[x_var, y_var]).copy()

    if hue_var is None:
        fig = px.density_heatmap(
            data_frame=df_plot,
            x=x_var,
            y=y_var,
            marginal_x="histogram",
            marginal_y="histogram",
            color_continuous_scale=px.colors.sequential.Oranges,
            histnorm='percent'
        )
    else:
        fig = px.scatter(
            data_frame=df_plot,
            x=x_var,
            y=y_var,
            color=hue_var,
            marginal_x="box",
            marginal_y="box",
            color_discrete_sequence=px.colors.sequential.Oranges_r,
            opacity=0.6
        )

    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color="#FC4C02"),
        margin=dict(l=20, r=20, t=40, b=20),
        xaxis=dict(gridcolor='rgba(252, 76, 2, 0.1)', linecolor='#FC4C02'),
        yaxis=dict(gridcolor='rgba(252, 76, 2, 0.1)', linecolor='#FC4C02')
    )

    st.plotly_chart(fig, use_container_width=True)


def plot_boxplot(df, x_var, y_var, hue_var=None):

    df_plot = df.dropna(subset=[x_var]).copy()

    fig = px.box(
        data_frame=df_plot,
        x=x_var,
        y=y_var,
        color=hue_var,
        points="outliers",
        color_discrete_sequence=px.colors.sequential.Oranges_r,
        category_orders={x_var: sorted(df_plot[x_var].unique().tolist())}
    )
    # Mise à jour du design pour correspondre à ton thème sombre/orange
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color="#FC4C02"),
        margin=dict(l=20, r=20, t=40, b=20),
        xaxis=dict(
            title=x_var,
            showgrid=False,
            linecolor='#FC4C02',
            tickangle=45
        ),
        yaxis=dict(
            title=y_var,
            showgrid=True,
            gridcolor='rgba(252, 76, 2, 0.1)', # Grille orange très subtile
            linecolor='#FC4C02'
        ),
        boxmode='group' # Espace les boîtes si tu as une variable de couleur
    )

    # Affichage dans Streamlit
    st.plotly_chart(fig, use_container_width=True)

@st.cache_data
def plot_montees(df, feature_distance, feature_altitude, var_montee):
    """Trace la courbe d'altitude et celles des montées pour voir si la détection est bonne"""
    fig = make_subplots(specs=[[{'secondary_y': True}]])
    fig.add_trace(
        go.Scatter(
            x=df[feature_distance],
            y=df[feature_altitude],
            name='Altitude',
            line=dict(color="#A0A0A0", width=2),
            fill='tozeroy'
        ),
        secondary_y=False
    )
    fig.add_trace(
        go.Scatter(
            x=df[feature_distance],
            y=df[var_montee],
            name="Détection Montée",
            line=dict(color="#FC4C02", width=1.5), # Orange Strava
        ),
        secondary_y=True,
    )

    # Configuration des titres et du style
    fig.update_layout(
        title_text="Profil d'Altitude vs Détection des Montées",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    # Nommage des axes (Y-label)
    fig.update_yaxes(title_text="Altitude (m)", secondary_y=False)
    fig.update_yaxes(title_text="Indice de Montée", secondary_y=True, showgrid=False)
    fig.update_xaxes(title_text="Distance")

    return fig


def agg_sql_df_period(df, period, feature, sport_type_list):
    # 1. Filtrage
    df_sport_type = df[df['sport_type'].isin(sport_type_list)]

    # 2. Agrégation
    df_agg = df_sport_type.groupby(by=period)[feature].sum().reset_index()

    # Dictionnaire pour les noms de mois si nécessaire
    dict_month = {1: 'Janv.', 2: 'Févr.', 3: 'Mars', 4: 'Avril', 5: 'Mai', 6: 'Juin',
                  7: 'Juil.', 8: 'Août', 9: 'Sept.', 10: 'Oct.', 11: 'Nov.', 12: 'Déc.'}

    if period == 'month':
        df_agg['month_label'] = df_agg['month'].map(dict_month)
        x_axis = 'month_label'
    else:
        x_axis = period

    custom_scale = [[0, '#e0e0e0'], [1, '#FC4C02']]

    # 3. Création du graphique avec un dégradé (Viridis par défaut)
    fig = px.bar(
        df_agg,
        x=x_axis,
        y=feature,
        color=feature,  # Le dégradé se fait automatiquement sur la valeur
        color_continuous_scale=custom_scale,  # Remplace ton cmap
        labels={feature: feature.replace('_', ' '), period: period.capitalize()},
        template="plotly_white"
    )

    # 4. Ajustements esthétiques (équivalent de tes réglages d'axes)
    fig.update_layout(
        showlegend=False,
        margin=dict(l=20, r=20, t=30, b=20),
        height=350,
        coloraxis_showscale=True, # Affiche la barre de couleur (colorbar)
        xaxis=dict(
            tickangle=-45 if period == 'month' else 0,
            dtick=3 if period == 'week' else None # Équivalent MultipleLocator(3)
        )
    )

    # Configuration de la barre de couleur
    fig.update_coloraxes(colorbar_title_side="top", colorbar_thickness=15)

    return fig


def calculate_vap_curve(df: pd.DataFrame, intervals: list):
    """Calculer le Record d'Allure moyen par durée"""
    vap_results = {}
    vap_series = df['vap_allure']
    for duration in intervals:
        if duration == 1:
            best_vap = vap_series.min()
        else:
            rolling_avg = vap_series.rolling(window=duration, min_periods=duration).mean()
            best_vap = rolling_avg.min()
        vap_results[duration] = best_vap
    return vap_results


def time_formatter(x):
    """Convertit la valeur décimale de l'allure (ex: 4.5) en mm:ss (ex: 04:30)."""
    minutes = int(x)
    seconds = int((x - minutes) * 60)
    return f'{minutes:02d}:{seconds:02d}'


def plot_vap_curve(vap):
    vap_df = pd.DataFrame(
        list(vap.items()),
        columns=['Duree_s', 'Allure_Min_Moy']
    )

    vap_df['Allure_str'] = vap_df['Allure_Min_Moy'].apply(time_formatter)

    # Création des labels lisibles pour l'axe X et le survol
    def format_duration(s):
        s = int(s)
        if s >= 3600: return f"{s//3600}h{ (s%3600)//60 :02d}"
        if s >= 60: return f"{s//60}m"
        return f"{s}s"

    vap_df['Label'] = vap_df['Duree_s'].apply(format_duration)

    # 2. Création du graphique
    fig = go.Figure()

    # Ajout de la ligne de record
    fig.add_trace(go.Scatter(
        x=vap_df['Duree_s'],
        y=vap_df['Allure_Min_Moy'],
        mode='lines+markers',
        line=dict(color='#FC4C02', width=4),
        marker=dict(size=8, color='#333333', line=dict(width=1, color='#FC4C02')),
        customdata=vap_df[['Label', 'Allure_str']],
        hovertemplate="<b>Durée :</b> %{customdata[0]}<br><b>Allure VAP :</b> %{customdata[1]} min/km<extra></extra>"
    ))

    # 3. Configuration des axes (Log + Inversion Y)
    fig.update_layout(
        title="Profil de Performance - Allure Ajustée (VAP)",
        xaxis=dict(
            title="Durée de l'effort",
            type='log', # Échelle logarithmique
            tickvals=vap_df['Duree_s'],
            ticktext=vap_df['Label'],
            tickangle=45,
            gridcolor='rgba(252, 76, 2, 0.1)'
        ),
        yaxis=dict(
            title="Allure (min/km)",
            autorange='reversed', # INVERSION : le plus rapide en haut
            tickformat='%M:%S',   # Formatage automatique en mm:ss
            gridcolor='rgba(252, 76, 2, 0.1)'
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color="#FC4C02"),
        height=500,
        margin=dict(l=50, r=20, t=60, b=50)
    )

    st.plotly_chart(fig, use_container_width=True)


def plot_vap_curve_comparative(vap_curves: dict, sport_type: str):
    if not vap_curves or all(not d for d in vap_curves.values()):
        st.info("Aucune donnée VAP à tracer pour la comparaison.")
        return

    fig = go.Figure()

    # Génération d'un dégradé d'oranges/gris selon le nombre de courbes
    num_curves = len(vap_curves)
    # On utilise une palette séquentielle d'oranges
    colors = px.colors.sample_colorscale("Oranges", [i/(num_curves+1) for i in range(1, num_curves+1)])

    # 1. Boucle sur les courbes
    for i, (label, vap_dict) in enumerate(vap_curves.items()):
        if not vap_dict:
            continue

        # Préparation du DataFrame
        df = pd.DataFrame(list(vap_dict.items()), columns=['duration', 'pace_dec']).sort_values('duration')

        # Formatage des étiquettes pour le survol
        df['pace_str'] = df['pace_dec'].apply(time_formatter) # Utilise ta fonction existante
        df['duration_label'] = df['duration'].apply(lambda s: f"{int(s)//3600}h{int(s%3600)//60:02d}m" if s >= 3600
                                                    else f"{int(s)//60}m" if s >= 60
                                                    else f"{int(s)}s")

        # Style spécifique pour le "Record Absolu"
        is_absolute = "Absolu" in label
        line_color = "#FC4C02" if is_absolute else colors[i]
        line_width = 5 if is_absolute else 2

        # Ajout de la trace
        fig.add_trace(go.Scatter(
            x=df['duration'],
            y=df['pace_dec'],
            mode='lines+markers',
            name=label,
            line=dict(color=line_color, width=line_width),
            marker=dict(size=6),
            customdata=np.stack((df['duration_label'], df['pace_str']), axis=-1),
            hovertemplate="<b>%{name}</b><br>Durée : %{customdata[0]}<br>Allure : %{customdata[1]} min/km<extra></extra>"
        ))

        # OPTIONNEL : Zone de tolérance de 0.3% sur le Record Absolu
        if is_absolute:
            # On crée une limite haute et basse à +/- 0.3%
            fig.add_trace(go.Scatter(
                x=df['duration'], y=df['pace_dec'] * 1.003,
                line=dict(width=0), showlegend=False, hoverinfo='skip'
            ))
            fig.add_trace(go.Scatter(
                x=df['duration'], y=df['pace_dec'] * 0.997,
                line=dict(width=0), fill='tonexty',
                fillcolor='rgba(252, 76, 2, 0.1)', # Orange très transparent
                name="Zone de Performance (0.3%)", showlegend=True, hoverinfo='skip'
            ))

    # 2. Configuration des Axes et Layout
    fig.update_layout(
        xaxis=dict(
            title="Durée de l'effort (log)",
            type='log',
            gridcolor='rgba(252, 76, 2, 0.1)',
            tickvals=[1, 5, 10, 30, 60, 300, 600, 1800, 3600, 7200, 18000],
            ticktext=['1s', '5s', '10s', '30s', '1m', '5m', '10m', '30m', '1h', '2h', '5h'],
            tickangle=45
        ),
        yaxis=dict(
            title=f"Allure ({sport_type})",
            autorange='reversed', # Important : Allure rapide en haut
            tickformat='%M:%S',
            gridcolor='rgba(252, 76, 2, 0.1)'
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color="#FC4C02"),
        legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99, bgcolor="rgba(0,0,0,0)"),
        margin=dict(l=20, r=20, t=60, b=40),
        height=600
    )

    st.plotly_chart(fig, use_container_width=True)


def plot_record_regression(df_record):
    # 1. Nettoyage et tri pour le calcul
    df_plot = df_record.sort_values('distance_km').dropna(subset=['distance_km', 'best_time_min'])

    # 2. Calcul de la régression polynomiale (Ordre 2) avec NumPy
    # On évite ainsi d'utiliser statsmodels qui cause l'ImportError
    z = np.polyfit(df_plot['distance_km'], df_plot['best_time_min'], 2)
    p = np.poly1d(z)

    # Génération des points de la courbe de tendance
    x_range = np.linspace(df_plot['distance_km'].min(), df_plot['distance_km'].max(), 100)
    y_range = p(x_range)

    # 3. Création du graphique de base (Points)
    fig = px.scatter(
        df_plot,
        x='distance_km',
        y='best_time_min',
        labels={'distance_km': 'Distance (km)', 'best_time_min': 'Temps (min)'},
    )

    # 4. Ajout manuel de la ligne de tendance
    fig.add_trace(go.Scatter(
        x=x_range,
        y=y_range,
        mode='lines',
        name='Tendance (Ordre 2)',
        line=dict(color='#FC4C02', width=3, dash='dash'),
        hoverinfo='skip'
    ))

    def format_to_hms(minutes_dec):
        total_seconds = int(minutes_dec * 60)
        hours = total_seconds // 3600
        minutes = (total_seconds % 3600) // 60
        seconds = total_seconds % 60

        if hours > 0:
            return f"{hours}h {minutes:02d}m {seconds:02d}s"
        return f"{minutes:02d}m {seconds:02d}s"

    df_plot['time_str'] = df_plot['best_time_min'].apply(format_to_hms)

    fig.update_traces(
        marker=dict(size=10, color='#333333', line=dict(width=1, color='#FC4C02')),
        customdata=df_plot['time_str'],
        hovertemplate="<b>Distance:</b> %{x:.2f} km<br><b>Temps:</b> %{customdata}<extra></extra>",
        selector=dict(mode='markers')
    )

    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color="#FC4C02"),
        xaxis=dict(gridcolor='rgba(252, 76, 2, 0.1)', showgrid=True),
        yaxis=dict(gridcolor='rgba(252, 76, 2, 0.1)', showgrid=True, title="Temps (min)"),
        showlegend=False
    )

    st.plotly_chart(fig, use_container_width=True)