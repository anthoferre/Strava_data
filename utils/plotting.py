# plotting.py

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import streamlit as st
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


def crosstab(df, feature, aggfunc, vmax=None,):
    fmt_heatmap = get_format(feature, aggfunc)
    dtype_final = float if 'f' in fmt_heatmap else int
    fig, ax = plt.subplots(figsize=(15, 5))
    crosstab = pd.crosstab(df['tranche_distance'], df['tranche_pente'], df[feature], aggfunc=aggfunc).fillna(0).astype(dtype_final)
    min = crosstab.values.flatten()
    min = min[min > 0].min()
    if not pd.isna(min):
        vmin = min
    sns.heatmap(crosstab, annot=True, cmap='viridis', linewidths=0.5, fmt=fmt_heatmap, cbar=True, ax=ax, vmin=vmin, vmax=vmax,
                annot_kws={'fontsize': 12})
    st.pyplot(fig)
    plt.close(fig)


def plot_jointplot(df, x_var, y_var, hue_var=None):
    kwargs = {
        'data': df,
        'x' : x_var,
        'y': y_var,
        'kind': 'hex'
    }
    if hue_var is not None:
        kwargs['hue'] = hue_var
        kwargs['kind'] = 'scatter'
    joint_grid = getattr(sns, 'jointplot')(**kwargs)
    fig = joint_grid.figure
    fig.set_size_inches(10, 5)
    st.pyplot(fig)
    plt.close(fig)

def plot_boxplot(df,x_var,y_var, hue_var=None):
    kwargs = {
            'data': df,
            'x' : x_var,
            'y': y_var,
        }
    if hue_var is not None:
        kwargs['hue'] = hue_var

    fig, ax = plt.subplots(figsize=(12,5))
    getattr(sns,'boxplot')(**kwargs)
    plt.xticks(rotation=45, ha='right')
    st.pyplot(fig)
    plt.close(fig)

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


def time_formatter(x, pos):
    """Convertit la valeur décimale de l'allure (ex: 4.5) en mm:ss (ex: 04:30)."""
    minutes = int(x)
    seconds = int((x - minutes) * 60)
    return f'{minutes:02d}:{seconds:02d}'


def plot_vap_curve(vap):
    vap_df = pd.DataFrame(
        list(vap.items()),
        columns=['Durée(s)', 'Allure_Min_Moy']
    )

    fig, ax = plt.subplots()
    sns.lineplot(data=vap_df, x='Durée(s)', y='Allure_Min_Moy', marker='o', linewidth=3)
    plt.xscale('log')
    ax.invert_yaxis()
    duration_labels = [
        '1h30' if s == 5400
        else f'{s//3600}h' if s >= 3600
        else f'{s//60}m' if s >= 60
        else f'{s}s' for s in vap_df['Durée(s)']
    ]
    plt.title("Profil de performance en trail - Record d'Allure ajustée moyenne (VAP)")
    plt.ylabel("Allure Moyenne (min/km)")
    plt.xlabel("Durée de l'effort (échelle logarithmique)")
    plt.xticks(vap_df['Durée(s)'], duration_labels, rotation=45, fontsize=6)
    ax.yaxis.set_major_formatter(FuncFormatter(time_formatter))
    plt.yticks(fontsize=6)
    plt.grid(True, which="both", linestyle='--', alpha=0.5)
    sns.despine(left=True, bottom=True)
    st.pyplot(fig)
    plt.close(fig)

def plot_vap_curve_comparative(vap_curves: dict, title: str, sport_type: str):
    """
    Trace une ou plusieurs courbes de performance VAP pour comparaison.

    Args:
        vap_curves (dict): Dictionnaire où la clé est le nom de la courbe (ex: date, "Absolu")
                           et la valeur est le dictionnaire {duration: record}.
        title (str): Titre du graphique.
        sport_type (str): Type de sport (pour le titre).
    """
    if not vap_curves or all(not d for d in vap_curves.values()):
        st.info("Aucune donnée VAP à tracer pour la comparaison.")
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    # Couleurs pour la progression, basées sur le nombre de courbes
    num_curves = len(vap_curves)
    # Utilisation d'une palette pour la distinction
    couleurs = plt.cm.viridis(np.linspace(0, 1, num_curves))

    # Inversion de l'axe Y (Allure : plus bas = plus rapide)
    ax.invert_yaxis()

    all_durations = set()

    for i, (label, vap_dict) in enumerate(vap_curves.items()):
        if not vap_dict:
            continue

        vap_df = pd.DataFrame(
            list(vap_dict.items()),
            columns=['Durée(s)', 'Allure_Min_Moy']
        ).sort_values('Durée(s)') # Assure un tracé ordonné

        # Définir le style de la ligne (la ligne 'Absolu' ou la plus récente peut être plus épaisse)
        linewidth = 2.5 if "Absolu" in label or i == num_curves - 1 else 1.5

        # Tracé
        sns.lineplot(
            data=vap_df,
            x='Durée(s)',
            y='Allure_Min_Moy',
            marker='o',
            linewidth=linewidth,
            ax=ax,
            color=couleurs[i],
            label=label
        )
        all_durations.update(vap_df['Durée(s)'].tolist())

    # Préparation de l'axe X (avec les étiquettes de temps)
    sorted_durations = sorted(list(all_durations))

    duration_labels = [
        '2h' if s == 7200
        else '1h30' if s == 5400
        else f'{s//3600}h' if s >= 3600 and s % 3600 == 0
        else f'{s//60}m' if s >= 60 and s % 60 == 0
        else f'{s}s' for s in sorted_durations
    ]

    # Configuration des axes
    plt.xscale('log')
    plt.title(title, fontsize=14)
    plt.ylabel(f"Allure Moyenne {sport_type} (min/km)", fontsize=12)
    plt.xlabel("Durée de l'effort (échelle logarithmique)", fontsize=12)

    # Configuration des ticks de l'axe X (on n'affiche que les intervalles qui existent)
    ax.set_xticks(sorted_durations)
    ax.set_xticklabels(duration_labels, rotation=45, fontsize=10)

    # Configuration du formatteur de l'axe Y (mm:ss)
    ax.yaxis.set_major_formatter(FuncFormatter(time_formatter))

    # Légende et Grille
    plt.grid(True, which="both", linestyle='--', alpha=0.6)
    plt.legend(title="Courbes Comparées", loc='upper right', fontsize=9)
    sns.despine(left=False, bottom=False)

    st.pyplot(fig)
    plt.close(fig)
