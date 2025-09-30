import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import folium
from streamlit_folium import folium_static

# NOTE: Assurez-vous que ces fonctions sont disponibles dans components/utils.py
# Si ce n'est pas le cas, vous devez les définir ici pour que le code fonctionne.
# Exemple simple (à adapter à votre implémentation réelle) :
def format_allure(minutes_per_km):
    if pd.isna(minutes_per_km): return "N/A"
    minutes = int(minutes_per_km)
    seconds = round((minutes_per_km - minutes) * 60)
    return f"{minutes:01d}:{seconds:02d}"

def format_allure_std(std):
    return f"{std:.2f}"

# --- Définition des Variables Disponibles pour l'analyse personnalisée ---
METRICS_MAP = {
    'allure_lisse_corrigee': "Allure (min/km)",
    'vitesse_kmh': "Vitesse (km/h)",
    'fc_lisse': "Fréquence Cardiaque (bpm)",
    'pente_lisse': "Pente (°)",
    'altitude_m': "Altitude (m)",
}

# ----------------------------------------------------------------------
## Fonction d'Affichage de la Carte (Folium)
# ----------------------------------------------------------------------

def display_map(df, activity_name):
    """Affiche le tracé GPS de l'activité à l'aide de Folium."""
    if 'latlng' not in df.columns or df['latlng'].isnull().all():
        st.info("Les données GPS (latitude/longitude) ne sont pas disponibles pour cette activité.")
        return
        
    df_map = df.dropna(subset=['latlng']).copy()
    if df_map.empty:
        st.info("Aucun point GPS valide à afficher sur la carte.")
        return
        
    # Coordonnées de départ pour centrer la carte
    start_lat = df_map['latlng'].iloc[0][0]
    start_lon = df_map['latlng'].iloc[0][1]

    # Créer la carte Folium
    m = folium.Map(location=[start_lat, start_lon], zoom_start=13, tiles="openstreetmap")

    # Ajouter le tracé de l'activité
    points = df_map['latlng'].values.tolist()
    
    folium.PolyLine(
        points,
        color="#FC4C02", # Couleur Strava
        weight=5,
        opacity=0.8
    ).add_to(m)

    # Marqueur de départ
    folium.Marker(
        [start_lat, start_lon],
        popup=f"Départ : {activity_name}",
        icon=folium.Icon(color='green', icon='play')
    ).add_to(m)

    # Marqueur d'arrivée
    end_lat = df_map['latlng'].iloc[-1][0]
    end_lon = df_map['latlng'].iloc[-1][1]
    folium.Marker(
        [end_lat, end_lon],
        popup=f"Arrivée : {activity_name}",
        icon=folium.Icon(color='red', icon='flag')
    ).add_to(m)

    # Afficher la carte dans Streamlit
    with st.container():
        st.subheader("Tracé GPS de l'Activité")
        folium_static(m, width=900, height=450)


# ----------------------------------------------------------------------
## Fonctions de Création de Graphiques d'Analyse
# ----------------------------------------------------------------------

def creer_graphique_interactif(df, title, key=None):
    """Crée le graphique principal interactif avec Altitude, Allure, FC/Efficacité."""
    # (Le code de cette fonction reste inchangé, gérant le tracé principal de l'activité)
    # ... (omission pour concision, utilisez votre code original)
    
    if df is None or df.empty: return
    
    axes_definitions = {'yaxis': dict(title=dict(text='Altitude (m)', font=dict(color='blue')), tickfont=dict(color='blue'), side='left')}
    axis_count = 1
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['distance_km'], y=df['altitude_m'], mode='lines', name='Altitude', line=dict(color='blue'), yaxis='y'))
    
    # Axe Y pour l'Allure (Y-axis 2)
    if 'allure_lisse_corrigee' in df.columns and df['allure_lisse_corrigee'].any():
        axis_name_trace = f'y{axis_count + 1}'
        axis_name_layout = f'yaxis{axis_count + 1}'
        fig.add_trace(go.Scatter(x=df['distance_km'], y=df['allure_lisse_corrigee'], mode='lines', name='Allure Brute', yaxis=axis_name_trace, line=dict(color='green', dash='dot')))
        axes_definitions[axis_name_layout] = dict(title=dict(text='Allure (min/km)', font=dict(color='green')), tickfont=dict(color='green'), overlaying='y', side='right', anchor='x')
        axis_count += 1
        
    
    # Axe Y pour la Fréquence Cardiaque (Y-axis 3)
    if 'fc_lisse' in df.columns and df['fc_lisse'].any():
        axis_name_trace = f'y{axis_count + 1}'
        axis_name_layout = f'yaxis{axis_count + 1}'
        fig.add_trace(go.Scatter(x=df['distance_km'], y=df['fc_lisse'], mode='lines', name='Fréquence Cardiaque', yaxis=axis_name_trace, line=dict(color='pink')))
        
        axes_definitions[axis_name_layout] = dict(
            title=dict(text='FC (bpm)', font=dict(color='pink')), 
            tickfont=dict(color='pink'), 
            overlaying='y', 
            side='right', 
            anchor='free', 
            position=0.95
        )
        axis_count += 1
        
    # Axe Y pour l'Efficacité (Y-axis 4)
    if 'efficacite_course' in df.columns and df['efficacite_course'].any():
        axis_name_trace = f'y{axis_count + 1}'
        axis_name_layout = f'yaxis{axis_count + 1}'
        fig.add_trace(go.Scatter(x=df['distance_km'], y=df['efficacite_course'], mode='lines', name='Efficacité', yaxis=axis_name_trace, line=dict(color='orange')))
        
        axes_definitions[axis_name_layout] = dict(
            title=dict(text='Efficacité (km/h/bpm)', font=dict(color='orange')), 
            tickfont=dict(color='orange'), 
            overlaying='y', 
            side='right', 
            anchor='free', 
            position=0.90
        )
        axis_count += 1
        
    fig.update_layout(title=title, xaxis_title='Distance (km)', hovermode='x unified', legend=dict(x=0, y=1.1, orientation='h'), showlegend=True, height=500, **axes_definitions)
    
    return st.plotly_chart(fig, use_container_width=True, key=key)


def creer_graphique_allure_pente(df, title="Distribution de l'Allure vs Pente (Box Plot)"):
    """Crée le graphique d'Allure vs Pente en utilisant des boîtes à moustaches (Box Plot)."""
    # (Le code de cette fonction reste inchangé, utilisant le Box Plot comme convenu)
    # ...
    if df is None or df.empty:
        st.warning("Données d'activité manquantes pour le graphique Allure vs Pente.")
        return
        
    st.subheader(title)
    
    # --- 1. Filtrage et Préparation des données ---
    df_filtered = df[
        (df['allure_lisse_corrigee'] < 20) & 
        (df['allure_lisse_corrigee'] > 0) & 
        (df['pente_lisse'].abs() < 30)
    ].dropna(subset=['pente_lisse', 'allure_lisse_corrigee']).copy()
    
    if df_filtered.empty: 
        st.info("Après filtrage, données insuffisantes pour créer le graphique Box Plot."); 
        return
        
    df_filtered['pente_arrondie'] = df_filtered['pente_lisse'].round().astype(int).astype(str)
    
    # --- 2. Création du Box Plot avec Plotly Express ---
    fig = px.box(
        df_filtered, 
        x='pente_arrondie', 
        y='allure_lisse_corrigee', 
        title=title, 
        labels={
            'pente_arrondie': 'Pente Arrondie (%)', 
            'allure_lisse_corrigee': 'Allure (min/km)'
        },
        points='outliers',
        notched=True
    )
    
    # --- 3. Personnalisation ---
    fig.update_layout(
        xaxis_title="Pente (en degrés)", 
        yaxis_title="Allure lissée (min/km)", 
        height=500
    )
    fig.update_yaxes(autorange="reversed") 
    
    st.plotly_chart(fig, use_container_width=True)

def creer_graphique_vam(df, title="VAM vs Pente"):
    """Crée le graphique de Vitesse Ascensionnelle Métrique (VAM) moyenne par Pente."""
    # (Le code de cette fonction reste inchangé)
    # ... (omission pour concision)
    if df is None or df.empty:
        st.warning("Données d'activité manquantes pour le graphique VAM vs Pente.")
        return
        
    st.subheader(title)
    df_montées = df[(df['pente_lisse'] > 2) & (df['pente_lisse'] < 30) & (df['allure_lisse_corrigee'] > 0)].copy()
    if not df_montées.empty:
        df_montées['delta_altitude'] = df_montées['altitude_m'].diff().fillna(0)
        df_montées['temps_diff_seconds'] = df_montées['temps'].diff().dt.total_seconds().replace(0, np.nan).fillna(1)
        
        # Filtre de temps : VAM n'a de sens que sur les segments où il y a eu une durée
        df_montées = df_montées[df_montées['temps_diff_seconds'] > 1]
        
        df_montées['vam_calculée'] = (df_montées['delta_altitude'] / df_montées['temps_diff_seconds'] * 3600).fillna(0)
        df_montées = df_montées[df_montées['vam_calculée'] > 0]
        
        if df_montées.empty:
             st.info("Aucune donnée de montée significative (VAM > 0) pour l'analyse VAM après filtrage.")
             return

        df_montées['pente_arrondie'] = df_montées['pente_lisse'].round().astype(int)
        grouped_vam = df_montées.groupby('pente_arrondie').agg(
            vam_moyenne=('vam_calculée', 'mean'),
            vam_std=('vam_calculée', 'std')
        ).reset_index()
        
        grouped_vam['vam_upper_bound'] = grouped_vam['vam_moyenne'] + grouped_vam['vam_std']
        grouped_vam['vam_lower_bound'] = grouped_vam['vam_moyenne'] - grouped_vam['vam_std']
        
        fig_vam = go.Figure()
        fig_vam.add_trace(go.Scatter(x=grouped_vam['pente_arrondie'], y=grouped_vam['vam_upper_bound'], mode='lines', line=dict(width=0), showlegend=False, hoverinfo='skip'))
        fig_vam.add_trace(go.Scatter(x=grouped_vam['pente_arrondie'], y=grouped_vam['vam_lower_bound'], mode='lines', line=dict(width=0), fill='tonexty', fillcolor='rgba(173, 216, 230, 0.4)', name='Écart type'))
        fig_vam.add_trace(go.Scatter(x=grouped_vam['pente_arrondie'], y=grouped_vam['vam_moyenne'], mode='lines+markers', line=dict(color='blue'), name='VAM moyenne', hovertemplate="Pente: %{x}°<br>VAM: %{y:.0f} m/h"))
        
        fig_vam.update_layout(title='VAM moyenne en fonction de la Pente', xaxis_title='Pente (°)', yaxis_title='VAM (m/h)', hovermode='x unified', height=400)
        st.plotly_chart(fig_vam, use_container_width=True)
    else:
        st.info("Aucune donnée de montée significative (pente > 2°) pour l'analyse VAM.")


def creer_graphique_fc_pente(df, title="FC vs Pente"):
    """Crée le graphique de Fréquence Cardiaque (FC) moyenne par Pente."""
    # (Le code de cette fonction reste inchangé)
    # ... (omission pour concision)
    if df is None or df.empty or 'fc_lisse' not in df.columns or df['fc_lisse'].isnull().all():
        st.warning("Impossible d'effectuer l'analyse de la FC : données manquantes.")
        return
        
    st.subheader(title)
    df_fc = df[(df['fc_lisse'] > 0) & (df['pente_lisse'].abs() < 30)].copy()
    if not df_fc.empty:
        df_fc['pente_arrondie'] = df_fc['pente_lisse'].round().astype(int)
        df_fc['poids_temporel'] = df_fc['temps'].diff().dt.total_seconds().fillna(0)
        
        grouped_fc = df_fc.groupby('pente_arrondie').apply(lambda x: pd.Series({
            'fc_moyenne': (x['fc_lisse'] * x['poids_temporel']).sum() / x['poids_temporel'].sum(),
            'fc_std': np.std(x['fc_lisse'])
        }) if x['poids_temporel'].sum() > 0 else pd.Series({'fc_moyenne': np.nan, 'fc_std': np.nan})).reset_index()
        
        grouped_fc.dropna(subset=['fc_moyenne'], inplace=True)
        
        grouped_fc['fc_upper_bound'] = grouped_fc['fc_moyenne'] + grouped_fc['fc_std']
        grouped_fc['fc_lower_bound'] = grouped_fc['fc_moyenne'] - grouped_fc['fc_std']
        
        fig_fc = go.Figure()
        fig_fc.add_trace(go.Scatter(x=grouped_fc['pente_arrondie'], y=grouped_fc['fc_upper_bound'], mode='lines', line=dict(width=0), showlegend=False, hoverinfo='skip'))
        fig_fc.add_trace(go.Scatter(x=grouped_fc['pente_arrondie'], y=grouped_fc['fc_lower_bound'], mode='lines', line=dict(width=0), fill='tonexty', fillcolor='rgba(255, 105, 180, 0.4)', name='Écart type'))
        fig_fc.add_trace(go.Scatter(x=grouped_fc['pente_arrondie'], y=grouped_fc['fc_moyenne'], mode='lines+markers', line=dict(color='deeppink'), name='FC moyenne', hovertemplate="Pente: %{x}°<br>FC: %{y:.0f} bpm"))
        
        fig_fc.update_layout(title='Fréquence Cardiaque moyenne en fonction de la Pente', xaxis_title='Pente (°)', yaxis_title='FC (bpm)', hovermode='x unified', height=400)
        st.plotly_chart(fig_fc, use_container_width=True)
    else:
        st.info("Données de fréquence cardiaque insuffisantes pour l'analyse.")


def creer_graphique_ratio_vitesse_fc(df, title="Efficacité de foulée"):
    """Crée le graphique du ratio Vitesse/FC par Pente."""
    # (Le code de cette fonction reste inchangé)
    # ... (omission pour concision)
    if df is None or df.empty or 'vitesse_kmh' not in df.columns or 'fc_lisse' not in df.columns or df['fc_lisse'].isnull().all() or df['vitesse_kmh'].isnull().all():
        st.warning("Impossible de calculer l'efficacité de la foulée : données manquantes.")
        return
        
    st.subheader(title)
    df_foulée = df[(df['vitesse_kmh'] > 0) & (df['fc_lisse'] > 0) & (df['pente_lisse'].abs() < 30)].copy()
    if not df_foulée.empty:
        df_foulée['ratio_vitesse_fc'] = df_foulée['vitesse_kmh'] / df_foulée['fc_lisse']
        df_foulée['pente_arrondie'] = df_foulée['pente_lisse'].round().astype(int)
        df_foulée['poids_temporel'] = df_foulée['temps'].diff().dt.total_seconds().fillna(0)
        
        grouped_ratio = df_foulée.groupby('pente_arrondie').apply(lambda x: pd.Series({
            'ratio_moyen': (x['ratio_vitesse_fc'] * x['poids_temporel']).sum() / x['poids_temporel'].sum(),
            'ratio_std': np.std(x['ratio_vitesse_fc'])
        }) if x['poids_temporel'].sum() > 0 else pd.Series({'ratio_moyen': np.nan, 'ratio_std': np.nan})).reset_index()
        
        grouped_ratio.dropna(subset=['ratio_moyen'], inplace=True)
        grouped_ratio.sort_values('pente_arrondie', inplace=True)
        
        grouped_ratio['ratio_upper_bound'] = grouped_ratio['ratio_moyen'] + grouped_ratio['ratio_std']
        grouped_ratio['ratio_lower_bound'] = grouped_ratio['ratio_moyen'] - grouped_ratio['ratio_std']
        
        fig_ratio = go.Figure()
        fig_ratio.add_trace(go.Scatter(x=grouped_ratio['pente_arrondie'], y=grouped_ratio['ratio_upper_bound'], mode='lines', line=dict(width=0), showlegend=False, hoverinfo='skip'))
        fig_ratio.add_trace(go.Scatter(x=grouped_ratio['pente_arrondie'], y=grouped_ratio['ratio_lower_bound'], mode='lines', line=dict(width=0), fill='tonexty', fillcolor='rgba(255, 165, 0, 0.4)', name='Écart type'))
        fig_ratio.add_trace(go.Scatter(x=grouped_ratio['pente_arrondie'], y=grouped_ratio['ratio_moyen'], mode='lines+markers', line=dict(color='orange'), name='Ratio moyen', hovertemplate="Pente: %{x}°<br>Ratio: %{y:.3f}"))
        
        fig_ratio.update_layout(title='Ratio Vitesse/FC en fonction de la Pente', xaxis_title='Pente (°)', yaxis_title='Ratio Vitesse/FC', hovermode='x unified', height=400)
        st.plotly_chart(fig_ratio, use_container_width=True)
    else:
        st.info("Impossible de calculer l'efficacité de la foulée : données manquantes.")


def creer_graphique_comparaison(df1, name1, df2, name2, variable, y_title):
    """Crée un graphique pour comparer une variable entre deux activités."""
    # (Le code de cette fonction reste inchangé)
    # ... (omission pour concision)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df1['distance_km'], y=df1[variable], mode='lines', name=name1, line=dict(width=3)))
    if df2 is not None and not df2.empty:
        fig.add_trace(go.Scatter(x=df2['distance_km'], y=df2[variable], mode='lines', name=name2, line=dict(dash='dash', width=3)))
    
    if variable == 'allure_lisse_corrigee':
        fig.update_yaxes(autorange="reversed")
        
    fig.update_layout(title=f"Comparaison de **{y_title}** en fonction de la distance", xaxis_title="Distance (km)", yaxis_title=y_title, height=500)
    st.plotly_chart(fig, use_container_width=True)


# ----------------------------------------------------------------------
## NOUVELLE FONCTION : Analyse de Segment Personnalisée (Avec Choix du Type de Graphique)
# ----------------------------------------------------------------------

def creer_analyse_segment_personnalisee(df, start_km, end_km):
    """
    Permet à l'utilisateur de sélectionner une portion de l'activité, une variable
    et le type de graphique pour visualiser la distribution ou l'évolution.
    """
    if df is None or df.empty:
        st.warning("Données d'activité manquantes pour l'analyse de segment.")
        return
    
    # 1. Contrôles Utilisateur (Widgets Streamlit)
    col_sel, col_graph = st.columns([1, 2])
    
    with col_sel:
        # Choix de la Variable
        variable_choisie = st.selectbox(
            "1. Variable à étudier :",
            options=list(METRICS_MAP.keys()),
            format_func=lambda x: METRICS_MAP.get(x, x),
            key="segment_var_select"
        )
        variable_titre = METRICS_MAP.get(variable_choisie, variable_choisie)

        # Choix du Type de Graphique
        type_graphique = st.radio(
            "2. Type de Visualisation :",
            options=['Évolution (Ligne)', 'Distribution (Histogramme)'],
            key="segment_graph_type"
        )
        
    with col_graph:
        # Sélecteur de Segment (Distance)
        if 'distance_km' not in df.columns:
            st.warning("Colonne 'distance_km' manquante pour la sélection de segment.")
            return

    
    
    # 2. Filtrage des Données
    df_segment = df[
        (df['distance_km'] >= start_km) & 
        (df['distance_km'] <= end_km)
    ].dropna(subset=[variable_choisie]).copy()
    
    if df_segment.empty:
        st.info(f"Aucune donnée valide pour le segment [{start_km:.1f} km - {end_km:.1f} km] pour la variable **{variable_titre}**.")
        return

    # 3. Création du Graphique en fonction du Choix
    st.markdown("---")

    if type_graphique == 'Évolution (Ligne)':
        # Graphique d'Évolution (Ligne)
        fig = px.line(
            df_segment, 
            x='distance_km', 
            y=variable_choisie, 
            title=f"Évolution de la {variable_titre} sur le segment",
            labels={'distance_km': 'Distance (km)', variable_choisie: variable_titre}
        )
        if variable_choisie in ['allure_lisse_corrigee']:
            fig.update_yaxes(autorange="reversed")
        
    elif type_graphique == 'Distribution (Histogramme)':
        # Graphique de Distribution (Histogramme)
        fig = px.histogram(
            df_segment, 
            x=variable_choisie, 
            nbins=30, 
            marginal="box",
            title=f"Distribution de la {variable_titre} sur le segment"
        )
        mean_val = df_segment[variable_choisie].mean()
        median_val = df_segment[variable_choisie].median()
        
        fig.add_vline(x=mean_val, line_dash="dash", line_color="red")
        fig.add_vline(x=median_val, line_dash="dash", line_color="green")

        
    
        if variable_choisie in ['allure_lisse_corrigee']:
            fig.update_yaxes(autorange="reversed")

    else:
        st.error("Type de graphique non reconnu.")
        return

    # Finalisation et Affichage
    fig.update_layout(height=450)
    st.plotly_chart(fig, use_container_width=True)