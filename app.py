import streamlit as st
from datetime import datetime, timezone
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.ticker import MultipleLocator
import matplotlib.colors as mcolors
from db_manager import init_db, get_db_connection, save_performance_records, load_performance_records, init_db, load_activity_records_by_key, sql_df
from strava_api import get_last_activity_ids, get_activity_data_from_api
from scipy.stats import linregress  

st.set_page_config(layout='wide')
# --- Configuration et Initialisation des Secrets ---
# Stocker les secrets en session state pour une vérification rapide
if 'CLIENT_ID' not in st.session_state:
    try:
        st.session_state['CLIENT_ID'] = st.secrets["CLIENT_ID"]
        st.session_state['CLIENT_SECRET'] = st.secrets["CLIENT_SECRET"]
        st.session_state['ACCESS_TOKEN'] = st.secrets["ACCESS_TOKEN"]
        st.session_state['REFRESH_TOKEN'] = st.secrets["REFRESH_TOKEN"]
        st.session_state['EXPIRES_AT'] = st.secrets["EXPIRES_AT"]
    except KeyError:
        pass


# ----------------------------------------------------------------------
# FONCTIONS CACHÉES POUR LA PERFORMANCE
# ----------------------------------------------------------------------

@st.cache_resource
def init_db_cached():
    """Initialise la base de données (mise en cache)."""
    # init_db() devrait être rapide s'il gère les connexions existantes
    init_db() 
    return True

@st.cache_resource
def get_db_connection_cached():
    """Récupère la connexion DB (mise en cache)."""
    return get_db_connection()

@st.cache_data(ttl=3600) # Cache pendant 1 heure
def get_last_activity_ids_cached(limit=200):
    """Récupère les IDs d'activités récentes (mise en cache)."""
    return get_last_activity_ids(limit)

@st.cache_data(show_spinner="Téléchargement des données brutes Strava...", ttl=300) 
def get_activity_data_from_api_cached(activity_id):
    """Récupère et met en cache les données brutes d'une activité Strava."""
    return get_activity_data_from_api(activity_id)

def calculate_vap(vitesse_km_h: pd.Series, pente_perc: pd.Series) -> pd.Series:
    """Calcule la Vitesse Ajustée à la Pente (VAP)."""
    i = pente_perc / 100
    # Coefficients tirés d'un modèle (ex: Minetti)
    Cr = (155.4 * (i**5) - 30.4 * (i**4) - 43.3 * (i**3) + 46.3 * (i**2) + 19.5 * i + 3.6)
    Cout_Plat = 3.6
    # allure_vap = allure_plate * (Coût_Plat / Coût_Pente)
    vitesse_vap = vitesse_km_h * (Cout_Plat / Cr)
    return vitesse_vap

def normalisation_data(df, feature):
        """Normalisation des données entre 0 et 100%"""
        reset = df[feature] - df[feature].min()
        total = df[feature].max() - df[feature].min()
        normalisation = df[feature] / total
        return reset, normalisation
    
def cutting_data_percent(df, feature, min_list=0, max_list=100, nb_bins=10):
    """Coupe les données"""
    step_list = (max_list - min_list) / nb_bins
    list = [f"{i} to {i+step_list}%" for i in range(min_list,max_list,nb_bins)]
    feature_cut = pd.cut(df[feature], bins=nb_bins, include_lowest=False, labels=list)
    return feature_cut

def allure_format(feature):
    """Conversion de l'allure en min/km"""
    allure = pd.Series(feature)
    allure.fillna(value=0, inplace=True)
    allure_s = np.round(allure % 1.0 * 60).astype(int)
    allure_min = np.floor(allure).astype(int)
    allure_min_str = allure_min.astype(str).str.zfill(2) # "4" -> "04"
    allure_s_str = allure_s.astype(str).str.zfill(2)     # "5" -> "05"
    allure_formatee = (allure_min_str.str.cat(allure_s_str, sep='.')).astype(float)
    if allure_formatee.size == 1:
        return (allure_min_str.str.cat(allure_s_str, sep=':')).iloc[0] # Retourne le float simple
    else:
        return (allure_min_str.str.cat(allure_s_str, sep='.')).astype(float) # Retourne la Série complète

def calculer_denivele(feature):
    """Calculer le dénivelé positif et négatif cumulé sur la sortie"""
    diff_altitude = feature.diff()
    denivele_pos_instant = diff_altitude.clip(lower=0).fillna(0)
    denivele_neg_instant = diff_altitude.clip(upper=0).fillna(0)
    d_pos_cumule = denivele_pos_instant.cumsum().fillna(0)
    denivele_neg_cumule = denivele_neg_instant.cumsum().fillna(0)
    return denivele_pos_instant, denivele_neg_instant, d_pos_cumule, denivele_neg_cumule

def conversion_temps_total(feature_h, feature_min):
    """Convertit le temps total de la sortie en hh:mm:ss"""
    temps_total_h = np.floor(feature_h.iloc[-1]).astype(int)
    temps_total_min = np.floor((feature_h.iloc[-1] % 1) *60.0).astype(int)
    temps_total_sec = np.round((feature_min.iloc[-1] % 1) * 60,0).astype(int)

    temps_formatte = f"{temps_total_h:02d}:{temps_total_min:02d}:{temps_total_sec:02d}"
    return temps_formatte

def coefficient_variation(feature):
    """Calcule le Coefficient de Variation (CV) : écart-type / moyenne."""
    mean = feature.mean()
    std = feature.std()
    
    # Évite la division par zéro si la moyenne est 0
    if mean == 0:
        return 0 
    
    # Retourne le CV (souvent affiché en pourcentage si besoin, ici juste le ratio)
    return std / mean

def min_max_scaler(feature):
    vmin = 0
    vmax = 30/200 #30km/h pour 200bpm
    valeur_normalisee = (feature-vmin) / (vmax-vmin) * 100
    return valeur_normalisee

def plot_jointplot(df,x_var,y_var, hue_var=None):
        kwargs = {
            'data': df,
            'x' : x_var,
            'y': y_var,
            'kind': 'hex'
        }
        if hue_var is not None:
            kwargs['hue'] = hue_var
            kwargs['kind'] = 'scatter'
        joint_grid = getattr(sns,'jointplot')(**kwargs)
        fig = joint_grid.figure
        fig.set_size_inches(10,5)
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

def calculer_limites_iqr(df, feature):
    """
    """
    # 1. Calcul des quartiles
    Q1 = df[feature].quantile(0.25)
    Q3 = df[feature].quantile(0.75)
    
    # 2. Calcul de l'Écart Interquartile (IQR)
    IQR = Q3 - Q1

    # 3. Facteur iqr value
    iqr_factor = 1.5 * IQR

    # 4. Calcul des limites d'aberration
    limite_inf = Q1 - iqr_factor
    limite_sup = Q3 + iqr_factor
    
    return limite_inf, limite_sup

def drop_extreme_value(df, feature, FENETRE_LISSAGE):
    """Remplacer les valeurs aberrantes par une interpolation linéaire puis réaliser un lissage par la suite"""
    seuil_min, seuil_max = calculer_limites_iqr(df, feature)
    df_temp = df.copy()
    masque = (df[feature] > seuil_max) | (df[feature] < seuil_min)
    df_temp.loc[masque, feature] = np.nan
    df_temp[feature].interpolate(method='linear', inplace=True)
    feature_lissee = df_temp[feature].rolling(window=FENETRE_LISSAGE, center=True).mean()
    feature_lissee.fillna(df_temp[feature], inplace=True)
    
    return feature_lissee

@st.cache_data
def calculate_all_records(df, feature_distance, feature_tps, distances_a_calculer):
    """
    Calcule la meilleure performance (temps et allure) pour toutes les distances cibles 
    dans une activité, en utilisant la méthode du balayage et de l'interpolation.
    
    Args:
        df (pd.DataFrame): Le DataFrame de l'activité.
        feature_distance (str): Nom de la colonne de distance (e.g., 'distance_effort_itra').
        feature_tps (str): Nom de la colonne de temps (e.g., 'temps_relatif_sec').
        distances_a_calculer (list): Liste des distances cibles en km (e.g., [5, 10, 21.1]).
        
    Returns:
        pd.DataFrame: DataFrame contenant 'Distance (km)', 'Meilleur Temps (min)', 
                      et 'Allure (min/km)' pour les records trouvés.
    """
    all_records = []
    
    # Récupérer les données brutes pour éviter de faire .loc[] répétitif
    distances_cumulees = df[feature_distance]
    temps_cumules = df[feature_tps]
    
    # 1. Itération sur chaque distance cible (e.g., 5 km, 10 km)
    for distance_obj in distances_a_calculer:
        
        # Vérification si l'activité est assez longue pour la distance cible
        if distance_obj > distances_cumulees.max():
            # Si la course n'est pas assez longue, passer à la distance suivante
            continue 

        segment_temps = []
        
        # 2. Balayage de tous les points de départ possibles 'i'
        for i in range(len(df)):
            # Distance cible que l'on essaie d'atteindre : distance_au_point_i + distance_obj
            dist_cible = distances_cumulees.loc[i] + distance_obj

            # Trouver le premier point 'j' après 'i' qui a dépassé ou atteint dist_cible
            # (mask_fin est basé sur l'index 'i+1:' de la série initiale)
            mask_fin = distances_cumulees.loc[i+1:] >= dist_cible

            if not mask_fin.any():
                # Si la fin de l'activité ne peut pas atteindre dist_cible, on arrête le balayage pour cette distance
                break 

            # L'index j est le premier index où la condition est VRAIE
            j = np.argmax(mask_fin.values) + i + 1
            
            # --- Interpolation ---
            
            # Temps et distance des points j et j-1
            tps_j = temps_cumules.loc[j]
            tps_prec = temps_cumules.loc[j-1]
            dist_j = distances_cumulees.loc[j]
            dist_prec = distances_cumulees.loc[j-1]
            
            # Temps estimé pour atteindre précisément dist_cible
            tps_interpole = tps_prec + (tps_j - tps_prec) * (dist_cible - dist_prec) / (dist_j - dist_prec)
            
            # Temps total pour le segment [point i -> dist_cible]
            tps_segment = tps_interpole - temps_cumules.loc[i]
            
            segment_temps.append(tps_segment)
        
        # 3. Stockage du meilleur record pour cette distance_obj
        if segment_temps:
            best_time_min = min(segment_temps)
            pace_min_per_km = best_time_min / distance_obj
            
            all_records.append({
                'Distance (km)': distance_obj,
                'Meilleur Temps (min)': best_time_min,
                'Allure (min/km)': pace_min_per_km
            })

    # 4. Retour du DataFrame final
    return pd.DataFrame(all_records)

def fit_and_predict_time(df_records, new_distance_km, new_denivele_pos):
    """
    Détermine votre profil d'endurance par régression (Courbe de Puissance) 
    et prédit le temps pour une nouvelle distance.

    Args:
        df_records (pd.DataFrame): DataFrame des records historiques (Distance_km, Time_hours).
        new_distance_km (float): Distance horizontale de la nouvelle course.
        distance_type (str): 'route' ou 'trail'. Si 'trail', le D+ est inclus.
        d_plus_m (int): Dénivelé positif (D+) en mètres pour la nouvelle course (si trail).

    Returns:
        float: Temps prédit en heures.
    """
    
    # === ÉTAPE 1: PRÉPARATION DES DONNÉES ET TRANSFORMATION LOG ===
    
    # Calcul de la vitesse moyenne (V = D / T)
    df_records['vitesse_km_h'] = df_records['distance_km'] / (df_records['best_time_min'] / 60)
    
    # Application de la transformation logarithmique (Log-Log Plot)
    df_records['log_D'] = np.log(df_records['distance_km'])
    df_records['log_V'] = np.log(df_records['vitesse_km_h'])
    
    # === ÉTAPE 2: RÉGRESSION LINÉAIRE (Détermination de votre profil 'a' et 'b') ===
    
    # Régression: log(V) = a - b * log(D)
    # Dans scipy.stats.linregress, nous obtenons la pente et l'ordonnée à l'origine (intercept).
    # La pente est -b, l'intercept est a.
    slope, intercept, r_value, p_value, std_err = linregress(
        df_records['log_D'], 
        df_records['log_V']
    )
    
    a = intercept
    b = -slope # L'exposant b doit être positif, car la vitesse diminue quand la distance augmente.
    
    print(f"--- Profil d'Endurance Personnalisé ---")
    print(f"Coefficient 'a' (Vitesse maximale théorique): {a:.4f}")
    print(f"Coefficient 'b' (Facteur de Dégradation): {b:.4f}")
    print(f"Qualité du Fit (R²): {r_value**2:.4f}")
    print("---------------------------------------")

    df_records['log_V_pred'] = a + slope * df_records['log_D']

    # 1. Crée la figure Matplotlib
    fig, ax = plt.subplots(figsize=(8, 5))
    
    # 2. Utilise regplot de Seaborn
    # 'regplot' trace les points de données et la ligne de régression linéaire.
    # Il calcule la régression directement entre 'log_D' et 'log_V'.
    sns.regplot(
        x='log_D', 
        y='log_V', 
        data=df_records, 
        ax=ax,
        ci=95, # Intervalle de confiance à 95% (l'ombre bleue autour de la droite)
        scatter_kws={'color': 'blue', 'alpha': 0.8},
        line_kws={'color': 'red', 'label': f'R²={r_value**2:.2f}'}
    )
    
    # 3. Ajouter les statistiques calculées dans le titre ou les étiquettes
    ax.set_title(
        f"Courbe de Puissance (Log-Log Plot)\n"
        f"Profil: ln(V) = {a:.4f} - {b:.4f} * ln(D)"
    )
    ax.set_xlabel("Logarithme de la Distance (ln(D))")
    ax.set_ylabel("Logarithme de la Vitesse (ln(V))")
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.6)
    
    # Ajoutez le point de prédiction (Étape 3) si vous le souhaitez
    
    
    
    # === ÉTAPE 3: PRÉDICTION ===
    
    # 1. Calcul de log(D) pour la nouvelle distance
    if new_denivele_pos is not None:
        new_distance_itra = new_distance_km + (new_denivele_pos/100)
    else:
        new_distance_itra = new_distance_km

    log_D_new = np.log(new_distance_itra)
    
    # 2. Prédiction de log(V)
    log_V_pred = a - b * log_D_new # Vitesse en km/h
    
    # 3. Inversion du logarithme pour obtenir Vitesse_pred
    V_pred_kmh = np.exp(log_V_pred)
    
    # 4. Calcul du Temps (T = D / V)
    Time_pred_hours = new_distance_itra / V_pred_kmh
    
    return Time_pred_hours, fig

@st.cache_data
def process_activity(df_raw):

    # Suppression des colonnes avec que des valeurs manquantes
    df_raw.dropna(axis='columns', how='all', inplace=True)

    # On supprime les temps de repos
    df_raw = df_raw[df_raw['resting'] == False]

    # Supprimer les premières lignes où moving == False car pas de détection de mouvement par la montre
    premier_idx_moving = df_raw['moving'].idxmax()
    df_raw = df_raw.loc[premier_idx_moving:]

    # Remettre à zéro les colonnes 'temps_relatif_sec' et 'distance_m'
    df_raw['temps_relatif_sec'] -= df_raw['temps_relatif_sec'].min()
    df_raw['distance_m'] -= df_raw['distance_m'].min()
    

    # Application de la normalisation pour le temps et la distance
    df_raw['temps_reel_s'], df_raw['temps_normalisee'] = normalisation_data(df_raw, 'temps_relatif_sec') # en secondes
    df_raw['distance_reelle_m'], df_raw['distance_normalisee'] = normalisation_data(df_raw,'distance_m') # en mètres
    
    # Conversion des distances et temps dans les différentes unités possibles
    df_raw['temps_h'] = df_raw['temps_reel_s'] / 3600
    df_raw['temps_min'] = df_raw['temps_reel_s'] / 60
    df_raw['distance_km'] = df_raw['distance_reelle_m'] / 1000

    # Calcul de la vitesse lissee en km_h
    df_raw['vitesse_km_h'] = df_raw['vitesse_lissee'] * 3.6

    df_raw ['vitesse_km_h'] = np.where(np.isinf(60 / df_raw['vitesse_km_h']), np.nan, df_raw['vitesse_km_h'])

    # Gestion des valeurs extrêmes pour la vitesse et la pente
    df_raw['vitesse_km_h_lissee'] = drop_extreme_value(df_raw, feature='vitesse_km_h', FENETRE_LISSAGE=5)
    df_raw['pente_lissee'] = drop_extreme_value(df_raw, feature='pente_lissee', FENETRE_LISSAGE=5)

    # Calcul de l'allure en min/km
    allure_min_km = 60 / df_raw['vitesse_km_h_lissee']
    df_raw['allure_min_km'] = allure_format(allure_min_km)

    # Fréquence cardiaque : on établie des zones de FC
    if 'frequence_cardiaque' in df_raw.columns.tolist():
        df_raw['fc_normalisee'] = df_raw['frequence_cardiaque'] / (200) * 100
        bins_fc = [0, 60, 68, 75, 82, 89, 94, 100] #modèle scientifique 7 zones
        labels_fc = []
        for i in range(len(bins_fc) - 1):
            start = bins_fc[i]
            end = bins_fc[i+1]
            zone_names = ['Récup', 'End. Base', 'End. Fond.', 'Tempo', 'Seuil', 'VO2 Max', 'Effort Max']
            label = f"({start} - {end}% FC Max) {zone_names[i]}"
            labels_fc.append(label)
        df_raw['zone_fc'] = pd.cut(x=df_raw['fc_normalisee'], bins=bins_fc, labels=labels_fc)
    else:
        pass # pas de données de FC

    # Puissance : on établie des zones de Puissance en fonction de la FTP
    if 'puissance_watts' in df_raw.columns.tolist():
        FTP_value = 250 # à modifier
        df_raw['puissance_normalisee'] = df_raw['puissance_watts'] / FTP_value * 100
        bins_puissance = [0, 55, 75, 90, 105, 120, 150, np.inf] #modèle scientifique 7 zones
        labels_puissance = []
        for i in range(len(bins_puissance) - 1):
            start = bins_puissance[i]
            end = bins_puissance[i+1]
            zone_names = ['Récup', 'Endurance', 'Tempo', 'Seuil', 'VO2 Max','Capacité Anaérobie', 'Effort Max']
            if end == np.inf:
                end_str = 'Max'
                label = f"({start}% to {end_str} FTP) {zone_names[i]}"
            else:
                label = f"({start} to {end}% FTP) {zone_names[i]}"
            labels_puissance.append(label)
        df_raw['zone_puissance'] = pd.cut(x=df_raw['puissance_normalisee'], bins=bins_puissance, labels=labels_puissance)
    else:
        pass # pas de données de Puissance


    # Application de la coupe des données pour la distance et la pente_lissee
    

    df_raw['tranche_distance'] = cutting_data_percent(df=df_raw, feature='distance_normalisee')
    df_raw['tranche_pente'] = cutting_data_percent(df=df_raw, feature='pente_lissee', min_list=-50, max_list=50)
    
    # Calcul de l'efficacité de course
    df_raw['efficacite_course'] = df_raw['vitesse_km_h_lissee'] / df_raw['frequence_cardiaque']
    df_raw['efficacite_course_normalisee'] = min_max_scaler(df_raw['efficacite_course'])

    # Calcul de la VAM VItesse Ascensionnelle en Montée
    df_raw['vam'] = (df_raw['altitude_m'].diff() / df_raw['temps_h'].diff()).fillna(0)

    # Calcul de l'Allure Ajustée selon la Pente
    df_raw['vap_allure'] = calculate_vap(df_raw['allure_min_km'],df_raw['pente_lissee'])

    # Calcul de la Différence de vitesse
    df_raw['diff_allure'] = df_raw['allure_min_km'] - df_raw['vap_allure']

    #Calcul du dénivelé positif et négatif
    df_raw['d_pos_diff'], df_raw['d_neg_diff'], df_raw['d_pos_cum'], df_raw['d_neg_cum'] = calculer_denivele(df_raw['altitude_m'])

    # Calcul d'indicateur de la sortie
    ratio_denivele_distance = df_raw['d_pos_cum'].max() / df_raw['distance_km'].max()
    km_effort_itra = np.round(df_raw['distance_km'].max() + (df_raw['d_pos_cum'].max() / 100),1)
    km_effort_611 = np.round(df_raw['distance_km'].max() + (df_raw['d_pos_cum'].max() * 6.11 / 1000),1)

    # Calcul de la distance d'effort avec formule itra basique 100m d+ = 1km en + d'effort
    df_raw['distance_effort_itra'] = df_raw['distance_km'] + (df_raw['d_pos_cum'] / 100)

    # Calcul du temps total au format hh:mm:ss
    temps_total_formatte = conversion_temps_total(df_raw['temps_h'], df_raw['temps_min'])

    # surface --> 0 route et 1--> chemin
    df_raw['surface'].replace({0: 'road', 1: 'trail'}, inplace=True)

    # étude des outliers

    # Remettre à jour l'indexation du df
    df_raw.reset_index(drop=True,inplace=True)

    # Supprimer les colonnes inutiles
    df_raw.drop(columns=['vitesse_lissee','vitesse_km_h','distance_m','latlng','resting','outlier'], inplace=True)
    df_raw.dropna(axis='columns', how='all', inplace=True)

    return df_raw, km_effort_itra, km_effort_611, temps_total_formatte, ratio_denivele_distance  


def get_format(feature, aggfunc):
    """Retourne le format pour le crosstab"""
    if feature in ['vitesse_km_h_lissee','vap_vitesse','diff_vitesse']:
        return '.1f'
    elif feature in ['allure_min_km','efficacite_course','vap_allure']:
        return '.2f'
    elif aggfunc is coefficient_variation:
        return '.2f'
    return 'd'  

def crosstab(df, feature, aggfunc, vmin=None,vmax=None,):
    fmt_heatmap = get_format(feature,aggfunc)
    dtype_final = float if 'f' in fmt_heatmap else int
    fig, ax = plt.subplots(figsize=(15,5))
    crosstab = pd.crosstab(df['tranche_distance'], df['tranche_pente'], df[feature],aggfunc=aggfunc).fillna(0).astype(dtype_final)
    sns.heatmap(crosstab, annot=True, cmap='viridis', linewidths=0.5, fmt=fmt_heatmap, cbar=True, ax=ax, vmin=vmin, vmax=vmax,
                annot_kws={'fontsize': 12})
    st.pyplot(fig)
    plt.close(fig)

def time_formatter(x, pos=None):
    """
    Formateur Matplotlib et Streamlit : Convertit les minutes décimales (float) en format MM:SS.
    C'est la fonction qui permet d'afficher 04:45 pour 4.75.
    """
    tps = x
    heures = int(tps // 60)
    minutes = int(np.floor(tps % 60))
    seconds = int(np.floor((tps % 1) * 60))
    
    # Gérer le cas où l'arrondi fait passer les secondes à 60
    if seconds == 60:
        minutes += 1
        seconds = 0
        
    return f"{heures:02d}:{minutes:02d}:{seconds:02d}"

@st.cache_data
def detection_montees(df, feature_altitude, window_rolling=90):
        """Détection des montées, descentes et plats"""
        altitude_lissee = df[feature_altitude].rolling(window=window_rolling, center=True).mean()
        diff_altitude_lissee = altitude_lissee.diff()
        montee = np.where(diff_altitude_lissee > 0.01, 1, np.where(diff_altitude_lissee < 0.01, -1, 0))
        return montee

@st.cache_data   
def plot_montees(df, feature_distance, feature_altitude, var_montee):
    """Trace la courbe d'altitude et celles des montées pour voir si la détection est bonne"""
    fig,ax1 = plt.subplots()
    sns.lineplot(data=df, x=feature_distance, y=feature_altitude, ax=ax1)
    ax2 = ax1.twinx()
    sns.lineplot(data=df, x=feature_distance, y=var_montee, ax=ax2, color='tab:red')
    return fig

st.title("🏃‍♂️ Analyse d'Activité Strava")

# --- Configuration de la barre latérale pour l'analyse ---
st.sidebar.header("Configuration de l'activité")

try:
    # Utilisation de la version cachée
    recent_activities = get_last_activity_ids_cached(200)
except Exception as e:
    st.error(f"Erreur lors de la récupération des activités récentes via l'API Strava : {e}")
    recent_activities = []
    
activity_options = {f"{act['name']}": act['id'] for act in recent_activities}
activity_options = {'Sélectionner une activité': None} | activity_options | {'Saisir un autre ID': 'manual'}

selected_option = st.sidebar.selectbox("Sélectionnez une activité récente (1) :", list(activity_options.keys()), key="select_act_1")



activity_id_input = None
if activity_options[selected_option] == 'manual':
    activity_id_input = st.sidebar.text_input("Entrez l'ID de l'activité (1)", '', key="input_act")
else:
    activity_id_input = activity_options[selected_option]
    

# Bouton de chargement (déclenche le processus)
st.sidebar.markdown("---")

# Utilisation d'un conteneur pour les messages de chargement
status_container = st.empty()

if st.sidebar.button("🚀 Charger l'activité"):
    
    if not activity_id_input:
        status_container.warning("Veuillez sélectionner ou entrer l'ID de la première activité.")

    activity_id = None
    try:
        activity_id = int(activity_id_input)
    except ValueError:
        status_container.error("L'ID de l'activité doit être un nombre entier.")

    # 1. Traitement de l'activité 1 (Chargement brut et mise en cache)
    try:
        # Utilisation de la version cachée de l'API
        df_raw, activity_name, sport_type, activity_date = get_activity_data_from_api_cached(activity_id)    

        df_raw, km_effort_itra, km_effort_611, temps_total_formatte, ratio_denivele_distance = process_activity(df_raw)
        
        
        
        if df_raw.empty:
            status_container.warning(f"L'activité **'{activity_name}'** n'a pas de données de stream ou est manuelle. Analyse impossible.")    
        
        # Stockage des résultats traités en session state
        st.session_state['df_raw'] = df_raw
        st.session_state['activity_name'] = activity_name
        st.session_state['sport_type'] = sport_type
        st.session_state['activity_id'] = activity_id
        st.session_state['activity_date'] = activity_date
        status_container.success(f"Données de l'activité **{activity_name}** chargées et traitées avec succès!")
        
    except Exception as e:
        status_container.error(f"❌ Erreur critique lors du chargement/traitement de l'activité {activity_id} : {e}")
        
if 'df_raw' in st.session_state:
    df_raw = st.session_state['df_raw']
    activity_name = st.session_state['activity_name']
    sport_type = st.session_state['sport_type']
    activity_date = st.session_state['activity_date']
    st.header(f"Activité Principale : **{activity_name}**")

    sport_icon_map = {'Run': '🏃‍♂️', 'TrailRun': '⛰️', 'Ride': '🚴‍♂️', 'Hike': '🚶‍♂️'}
    sport_icon = sport_icon_map.get(sport_type, '❓')
    st.markdown(f"**Type d'activité :** *{sport_type}* {sport_icon}")
    
    
    ####################################################

    col1, col2, col3, col4, col5, col6 = st.columns(6)
    with col1:
        st.metric("Distance", value=f"{np.round(df_raw['distance_km'].max(),1)} km", border=True)
    with col2:
        st.metric("Dénivelé +", value=f"{int(df_raw['d_pos_cum'].max())}m", border=True)
    with col3:
        st.metric("Temps", value=f"{time_formatter(df_raw['temps_min'].max())}", border=True)
    with col4:
        st.metric("Allure Moyenne", value=f"{allure_format(df_raw['allure_min_km'].mean())}", border=True)
    with col5:
        st.metric("VAP Moyenne", value=f"{allure_format(df_raw['vap_allure'].mean())}", border=True, help="Allure Ajustée à la Pente")
    with col6:
        st.metric("FC Moyenne", value=f"{int(df_raw['frequence_cardiaque'].mean())} bpm", border=True)

    st.divider()
    
    with st.expander("Détection des montées et des descentes"):
        window_rolling = st.slider("Fenêtre pour le lissage des données d'altitude", value=90, min_value=5, max_value=200)
        df_raw['segment'] = detection_montees(df_raw, feature_altitude='altitude_m',window_rolling=window_rolling)
        fig_montees = plot_montees(df_raw,'distance_km','altitude_m','segment')
        st.pyplot(fig_montees)
        plt.close(fig_montees)    

    df_raw['segment'].replace({1 : 'montées', -1 : 'descente', 0 : 'plat'}, inplace=True)  

    col1, col2, col3, col4, col5, col6 = st.columns(6)
    with col1:
        st.metric("Vitesse Asc", f"{int(df_raw[df_raw['segment'] == 'montées']['vam'].mean())} m/h", border=True)
    with col2:
        st.metric("Vitesse Desc", f"{int(df_raw[df_raw['segment'] == 'descente']['vam'].mean())} m/h", border=True)
    with col3:
        st.metric("Pente moy en montée", f"{int(df_raw[df_raw['segment'] == 'montées']['pente_lissee'].mean())} °", border=True)
    with col4:
        st.metric("Pente max en montée", f"{int(df_raw[df_raw['segment'] == 'montées']['pente_lissee'].max())} °", border=True)
    with col5:
        st.metric("FCmoy en montée", f"{int(df_raw[df_raw['segment'] == 'montées']['frequence_cardiaque'].mean())} bpm", border=True)
    with col6:
        st.metric("FCmoy en descente", f"{int(df_raw[df_raw['segment'] == 'descente']['frequence_cardiaque'].mean())} bpm", border=True)
    
    st.divider()
    

    list_col_all = df_raw.columns.tolist()
    list_col_num = df_raw.select_dtypes([int,float]).columns.tolist()
    list_col_cat = df_raw.select_dtypes([object,'category']).columns.tolist()

    with st.expander("Etude de la Fréquence Cardiaque"):
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


    st.divider()

    with st.expander("Etude de la Puissance (vélo)"):
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


  
    st.divider()

    with st.expander("Etude heatmap"):
        st.subheader("Paramètres")
        col1, col2, col3 = st.columns(3)
        with col1:
            feature_option = st.selectbox('Indicateur à étudier',options=[None, *list_col_all])
        with col2:
            vmin_option = st.number_input("Quelle valeur min souhaites tu mettre?", value=None, step=1)
        with col3:
            vmax_option = st.number_input("Quelle valeur max souhaites tu mettre?", value=None, step=1)
        aggfunc_dict = {
            'Moyenne' : np.mean,
            'Ecart_Type' : np.std,
            'Somme': np.sum,
            'Median': np.median,
            'Minimum' : np.min,
            'Maximum': np.max,
            'Coefficient_de_Variation': coefficient_variation
        }

        if feature_option is None:
            st.info("Sélectionner une variable")
        else:
            aggfunc_option = st.radio("Quelle fonction veux tu exécuter", options=list(aggfunc_dict.keys()))
            aggfunc = aggfunc_dict[aggfunc_option]
            st.subheader(f"{aggfunc_option} de la variable '{feature_option}' en fonction de la distance et de la pente_lissee")
            crosstab(df_raw,feature_option,aggfunc=aggfunc, vmin=vmin_option, vmax=vmax_option)
     
    st.divider()
    with st.expander("BoxPlot"):
        col_var_x, col_var_y = st.columns(2)
        with col_var_x:
            var_x = st.selectbox("Variable en abscisse", options=[None] + list_col_cat)
        with col_var_y:
            var_y = st.selectbox("Variable en ordonnée", options=[None] + list_col_num)
        var_hue = None
        if st.checkbox("Souhaites tu une troisième variable pour différentes couleurs?", key="var_hue_boxplot_option"):
            var_hue = st.selectbox("Variable couleur", options=[None] + list_col_cat, key="var_hue_boxplot")
        st.subheader(f"Boxplot de la variable {var_y} en fonction de la catégorie {var_x}")
        if var_x is None or var_y is None:
            st.info("Sélectionne les deux variables pour le graphique")
        elif var_hue is not None:
            plot_boxplot(df_raw,var_x,var_y, var_hue)
        else:
            plot_boxplot(df_raw,var_x,var_y)

    st.divider()

    ## Joint Plot

    with st.expander("Joint Plot"):
        
        col_var_x, col_var_y = st.columns(2)
        with col_var_x:
            var_x = st.selectbox("Variable en abscisse", options=[None] + list_col_num, key='var_x_joint_plot')
            list_col_num_var_y = [col for col in list_col_num if col != var_x ]
        with col_var_y:
            var_y = st.selectbox("Variable en ordonnée", options=[None] + list_col_num_var_y, key='var_y_joint_plot')
        var_hue = None
        if st.checkbox("Souhaites tu une troisième variable pour différentes couleurs?", key="var_hue_jointplot_option"):
            var_hue = st.selectbox("Variable couleur", options=[None] + list_col_cat, key="var_hue_jointplot")
        st.subheader(f"Joint Plot de la variable {var_y} en fonction de la catégorie {var_x}")
        if var_x is None or var_y is None:
            st.info("Sélectionne les deux variables pour le graphique")
        elif var_hue is not None:
            plot_jointplot(df_raw, var_x, var_y,var_hue)
        else:
            plot_jointplot(df_raw, var_x, var_y)
    
    
    
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
    
    
    # Prédicteur de course

    st.divider()
    with st.expander("Prédicteur de temps de course"):
        st.subheader("Prédicteur de temps de course")
        col_distance, col_denivele = st.columns(2)
        with col_distance:
            new_distance = st.number_input("Nouvelle distance à prédire", key="New_distance", value=None)
        with col_denivele:
            new_denivele = st.number_input("Quel dénivelé positif sur la course à prédire", key="New_dénivelé_pos", value=None)
        if new_distance is None:
            st.info("Rentre la nouvelle distance à prédire")
        else:
            new_time, fig_reg = fit_and_predict_time(df_record_per_distance, new_distance, new_denivele)
            st.metric(label="Temps prédit",value=time_formatter(new_time * 60))
            with st.expander("Veux tu regarder la courbe de Puissance?"):
                st.pyplot(fig_reg)
                plt.close(fig_reg)

        # Prendre la valeur la plus basse pour une meme distance
        
        
        st.subheader('Meilleure performance historique par distance')
        fig,ax = plt.subplots()
        sns.regplot(data=df_record_per_distance, x='distance_km', y='best_time_min', ci=95, ax=ax, order=2)
        formatter = ticker.FuncFormatter(time_formatter)
        # Applique le formateur à l'axe Y
        ax.yaxis.set_major_formatter(formatter)
        plt.ylabel('Temps_min')
        st.pyplot(fig)
        plt.close(fig)

    st.divider()
    st.subheader("Evolution des performances")

    df_sql = sql_df()
    df_sql['year'] = pd.to_datetime(df_sql['activity_start_date']).dt.year
    df_sql['month'] = pd.to_datetime(df_sql['activity_start_date']).dt.month
    df_sql['week'] = pd.to_datetime(df_sql['activity_start_date']).dt.isocalendar().week
    df_sql['delta_day'] = datetime.now(timezone.utc) - pd.to_datetime(df_sql['activity_start_date'])
    df_sql['delta_day'] = df_sql['delta_day'].dt.days

    def agg_sql_df_period(df, period, feature, sport_type_list):

        df_sport_type = df[df['sport_type'].isin(sport_type_list)]

        df_agg = df_sport_type.groupby(by=period)[feature].sum().reset_index()

        cmap = plt.cm.viridis # Vous pouvez choisir 'viridis', 'plasma', 'magma', etc.

        # 2. Normaliser les données de distance pour les faire correspondre aux couleurs
        norm = mcolors.Normalize(
            vmin=df_agg[feature].min(),
            vmax=df_agg[feature].max()
        )
        scalar_mappable = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
        scalar_mappable.set_array(df_agg[feature])

        st.caption(f"Evolution de {feature} par {period}")

        fig, ax = plt.subplots()
        bars = sns.barplot(data=df_agg, x=period, y=feature, ax=ax)
        for i, bar in enumerate(bars.patches):
            # Récupérer la valeur de distance correspondante
            distance_value = df_agg[feature].iloc[i]
            # Appliquer la couleur basée sur la normalisation
            bar.set_color(scalar_mappable.to_rgba(distance_value))
        dict_month = {1: 'Janv.', 2: 'Févr.', 3: 'Mars', 4: 'Avril', 5: 'Mai', 6: 'Juin', 7: 'Juil.', 8: 'Août', 
                     9: 'Sept.', 10: 'Oct.', 11: 'Nov.', 12: 'Déc.'}
        month_number = df['month'].unique().tolist()
        month_labels = [dict_month[m] for m in month_number]
        
        if period == 'week':
            ax.xaxis.set_major_locator(MultipleLocator(3))
        elif period =='month':
            ax.set_xticklabels(month_labels, rotation=45, ha='right')


        # 5. Ajouter la barre de couleur (Colorbar)
        cbar = fig.colorbar(scalar_mappable, ax=ax, orientation='vertical', pad=0.03)
        return fig
    
    list_of_sport_type = df_sql['sport_type'].unique().tolist()
    sport_type_options = st.multiselect("Sélectionne le ou les sports que tu souhaites voir l'évolution?", 
                                        options=list_of_sport_type, default=['Hike','Run','TrailRun'])
    
    
    col_dist_week, col_d_pos_week = st.columns(2)
    with col_dist_week:
        fig = agg_sql_df_period(df_sql, period='week', feature='total_distance_km', sport_type_list=sport_type_options)
        st.pyplot(fig)
        plt.close(fig)
    with col_d_pos_week:
        fig = agg_sql_df_period(df_sql, period='week', feature='total_elevation_gain_m', sport_type_list=sport_type_options)
        st.pyplot(fig)
        plt.close(fig)

    col_dist_month, col_d_pos_month = st.columns(2)
    with col_dist_month:
        fig = agg_sql_df_period(df_sql, period='month', feature='total_distance_km', sport_type_list=sport_type_options)
        st.pyplot(fig)
        plt.close(fig)
    with col_d_pos_month:
        fig = agg_sql_df_period(df_sql, period='month', feature='total_elevation_gain_m', sport_type_list=sport_type_options)
        st.pyplot(fig)
        plt.close(fig)

    


    

    

        
else:
    st.info("Veuillez sélectionner ou entrer un ID d'activité et cliquer sur **'🚀 Charger l'activité'** pour commencer l'analyse.")