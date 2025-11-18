# data_processing.py

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import linregress  

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



def min_max_scaler(feature):
    vmin = 0
    vmax = 30/200 #30km/h pour 200bpm
    valeur_normalisee = (feature-vmin) / (vmax-vmin) * 100
    return valeur_normalisee



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
def detection_montees(df, feature_altitude, window_rolling=90):
        """Détection des montées, descentes et plats"""
        altitude_lissee = df[feature_altitude].rolling(window=window_rolling, center=True).mean()
        diff_altitude_lissee = altitude_lissee.diff()
        montee = np.where(diff_altitude_lissee > 0.01, 1, np.where(diff_altitude_lissee < 0.01, -1, 0))
        return montee


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
