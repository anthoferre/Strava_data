# plotting.py

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import streamlit as st

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

@st.cache_data   
def plot_montees(df, feature_distance, feature_altitude, var_montee):
    """Trace la courbe d'altitude et celles des montées pour voir si la détection est bonne"""
    fig,ax1 = plt.subplots()
    sns.lineplot(data=df, x=feature_distance, y=feature_altitude, ax=ax1)
    ax2 = ax1.twinx()
    sns.lineplot(data=df, x=feature_distance, y=var_montee, ax=ax2, color='tab:red')
    return fig