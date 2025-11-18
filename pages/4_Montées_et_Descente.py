# Montées_et_Descente.py

import streamlit as st
import matplotlib.pyplot as plt
from utils.plotting import plot_montees
from utils.data_processing import detection_montees

st.title("Etude des phases de montées et de descentes")

if 'df_raw' in st.session_state:
    df_raw = st.session_state['df_raw']
    activity_name = st.session_state['activity_name']
    sport_type = st.session_state['sport_type']
    activity_date = st.session_state['activity_date']

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
    
else:
    st.info("Veuillez sélectionner ou entrer un ID d'activité et cliquer sur **'🚀 Charger l'activité'** pour commencer l'analyse.")