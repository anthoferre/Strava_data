#Prédiction.py

import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as ticker
from utils.data_processing import fit_and_predict_time, time_formatter, calculate_all_records
from db_manager import init_db, load_activity_records_by_key, load_performance_records, save_performance_records

st.title("Prédiction des prochains temps de course")

if 'df_raw' in st.session_state:
    df_raw = st.session_state['df_raw']
    activity_name = st.session_state['activity_name']
    sport_type = st.session_state['sport_type']
    activity_date = st.session_state['activity_date']

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

    