# Etude_Graphique.py

import streamlit as st
import numpy as np
from utils.plotting import plot_boxplot, plot_jointplot, coefficient_variation, crosstab

st.title("Visualisation des indicateurs de la performance")

if 'df_raw' in st.session_state:
    df_raw = st.session_state['df_raw']
    activity_name = st.session_state['activity_name']
    sport_type = st.session_state['sport_type']
    activity_date = st.session_state['activity_date']

    list_col_all = df_raw.columns.tolist()
    list_col_num = df_raw.select_dtypes([int,float]).columns.tolist()
    list_col_cat = df_raw.select_dtypes([object,'category']).columns.tolist()


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

else:
    st.info("Veuillez sélectionner ou entrer un ID d'activité et cliquer sur **'🚀 Charger l'activité'** pour commencer l'analyse.")