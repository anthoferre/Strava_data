import streamlit as st


def inject_custom_css():
    st.markdown(f"""
        <style>

        /* 2. Labels des widgets (boutons radio, sliders, selectbox, etc.) */
        .stWidgetLabel p, label {{
            color: #FC4C02 !important;
        }}

        /* Colorer tout le texte (paragraphes) à l'intérieur des conteneurs markdown */
        [data-testid="stMainBlockContainer"] p {{
            color: #FC4C02;
        }}

        [data-testid="stMainBlockContainer"] p {{
            color: #FC4C02;
        }}

        [data-testid="stSidebarNavItems"] span {{
            color: #FC4C02 !important;
            font-weight: 600; /* Optionnel : pour un look plus "Strava" */
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
