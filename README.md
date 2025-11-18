# 📈 Projet d'Analyse et de Prédiction de Performances Strava

## 🚀 Vue d'ensemble du Projet

Ce dépôt contient le code source d'un projet personnel de Data Science visant à analyser les données d'activités Strava afin d'identifier les facteurs clés de performance et permettre la **prédiction de temps de course** à partir des données historiques.

L'objectif est d'analyser au mieux la performance, d'exploiter des indicateurs clés pour l'optimisation de l'entraînement.

## ✨ Fonctionnalités Clés à travers les différentes pages de l'application

* **Extraction de données :** Connexion sécurisée à l'API Strava.
* **Feature Engineering avancé :** Création et détection de métriques clés.
* **Visualisation de la performance :** Rapports détaillés des analyses d'activités.
* **Prédiction temps de course :** A partir de l'historique du coureur, on a une estimation de son temps de course sur de nouvelles données.

---

## ⚙️ Installation et Mise en Place

Suivez ces étapes pour configurer et exécuter le projet localement.

### 1. Cloner le dépôt

```bash
git clone [https://github.com/anthoferre/Strava_data.git](https://github.com/anthoferre/Strava_data)
cd Strava_data
```

### 2.Configuration des Clés API (Fichier .env)

Pour se connecter à l'API Strava, vous devez fournir vos identifiants dans un fichier d'environnement local. Ce fichier n'est pas inclus dans le dépôt pour des raisons de sécurité (il est listé dans le `.gitignore`).

Obtenez vos clés : Rendez-vous sur [Strava Developers] pour enregistrer une application et obtenir votre Client ID et votre Client Secret.

Créer le fichier `.env` : À la racine du projet, créez un fichier nommé `.env` et ajoutez-y les lignes suivantes, en remplaçant les placeholders par vos identifiants réels :

```
# .env
# Ce fichier est ignoré par Git et ne doit JAMAIS être partagé publiquement.
STRAVA_CLIENT_ID="VOTRE_CLIENT_ID_ICI"
STRAVA_CLIENT_SECRET="VOTRE_CLIENT_SECRET_ICI"
```
Le projet utilise la bibliothèque `python-dotenv` pour charger ces variables dans l'environnement lors de l'exécution du script.

Vérification : Le projet utilise la bibliothèque python-dotenv pour charger ces variables dans l'environnement lors de l'exécution du script.

### 3. 💻 Exécution du Projet

Une fois la configuration terminée, vous pouvez lancer les scripts d'analyse :
```
Bash

# Lancer le script de récupération des précédentes données
python data_fetcher.py

# Lancer le script pour lancer l'application streamlit du modèle
python model_training.py
```
