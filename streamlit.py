# Importation des bibliothèques
import streamlit as st
import pandas as pd
import altair as alt
import matplotlib.pyplot as plt
from urllib.request import urlopen
import json
import numpy as np
import pickle
from sklearn.neighbors import NearestNeighbors
import random
import requests
import shap

# Titre de l'application
st.title('Implémentez un modèle de scoring')

# 1. Importation du dataframe et du modèle

# Lecture du fichier CSV
df = pd.read_csv("df_tabdashboard.csv", usecols=lambda col: col != 'TARGET', nrows=int(0.1 * pd.read_csv("df_tabdashboard.csv").shape[0]))  # Sélection de 10% des clients
liste_id = df['SK_ID_CURR'].tolist()

with open('model_streamlit.pkl', 'rb') as file1:
    model1 = pickle.load(file1)
    
with open('model_KNN_streamlit.pkl', 'rb') as file2:
    model2 = pickle.load(file2)

st.subheader('Prédiction de notre modèle')

# Barre de recherche
search_input = st.text_input("Entrez l'identifiant du client")

# Initialisation de filtered_df avec une valeur par défaut
filtered_df = pd.DataFrame()

# Vérifier si la valeur de search_input est vide ou non numérique
if search_input != "" and search_input.isdigit():
    # Recherche du client dans le dataframe
    filtered_df = df[df['SK_ID_CURR'] == int(search_input)]

    # Sélection des colonnes souhaitées
    selected_columns = ['SK_ID_CURR', 'Sexe', 'Revenus annuels', 'Revenus totaux', 'Somme des crédits', "Taux d’endettement", 'Propriétaire']
    filtered_df = filtered_df[selected_columns]

# Affichage du résultat
if len(filtered_df) == 0:
    st.write("Erreur ou absence identifiant")
else:
    st.write("Résultat de la recherche :")
    st.write(filtered_df)

    # Affichage du ratio du client en pourcentage avec une mise en forme personnalisée
    payment_rate = filtered_df["Taux d’endettement"].values[0] * 100
    st.write(f"<div style='display: flex; align-items: center; font-size: 15px;'>Endettement du client : <span style='font-size: 20px; font-weight: bold;'>{payment_rate}%</span></div>", unsafe_allow_html=True)


# Appel de l'API :
API_url = "https://buriez-flaskp7.herokuapp.com/api/predict"

if search_input:
    client_id = int(search_input) if search_input.isdigit() else None

    if client_id:
        # Effectuer la requête POST vers l'API
        response = requests.post(API_url, data={'client_id': client_id})

        # Vérifier le statut de la réponse
        if response.status_code == 200:
            # Obtenir les données JSON de la réponse
            API_data = response.json()

            # Vérifier si la prédiction existe dans les données renvoyées
            if 'prediction' in API_data:
                classe_predite = API_data['prediction']
                if classe_predite == 1:
                    etat = 'client à risque'
                elif classe_predite == 0:
                    etat = 'client peu risqué'
                else:
                    etat = 'Client non reconnu dans notre API'

                # Afficher le résultat
                st.markdown(f"<div style='border: 1px solid black; padding: 10px; text-align: center;'><p style='font-size: 25px; font-weight: bold;'>Prédiction : {etat}</p></div>", unsafe_allow_html=True)
            # Vérifier si une erreur est renvoyée dans les données
            elif 'error' in API_data:
                error_message = API_data['error']
                st.write(f"Erreur : {error_message}")
        else:
            # Gérer les erreurs de requête
            st.write("Erreur lors de la requête à l'API")

## Interprétabilité de la prédiction ##
# En fonction de l'idenfiant présent dans le panel, nous allons connaître l'impact des variables #

st.write('## Interprétabilité du résultat')                            
shap.initjs()   

# Récupération des données d'entrée X
if search_input and search_input.isdigit():
    X = df[df['SK_ID_CURR'] == int(search_input)]
else:
    X = pd.DataFrame()  # Créez un DataFrame vide

# Vérification si les données d'entrée ne sont pas vides
if not X.empty:
    # Calcul des valeurs SHAP
    explainer = shap.TreeExplainer(model1)
    shap_values = explainer.shap_values(X)

    # Affichage du texte expliquant la dépendance de la prédiction avec l'identifiant
    st.write("L'interprétation de la prédiction peut varier en fonction de l'identifiant du client.")

    # Appel de l'API pour chaque individu dans X
    for i in range(len(X)):
        client_id = X.iloc[i]['SK_ID_CURR']
        response = requests.post(API_url, data={'client_id': client_id})

        # Vérifier le statut de la réponse
        if response.status_code == 200:
            API_data = response.json()

            # Vérifier si la prédiction existe dans les données renvoyées
            if 'prediction' in API_data:
                classe_predite = API_data['prediction']
                if classe_predite == 1:
                    etat = 'client à risque'
                elif classe_predite == 0:
                    etat = 'client peu risqué'
                else:
                    etat = 'Client non reconnu dans notre API'

                # Afficher le résultat de l'interprétation pour chaque individu
                st.markdown(f"<p>Interprétation pour le client avec l'identifiant {client_id} : {etat}</p>", unsafe_allow_html=True)

    # Affichage du graphique SHAP
    fig, ax = plt.subplots(figsize=(5, 5))
    shap.summary_plot(shap_values, features=X, plot_type='bar', max_display=10, color_bar=False, plot_size=(10, 10))            
    st.pyplot(fig)
else:
    st.write("Les données d'entrée sont vides.")


# Préparation données pour graphique

# Vérifier si filtered_df est vide avant de l'utiliser pour préparer les données pour le graphique
if not filtered_df.empty:
    # Calculer le nombre de clients avec un crédit inférieur à celui du client recherché
    client_credit = filtered_df['Somme des crédits'].values[0]
    client_annuity = filtered_df["Taux d’endettement"].values[0]
    lower_credit_clients_count = len(df[df['Somme des crédits'] < client_credit])
    # Calculer le nombre de clients avec un crédit supérieur ou égal à celui du client recherché
    higher_credit_clients_count = len(df[df['Somme des crédits'] >= client_credit])
    # Calculer le nombre de clients avec une annuité inférieure à celle du client recherché
    lower_annuity_clients_count = len(df[df["Taux d’endettement"] < client_annuity])
    # Calculer le nombre de clients avec une annuité supérieure ou égale à celle du client recherché
    higher_annuity_clients_count = len(df[df["Taux d’endettement"] >= client_annuity])

    # Graphique 1: les crédits

    # Création des données pour le diagramme circulaire (AMT_CREDIT)
    credit_sizes = [lower_credit_clients_count, higher_credit_clients_count]
    credit_labels = ['Crédit inférieur', 'Crédit supérieur ou égal']
    credit_colors = ['Gold', 'Silver']
    # Création de la figure et de l'axe
    fig_credit, ax_credit = plt.subplots(figsize=(6, 6))

    # Tracer le diagramme circulaire pour la variable "AMT_CREDIT"
    ax_credit.pie(credit_sizes, labels=credit_labels, colors=credit_colors, autopct='%1.1f%%', startangle=90)
    ax_credit.axis('equal')
    ax_credit.set_title('Répartition des clients par rapport au crédit')

    # Affichage du graphique à l'aide de Streamlit
    st.pyplot(fig_credit)
    # Graphique 2: endettement

    # Création des données pour le diagramme circulaire (INCOME_CREDIT_PERC)
    annuity_sizes = [lower_annuity_clients_count, higher_annuity_clients_count]
    annuity_labels = ['Endettement inférieur', 'Endettement supérieur ou égal']
    annuity_colors = ['DodgerBlue', 'DarkOrange']
    # Création de la figure et de l'axe pour le diagramme circulaire (INCOME_CREDIT_PERC)
    fig_annuity, ax_annuity = plt.subplots(figsize=(6, 6))
    ax_annuity.pie(annuity_sizes, labels=annuity_labels, colors=annuity_colors, autopct='%1.1f%%', startangle=90)
    ax_annuity.axis('equal')
    ax_annuity.set_title("Répartition des clients par rapport à l'endettement")

    # Affichage du graphique du diagramme circulaire (INCOME_CREDIT_PERC) à l'aide de Streamlit
    st.pyplot(fig_annuity)

# Prédiction avec listing clients

# Listing KNN
# Récupérer les caractéristiques du client à prédire

if search_input != "" and search_input.isdigit():
    client_features = df[df['SK_ID_CURR'] == int(search_input)]

    if client_features.empty:
        st.write("Identifiant du client introuvable.")
    else:
        # Prédiction des voisins les plus proches
        prediction = model2.predict(client_features)

        # Liste des identifiants des voisins les plus proches
        nearest_neighbors_ids = [liste_id[i] for i in prediction.flatten()]

        # Reste du code pour afficher les informations des voisins les plus proches...

        # Affichage des informations des voisins les plus proches
        # Sélection aléatoire de 2 clients parmi les voisins les plus proches
        random_clients = random.sample(nearest_neighbors_ids, k=min(2, len(nearest_neighbors_ids)))
        random_clients_info = df[df['SK_ID_CURR'].isin(random_clients)][
            ['SK_ID_CURR', 'Sexe', 'Revenus annuels', 'Revenus totaux', 'Somme des crédits', "Taux d’endettement",
             'Propriétaire']]

        # Affichage des informations des clients sélectionnés de manière aléatoire
        st.write("Clients les plus proches :")
        st.table(random_clients_info)
