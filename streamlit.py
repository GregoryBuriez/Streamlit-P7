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
import warnings
import shap

# Titre de l'application
st.title('Implémentez un modèle de scoring')

# 1. Importation du dataframe et du modèle

# Lecture du fichier CSV

#df = pd.read_csv("df_tabdashboard.csv", usecols=lambda col: col != 'TARGET', nrows=int(0.1 * pd.read_csv("df_tabdashboard.csv").shape[0]))  # Sélection de 10% des clients
df = pd.read_csv("df_tabdashboard.csv")
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

# Interprétabilité locale

# Créer un explainer SHAP avec le modèle
explainer = shap.Explainer(model1, df)

# Calculer les valeurs SHAP pour toutes les observations
shap_values = explainer(df)

# Fonction pour générer le graphique d'interprétation locale en fonction de l'identifiant
def generate_local_interpretation_graph(identifier):
    observation_index = df[df['SK_ID_CURR'] == identifier].index[0]
    shap.initjs()  # Initialiser le support JavaScript
    
    # Créer une figure et un axe
    fig, ax = plt.subplots()
    shap.plots.waterfall(shap_values[observation_index, :], max_display=10, show=False)
    
    # Ajouter un ajustement de la mise en page
    plt.tight_layout()
    
    # Afficher le graphique via st.pyplot()
    st.pyplot(fig)

# Vérifier si un identifiant valide a été spécifié par l'utilisateur
if search_input != "" and search_input.isdigit():
    # Convertir la valeur en entier
    identifier = int(search_input)

    # Générer le graphique d'interprétation locale
    generate_local_interpretation_graph(identifier)
else:
    # Afficher un message d'erreur
    st.write("Erreur : veuillez entrer un identifiant de client valide.")

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



# Prédiction avec le modèle KNN
    if not filtered_df.empty:
        
        # Prédiction des voisins les plus proches
        neighbors_indices = model2.kneighbors(df)[1][0]

        # Liste des identifiants des voisins les plus proches
        nearest_neighbors_ids = [liste_id[i] for i in neighbors_indices]

        # Affichage des informations des voisins les plus proches
        # Sélection aléatoire de 2 clients parmi les voisins les plus proches
        random_clients = random.sample(nearest_neighbors_ids, k=min(5, len(nearest_neighbors_ids)))
        random_clients_info = df[df['SK_ID_CURR'].isin(random_clients)][
            ['SK_ID_CURR', 'Sexe', 'Revenus annuels', 'Revenus totaux', 'Somme des crédits', "Taux d’endettement",
             'Propriétaire']]

        # Affichage des informations des clients sélectionnés de manière aléatoire
        st.write("Clients les plus proches :")
        st.table(random_clients_info)
else:
    st.write("Veuillez spécifier un identifiant de client valide dans la barre de recherche.")
