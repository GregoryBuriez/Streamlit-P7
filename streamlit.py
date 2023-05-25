# Importation des bibliothèques
import streamlit as st
import pandas as pd
import altair as alt
import matplotlib.pyplot as plt
from urllib.request import urlopen
import json
import pickle
from sklearn.neighbors import NearestNeighbors
import random

# Titre de l'application
st.title('Implémentez un modèle de scoring')

# 1. Importation du dataframe et du modèle

# Lecture du fichier CSV
df = pd.read_csv("df_tabdashboard.csv", usecols=lambda col: col != 'TARGET', nrows=int(0.1 * pd.read_csv("df_tabdashboard.csv").shape[0]))  # Sélection de 10% des clients
liste_id = df['SK_ID_CURR'].tolist()



# Chargement du modèle
with open("model_streamlit.pkl", "rb") as file:
    model = pickle.load(file)
    
with open('model_KNN_streamlit.pkl', 'rb') as file2:
    model2 = pickle.load(file2)

       
    
st.subheader('Prédiction de notre modèle')

# Barre de recherche
search_input = st.text_input("Rechercher par identifiant SK_ID_CURR")

# Vérifier si la valeur de search_input est vide ou non numérique
if search_input != "" and search_input.isdigit():
    # Recherche du client dans le dataframe
    filtered_df = df[df['SK_ID_CURR'] == int(search_input)]
    
    # Sélection des colonnes souhaitées
    selected_columns = ['SK_ID_CURR', 'Sexe', 'Revenus annuels', 'Revenus totaux', 'Somme des crédits',"Taux d’endettement", 'Propriétaire']
    filtered_df = filtered_df[selected_columns]

    # Affichage du résultat
    if len(filtered_df) == 0:
        st.write("Erreur identifiant")
    else:
        st.write("Résultat de la recherche :")
        st.write(filtered_df)
        
        # Affichage du ratio du client en pourcentage avec une mise en forme personnalisée
        payment_rate = filtered_df["Taux d’endettement"].values[0] * 100
        st.write(f"<div style='display: flex; align-items: center; font-size: 25px;'>Endettement du client : <span style='font-size: 30px; font-weight: bold;'>{payment_rate}%</span></div>", unsafe_allow_html=True)
        
        
        
        ### Modélisation    
    
# Prédiction de la classe et probabilité
search_id = int(search_input)
if search_id in liste_id:
    filtered_df = df[df['SK_ID_CURR'] == int(search_input)]

    if not filtered_df.empty:
        prediction = model.predict(filtered_df)

        if prediction[0] == 0:
            st.write("<div style='font-size: 25px; font-weight: bold; text-align: center;'>Prédiction : Sans risque (0)</div>", unsafe_allow_html=True)
        else:
            st.write("<div style='font-size: 25px; font-weight: bold; text-align: center;'>Prédiction : En défaut (1)</div>", unsafe_allow_html=True)


##### Préparation données pour graphique####

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


#### Graphique 1 : les crédits ####

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
    
    
    #### Graphique 2 : endettement ####
    
    # Création des données pour le diagramme circulaire (INCOME_CREDIT_PERC)
    annuity_sizes = [lower_annuity_clients_count, higher_annuity_clients_count]
    annuity_labels = ['Endettement inférieur', 'Endettement supérieur ou égal']
    annuity_colors = ['DodgerBlue', 'DarkOrange']
    # Création de la figure et de l'axe pour le diagramme circulaire (INCOME_CREDIT_PERC)
    fig_annuity, ax_annuity = plt.subplots(figsize=(6, 6))
    ax_annuity.pie(annuity_sizes, labels=annuity_labels, colors=annuity_colors, autopct='%1.1f%%', startangle=90)
    ax_annuity.axis('equal')
    ax_annuity.set_title('Répartition des clients par rapport à l\'annuité')

# Affichage du graphique du diagramme circulaire (INCOME_CREDIT_PERC) à l'aide de Streamlit
    st.pyplot(fig_annuity)



###### Prédiction avec listing clients #####


    # Listing KNN
# Récupérer les caractéristiques du client à prédire
    client_features = df[df['SK_ID_CURR'] == int(search_input)].drop('SK_ID_CURR', axis=1)

# Prédiction des voisins les plus proches
    prediction = model2.predict(df)

# Liste des identifiants des voisins les plus proches
    nearest_neighbors_ids = [liste_id[i] for i in prediction.flatten()]

# Affichage des informations des voisins les plus proches
# Sélection aléatoire de 2 clients parmi les voisins les plus proches
    random_clients = random.sample(nearest_neighbors_ids, k=2)
    random_clients_info = df[df['SK_ID_CURR'].isin(random_clients)][['SK_ID_CURR', 'Sexe', 'Revenus annuels', 'Revenus totaux', 'Somme des crédits',"Taux d’endettement", 'Propriétaire']]

# Affichage des informations des clients sélectionnés de manière aléatoire
    st.write("Clients les plus proches :")
    st.table(random_clients_info)
            
else:
    st.write("Aucune donnée disponible pour l'ID recherché.")

    
