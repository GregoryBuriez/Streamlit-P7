Vous êtes Data Scientist au sein d'une société financière, nommée "Prêt à dépenser", qui propose des crédits à la consommation pour des personnes ayant peu ou pas du tout d'historique de prêt.

L’entreprise souhaite mettre en œuvre un outil de “scoring crédit” pour calculer la probabilité qu’un client rembourse son crédit, puis classifie la demande en crédit accordé ou refusé. Elle souhaite donc développer un algorithme de classification en s’appuyant sur des sources de données variées (données comportementales, données provenant d'autres institutions financières, etc.).

De plus, les chargés de relation client ont fait remonter le fait que les clients sont de plus en plus demandeurs de transparence vis-à-vis des décisions d’octroi de crédit. Cette demande de transparence des clients va tout à fait dans le sens des valeurs que l’entreprise veut incarner. Prêt à dépenser décide donc de développer un dashboard interactif pour que les chargés de relation client puissent à la fois expliquer de façon la plus transparente possible les décisions d’octroi de crédit, mais également permettre à leurs clients de disposer de leurs informations personnelles et de les explorer facilement.

I. Votre mission

- Construire un modèle de scoring qui donnera une prédiction sur la probabilité de faillite d'un client de façon automatique.
- Construire un dashboard interactif à destination des gestionnaires de la relation client permettant d'interpréter les prédictions faites par le modèle, et d’améliorer la connaissance client des chargés de relation client.
- Mettre en production le modèle de scoring de prédiction à l’aide d’une API, ainsi que le dashboard interactif qui appelle l’API pour les prédictions.
- Michaël, votre manager, vous incite à sélectionner un ou des kernels Kaggle pour vous faciliter l’analyse exploratoire, la préparation des données et le feature engineering nécessaires à l’élaboration du modèle de scoring. Si vous le faites, vous devez analyser ce ou ces kernels et le ou les adapterpour vous assurer qu’ils répond(ent) aux besoins de votre mission.


II. Compétences évaluées

- Définir et mettre en œuvre une stratégie de suivi de la performance d’un modèle
- Évaluer les performances des modèles d’apprentissage supervisé
- Utiliser un logiciel de version de code pour assurer l’intégration du modèle
- Définir la stratégie d’élaboration d’un modèle d’apprentissage supervisé
- Réaliser un dashboard pour présenter son travail de modélisation
- Rédiger une note méthodologique afin de communiquer sa démarche de modélisation
- Présenter son travail de modélisation à l'oral
- Déployer un modèle via une API dans le Web
- Définir et mettre en œuvre un pipeline d’entraînement des modèles

III. Présentation du tableau de bord

Notre dashboard interactif est développé en utilisant Streamlit, un framework convivial et puissant pour la création d'applications web de data science en Python. 
Streamlit simplifie le processus de construction de l'interface utilisateur et permet de visualiser facilement les informations.

Une fois que nous avons développé le dashboard avec Streamlit, nous prévoyons de le déployer sur le Cloud de Streamlit. 
Le Cloud de Streamlit offre une plateforme de déploiement simple et efficace pour les applications Streamlit
Ceci va permettrea ux utilisateurs d'accéder au dashboard en ligne sans avoir à se soucier de la configuration et de la gestion des serveurs.

L'objectif principal de notre dashboard est de permettre aux utilisateurs de visualiser rapidement les informations et d'interpréter les résultats de la prédiction et du client. 
Nous voulons fournir une interface intuitive où les utilisateurs pourront facilement comprendre les prédictions de remboursement et obtenir des informations détaillées sur chaque client.

Pour obtenir les prédictions, notre dashboard utilisera une collecte via notre API. 
Grâce à cette intégration, le dashboard pourra envoyer des requêtes à notre API pour obtenir les prédictions de remboursement en temps réel. 
Cela permettra aux utilisateurs d'obtenir des informations actualisées et fiables lorsqu'ils consultent le dashboard.

En résumé, notre dashboard interactif utilise Streamlit comme framework pour faciliter l'expérience utilisateur. 
Il est déployé sur le Cloud de Streamlit pour une accessibilité en ligne optimale. 
L'objectif est de permettre aux utilisateurs de visualiser rapidement les informations et d'interpréter les résultats de la prédiction et du client. 
Le dashboard utilise une collecte via notre API pour obtenir les prédictions de remboursement en temps réel et fournir ainsi des informations actualisées.
