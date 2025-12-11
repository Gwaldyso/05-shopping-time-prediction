# 🛒 Prédicteur de Temps de Shopping

Un projet complet de Machine Learning pour prédire le temps qu'une personne passera dans un magasin en fonction de sa liste de courses et de ses habitudes.

## 🎯 Objectif du Projet

Ce projet utilise des techniques de Machine Learning pour prédire le temps de shopping en analysant :
- Le profil du client (âge, habitudes)
- Le type de magasin
- La liste de courses (nombre d'articles par catégorie)
- Les facteurs temporels (jour, heure, weekend)
- Les conditions du magasin (affluence, heure de pointe)

## 🏗️ Architecture du Projet

\`\`\`
shopping-time-predictor/
├── scripts/
│   ├── 01_generate_dataset.py      # Génération du dataset synthétique
│   ├── 02_data_preprocessing.py    # Préparation des données
│   ├── 03_train_model.py          # Entraînement des modèles
│   ├── 04_prediction_interface.py  # Interface de prédiction
│   └── 05_model_evaluation.py     # Évaluation du modèle
├── data/                          # Données générées et modèles
├── models/                        # Modèles sauvegardés
└── README.md
\`\`\`

## 🚀 Installation et Utilisation

### Prérequis

\`\`\`bash
pip install pandas numpy scikit-learn matplotlib seaborn joblib scipy
\`\`\`

### Étapes d'exécution

1. **Génération du dataset** :
\`\`\`bash
python scripts/01_generate_dataset.py
\`\`\`

2. **Préparation des données** :
\`\`\`bash
python scripts/02_data_preprocessing.py
\`\`\`

3. **Entraînement du modèle** :
\`\`\`bash
python scripts/03_train_model.py
\`\`\`

4. **Interface de prédiction** :
\`\`\`bash
python scripts/04_prediction_interface.py
\`\`\`

5. **Évaluation du modèle** :
\`\`\`bash
python scripts/05_model_evaluation.py
\`\`\`

## 📊 Features du Modèle

### Variables d'entrée :
- **Profil client** : âge, type de comportement (rapide, normal, flâneur, méthodique)
- **Magasin** : type (hypermarché, supermarché, centre commercial, magasin spécialisé)
- **Temporel** : jour de la semaine, heure, weekend, heure de pointe
- **Shopping** : nombre total d'articles, nombre de catégories, présence d'une liste
- **Articles par catégorie** : alimentaire, vêtements, électronique, maison & jardin, beauté & santé, sport & loisirs, librairie
- **Autres** : gamme de budget, méthode de paiement

### Variables dérivées :
- Articles par catégorie moyenne
- Score de complexité du shopping
- Interaction weekend × heure de pointe

## 🤖 Modèles Utilisés

Le projet compare trois algorithmes :
1. **Random Forest** - Ensemble de arbres de décision
2. **Gradient Boosting** - Boosting séquentiel
3. **Régression Linéaire** - Modèle de base

Le meilleur modèle est automatiquement sélectionné basé sur le RMSE.

## 📈 Métriques d'Évaluation

- **RMSE** (Root Mean Square Error) - Erreur quadratique moyenne
- **MAE** (Mean Absolute Error) - Erreur absolue moyenne  
- **R²** (Coefficient de détermination) - Variance expliquée
- **MAPE** (Mean Absolute Percentage Error) - Erreur en pourcentage

## 🎮 Interface Interactive

L'interface permet de :
- Créer un profil de shopping personnalisé
- Obtenir une prédiction de temps
- Recevoir des conseils d'optimisation
- Analyser les facteurs influençant le temps

## 📊 Exemple de Résultats

\`\`\`
🎯 Métriques du modèle :
   RMSE: 8.5 minutes
   MAE: 6.2 minutes
   R²: 0.847

🎯 Précision des prédictions :
   ±5 minutes: 45.2%
   ±10 minutes: 72.8%
   ±15 minutes: 89.1%
\`\`\`

## 🔍 Features les Plus Importantes

1. Nombre total d'articles
2. Score de complexité du shopping
3. Type de magasin
4. Profil client
5. Nombre de catégories

## 🛠️ Personnalisation

Le modèle peut être facilement adapté pour :
- Ajouter de nouvelles catégories de produits
- Modifier les facteurs de complexité
- Intégrer des données réelles de magasins
- Ajouter des variables météorologiques ou saisonnières

## 📝 Cas d'Usage

- **Planification personnelle** : Estimer le temps nécessaire pour ses courses
- **Gestion de magasins** : Prévoir l'affluence et optimiser les ressources
- **Applications mobiles** : Intégrer dans des apps de shopping
- **Recherche** : Analyser les comportements de consommation

## 🔮 Améliorations Futures

- Intégration de données réelles de magasins
- Modèles de deep learning (réseaux de neurones)
- Prise en compte de la géolocalisation
- Analyse des trajets dans le magasin
- Prédiction en temps réel avec IoT

## 📄 Licence

Ce projet est à des fins éducatives et de démonstration.
