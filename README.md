# Scoring marketing — Prédiction de la réponse à une campagne de mailing

Modèle de scoring identifiant les clients les plus susceptibles de répondre positivement à une offre commerciale, à partir de leurs caractéristiques socio-démographiques et de leur historique d'achat.

## Contexte métier

Dans une campagne de mailing marketing, cibler l'ensemble de la base client est coûteux et peu efficace, la majorité des clients ne répondant pas à l'offre. L'enjeu est d'estimer, pour chaque client, sa probabilité de réponse positive, afin de prioriser les actions commerciales vers les profils les plus réceptifs.

## Dataset

- **Source** : base de données fournie dans un cadre académique (Université de Lille)
- **Volume** : plusieurs dizaines de milliers de clients, variables socio-démographiques (âge, sexe, statut marital, propriété, localité) et comportementales (historique d'achat, montant des commandes, score FICO)
- **Cible** : `RETOUR` (`OUI` = réponse positive au mailing, `NON` = pas de réponse)
- **Déséquilibre des classes** : 13,6 % de retours positifs contre 86,4 % de non-réponses

## Méthodologie

### 1. Exploration des données et tests statistiques
Analyse descriptive, discrétisation de l'âge, et tests statistiques pour identifier les variables discriminantes :
- **Test du Khi-deux** sur les variables qualitatives (âge, sexe, statut marital, propriété, localité)
- **Test de Student** sur les variables quantitatives (score FICO, historique d'achat)

### 2. Préparation des données
- Suppression des valeurs manquantes marginales (variable `SEX`)
- Encodage One-Hot des variables qualitatives retenues
- Standardisation des variables quantitatives
- Séparation train/test stratifiée (70/30)

### 3. Modélisation
Deux modèles comparés sur l'AUC-ROC :
- Régression logistique (avec équilibrage des classes)
- Random Forest (avec équilibrage des classes)

### 4. Scoring opérationnel
Transformation des probabilités prédites en 5 classes de score (`Très faible` à `Très élevé`), pour un usage direct en priorisation commerciale.

## Résultats

| Modèle | AUC-ROC |
|---|---|
| Régression logistique | 0.655 |
| Random Forest | 0.656 |

Les deux modèles obtiennent une performance quasi identique. Le Random Forest, capable de détecter des relations non-linéaires, n'apporte ici aucun gain significatif — ce qui suggère que la relation entre les variables explicatives et la probabilité de réponse est largement linéaire. Ce résultat plaide pour la régression logistique en production : à performance égale, elle est plus simple, plus stable et plus interprétable.

### Variables les plus discriminantes

Les tests statistiques identifient la classe d'âge, la localité, le statut de propriété, le nombre de commandes passées et le montant des commandes sur 24 mois comme les variables les plus associées à la réponse positive. La variable `INCOME` n'est en revanche pas significative et a été exclue de la modélisation.

### Classes de score

Les probabilités prédites sont réparties en 5 classes de score. Le taux de retour positif progresse nettement selon le niveau de score, de ~4.5 % pour la classe `Très faible` à ~23.2 % pour la classe `Très élevé` — une progression d'un facteur 5, qui confirme l'utilité opérationnelle du modèle pour prioriser les actions marketing.

## Stack technique

- Python, pandas, numpy
- scipy (tests statistiques : Khi-deux, Student)
- scikit-learn (préparation des données, modélisation)
- matplotlib, seaborn (visualisation)

## Structure du repository

```
scoring_marketing/
│
├── data/
│   └── mailing_fin.txt        (non inclus, données fournies dans un cadre académique)
│
├── notebooks/
│   ├── 01_EDA_et_tests_statistiques.ipynb
│   └── 02_modelisation.ipynb
│
├── README.md
└── requirements.txt
```

## Reproduire le projet

```bash
git clone https://github.com/<ton-pseudo>/scoring-marketing.git
cd scoring-marketing
pip install -r requirements.txt
```

Le fichier de données `mailing_fin.txt` n'étant pas public (source académique), il n'est pas inclus dans ce repository.
