# Projet de Classification d'Images Satellitaires

Ce projet implémente différentes méthodes de classification d'images satellitaires Sentinel-2 pour l'analyse d'occupation des sols à Trois-Rivières, Québec.

## Structure du Projet

```
classification_project/
├── modules/
│   ├── config.py         # Configuration du projet
│   ├── data_loader.py    # Chargement et préparation des données
│   ├── model.py          # Implémentation des modèles de classification
│   ├── train.py          # Extraction des échantillons d'entraînement
│   ├── evaluate.py       # Métriques d'évaluation
│   └── visualize.py      # Outils de visualisation
├── requirements.txt      # Dépendances Python
├── main.py               # Point d'entrée principal
└── README.md             # Documentation
```

## Fonctionnalités

- Classification par Maximum de Vraisemblance (MLC)
- Classification avec pondération des bandes
- Visualisation des signatures spectrales et scatterplots
- Analyse en Composantes Principales (PCA)
- Calcul des matrices de confusion et métriques de précision
- Cartes d'incertitude (entropie)

## Installation

1. Cloner ce dépôt:
   ```
   git clone https://github.com/votre-username/classification-sentinel2.git
   cd classification-sentinel2
   ```

2. Installer les dépendances:
   ```
   pip install -r requirements.txt
   ```

## Utilisation

### Exécution avec les paramètres par défaut:

```
python main.py
```

### Exécution avec un chemin d'entrée personnalisé:

```
python main.py "D:/chemin/vers/image.tif"
```

### Exécution avec un chemin d'entrée et un dossier de sortie personnalisés:

```
python main.py "D:/chemin/vers/image.tif" "D:/chemin/vers/dossier/sortie"
```

## Configuration

Le fichier `modules/config.py` contient tous les paramètres configurables du projet. Vous pouvez modifier directement ce fichier ou passer une configuration personnalisée via la fonction `run_classification()`.

## Résultats

Les résultats sont enregistrés dans le dossier spécifié et comprennent:

- Cartes de classification raster (.tif)
- Visualisations des classifications (.png)
- Signatures spectrales (.png)
- Scatterplots et visualisations PCA (.png)
- Matrices de confusion et rapports de précision (.png)
- Cartes d'incertitude (.tif et .png)
- Statistiques sur les classes (.png)

## Auteur

[Votre Nom] - pour TP3 GEO, UQTR, Hiver 2025
