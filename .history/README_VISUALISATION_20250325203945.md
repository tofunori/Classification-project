# Visualisation de Classification Non Supervisée

Ce projet contient des scripts pour visualiser les résultats d'une classification non supervisée stockée dans un fichier TIFF. Les scripts permettent de générer:

1. Une carte de classification
2. Une analyse en composantes principales (PCA) avec visualisations 2D et 3D
3. Une matrice de similarité/confusion entre les classes

## Prérequis

Les dépendances suivantes sont nécessaires:

```
numpy>=1.20.0
matplotlib>=3.4.0
seaborn>=0.11.0
rasterio>=1.2.0
scikit-learn>=0.24.0
scipy>=1.6.0
pandas>=1.2.0
```

Installez-les avec:

```bash
pip install -r requirements.txt
```

## Options d'utilisation

Plusieurs options sont disponibles selon votre préférence:

### Option 1: Script en ligne de commande

Le script `visualiser_classification.py` offre une interface en ligne de commande complète:

```bash
python visualiser_classification.py --classification chemin/vers/classification.tif --donnees_originales chemin/vers/donnees_originales.tif --output_dir ./resultats
```

Arguments:
- `--classification`: Chemin vers le fichier TIFF de classification (obligatoire)
- `--donnees_originales`: Chemin vers le fichier TIFF des données originales (facultatif)
- `--output_dir`: Répertoire de sortie (par défaut: ./resultats_visualisation)
- `--bands`: Indices des bandes à utiliser, base 1 (facultatif)

### Option 2: Script d'exemple

Le script `exemple_visualisation.py` est un wrapper pour l'option 1:

```bash
python exemple_visualisation.py --classification chemin/vers/classification.tif --donnees_originales chemin/vers/donnees_originales.tif
```

### Option 3: Script simplifié avec paramètres intégrés

Le script `visualiser_classification_simple.py` est le plus simple à utiliser:

1. Éditez le fichier pour modifier les variables en haut du script:
   ```python
   FICHIER_CLASSIFICATION = "chemin/vers/classification.tif"
   FICHIER_DONNEES_ORIGINALES = "chemin/vers/donnees.tif"
   REPERTOIRE_SORTIE = "resultats_visualisation"
   ```

2. Exécutez simplement le script:
   ```bash
   python visualiser_classification_simple.py
   ```

## Fichiers de sortie

Les scripts génèrent les fichiers suivants dans le répertoire de sortie:

1. `carte_classification.png`: Carte colorée de la classification
2. `analyse_pca.png`: Visualisations PCA (si des données originales sont fournies)
3. `matrice_similarite.png`: Matrice de similarité entre classes (si des données originales sont fournies)

## Exemples de résultats

### Carte de classification
La carte montre la répartition spatiale des classes identifiées par la classification non supervisée.

### Analyse PCA
Le graphique PCA montre comment les classes se distinguent dans l'espace des composantes principales, révélant les similarités et différences entre classes.

### Matrice de similarité
Cette matrice montre la proximité relative des centres des clusters dans l'espace spectral ou des composantes principales.

## Remarques importantes

1. Si vous fournissez uniquement le fichier de classification, seule la carte sera générée.
2. Pour l'analyse PCA et la matrice de similarité, vous devez fournir les données originales utilisées pour la classification.
3. Pour de gros fichiers, le script peut échantillonner les données pour éviter les problèmes de mémoire. 