"""
VISUALISATION PCA AVANCÉE
=======================================
Ce script génère une visualisation avancée de l'analyse en composantes principales (PCA)
pour les données de classification, incluant les projections 2D et 3D ainsi que
la matrice des coefficients des composantes principales.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import time
import json
import traceback

# Importer les modules du projet
from modules.data_loader import load_and_check_data, create_output_directory
from modules.visualize import create_pca_scatterplots
from modules.model import perform_classification

def main():
    """Fonction principale pour générer la visualisation PCA avancée."""
    start_time = time.time()
    
    print("======================================================================")
    print(" VISUALISATION PCA AVANCÉE ")
    print("======================================================================")
    
    # Configuration de base
    config = {
        "raster_path": r"D:\UQTR\Hiver 2025\Télédétection\TP3\tr_clip.tif",
        "shapefile_path": r"D:\UQTR\Hiver 2025\Télédétection\TP3\classes.shp",
        "class_column": "Classe",
        "selected_bands": [2, 3, 4, 5, 6, 7, 8],
        "class_params": {
            1: [1e-5, 0],   # Eau
            2: [1e-4, 0],   # Forêt
            3: [3e-4, 0],   # Tourbière
            4: [5e-4, 0],   # Herbes
            5: [1e-3, 0],   # Champs
            6: [1e-2, 0]    # Urbain
        },
        "class_names": {
            1: "Eau", 
            2: "Forêt", 
            3: "Tourbière",
            4: "Herbes", 
            5: "Champs", 
            6: "Urbain"
        },
        "class_colors": {
            1: "#3288bd",  # Eau - bleu
            2: "#66c164",  # Forêt - vert
            3: "#87CEFA",  # Tourbière - bleu clair
            4: "#ffff00",  # Herbes - vert clair
            5: "#f39c12",  # Champs - orange
            6: "#7f8c8d"   # Urbain - gris
        },
        "validation": {
            "enabled": False  # Désactiver la validation pour ce script
        },
        "comparison": {
            "enabled": False
        },
        "show_weight_plots": False,
        "skip_visualizations": False
    }
    
    # Utiliser des poids uniformes (1.0) pour chaque bande
    num_bands = len(config["selected_bands"])
    uniform_weights = np.ones(num_bands)
    
    print("Poids par bande (uniformes):")
    band_names = ["B2 - Bleu", "B3 - Vert", "B4 - Rouge", "B5 - RedEdge05", 
                  "B6 - RedEdge06", "B7 - RedEdge07", "B8 - PIR"]
    
    for i, band in enumerate(config["selected_bands"]):
        print(f"  {band_names[band-2]}: {uniform_weights[i]:.2f}")
    
    # Créer un répertoire pour les résultats avec un timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(r"D:\UQTR\Hiver 2025\Télédétection\TP3\resultats", f"pca_avance_{timestamp}")
    create_output_directory(output_dir)
    config["output_dir"] = output_dir
    
    print(f"\nRépertoire de sortie: {output_dir}")
    print("\nChargement des données pour l'analyse PCA...")
    
    try:
        # Chargement et vérification des données
        raster_data, meta, shapefile = load_and_check_data(config)
        
        # Extraction des échantillons d'entraînement
        print("\nExtraction des échantillons d'entraînement...")
        from main import extract_training_samples
        classes_info = extract_training_samples(raster_data, shapefile, config)
        
        # Génération des scatterplots PCA avancés
        print("\nGénération des visualisations PCA avancées...")
        
        # Utiliser la fonction complète du module visualize.py
        success = create_pca_scatterplots(classes_info, config)
        
        if success:
            print("\nAnalyse PCA avancée terminée avec succès!")
            print(f"Les visualisations sont disponibles dans: {output_dir}/scatterplots/")
        else:
            print("\nErreur lors de la génération des visualisations PCA.")
        
        # Afficher le temps d'exécution
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"\nTemps d'exécution total: {execution_time:.2f} secondes")
        
    except Exception as e:
        print(f"\nErreur lors de l'exécution: {str(e)}")
        print(traceback.format_exc())

if __name__ == "__main__":
    main() 