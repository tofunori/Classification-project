"""
CLASSIFICATION SANS PONDÉRATION
===============================
Ce script effectue une classification par maximum de vraisemblance sans pondération des bandes,
avec génération d'un fichier TIF, et affichage des signatures spectrales sans graphique de pondération.
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
from modules.data_loader import load_and_check_data, create_output_directory, save_raster
from modules.model import perform_classification
from modules.evaluate import validate_classification
from modules.visualize import visualize_spectral_signatures, generate_classification_map

def main():
    """Fonction principale pour exécuter la classification sans pondération."""
    start_time = time.time()
    
    print("======================================================================")
    print(" CLASSIFICATION SANS PONDÉRATION AVEC GÉNÉRATION DE TIF ")
    print("======================================================================")
    
    # Configuration de base
    config = {
        "raster_path": r"D:\UQTR\Hiver 2025\Télédétection\TP3\tr_clip.tif",
        "shapefile_path": r"D:\UQTR\Hiver 2025\Télédétection\TP3\classes.shp",
        "class_column": "Classe",
        "selected_bands": [2, 3, 4, 5, 6, 7, 8],
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
            "enabled": True,
            "shapefile_path": r"D:\UQTR\Hiver 2025\Télédétection\TP3\points_validation.shp"
        },
        "comparison": {
            "enabled": False
        },
        "show_weight_plots": False,  # Désactiver les graphiques de pondération
        "skip_visualizations": False  # Ne pas sauter les visualisations
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
    output_dir = os.path.join(r"D:\UQTR\Hiver 2025\Télédétection\TP3\resultats", f"classification_unweighted_tif_{timestamp}")
    create_output_directory(output_dir)
    config["output_dir"] = output_dir
    
    print(f"\nRépertoire de sortie: {output_dir}")
    print("\nLancement de la classification...")
    
    try:
        # Chargement et vérification des données
        raster_data, meta, shapefile = load_and_check_data(config)
        
        # Extraction des échantillons d'entraînement
        print("\nExtraction des échantillons d'entraînement...")
        from main import extract_training_samples
        classes_info = extract_training_samples(raster_data, shapefile, config)
        
        # Classification sans pondération
        print("\nClassification standard sans pondération...")
        classification, _ = perform_classification(raster_data, classes_info, config)
        
        # Validation de la classification
        print("\nValidation de la classification...")
        validation_results = validate_classification(classification, config)
        
        # Génération des signatures spectrales (sans poids)
        print("\nGénération des signatures spectrales...")
        visualize_spectral_signatures(classes_info, config)
        
        # Enregistrement du fichier TIF
        tif_path = os.path.join(output_dir, "classification_unweighted.tif")
        save_result = save_raster(classification, meta, tif_path)
        
        if save_result:
            print(f"\nFichier TIF généré avec succès: {tif_path}")
        else:
            print("\nAttention: Le fichier TIF n'a pas été généré correctement.")
        
        # Génération de la carte de classification
        print("\nGénération de la carte de classification...")
        generate_classification_map(classification, config)
        
        # Afficher les résultats de la validation
        if validation_results:
            print("\nRésultats de la classification:")
            print(f"  Précision validation: {validation_results['accuracy']:.4f}")
            print(f"  Kappa validation: {validation_results['kappa']:.4f}")
            
            # Enregistrer les résultats dans un fichier JSON
            results_dict = {
                "timestamp": timestamp,
                "precision": validation_results['accuracy'],
                "kappa": validation_results['kappa'],
                "weights": [float(w) for w in uniform_weights]
            }
            
            results_path = os.path.join(output_dir, "resultats_classification.json")
            with open(results_path, 'w') as f:
                json.dump(results_dict, f, indent=2)
            
            print(f"\nRésultats enregistrés dans: {results_path}")
        
        # Afficher le temps d'exécution
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"\nTemps d'exécution total: {execution_time:.2f} secondes")
        
        print("\nClassification terminée avec succès!")
        print(f"Les résultats sont disponibles dans: {output_dir}")
        print("\nLes signatures spectrales ont été générées et sauvegardées.")
        
        # Afficher le chemin vers la matrice de confusion
        confusion_matrix_path = os.path.join(output_dir, "matrice_confusion.png")
        confusion_matrix_percent_path = os.path.join(output_dir, "matrice_confusion_pourcent.png")
        
        if os.path.exists(confusion_matrix_path):
            print(f"\nVoir la matrice de confusion dans:\n{confusion_matrix_path}")
        
        if os.path.exists(confusion_matrix_percent_path):
            print(f"\nVoir la matrice de confusion en pourcentage dans:\n{confusion_matrix_percent_path}")
            
    except Exception as e:
        print(f"\nERREUR lors de la classification: {str(e)}")
        print(traceback.format_exc())

if __name__ == "__main__":
    main() 