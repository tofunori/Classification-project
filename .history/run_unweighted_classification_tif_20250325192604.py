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
import rasterio
import json

# Importer les modules du projet
from modules.data_loader import load_training_data, load_reference_data
from modules.model import perform_classification, apply_weights
from modules.evaluate import validate_classification
from modules.visualize import plot_spectral_signatures, plot_confusion_matrix
from modules.config import load_config, save_config

def main():
    """Fonction principale pour exécuter la classification sans pondération."""
    print("======================================================================")
    print(" CLASSIFICATION SANS PONDÉRATION AVEC GÉNÉRATION DE TIF ")
    print("======================================================================")
    
    # Charger la configuration
    config = load_config()
    
    # Utiliser des poids uniformes (1.0) pour chaque bande
    num_bands = len(config.selected_bands)
    uniform_weights = np.ones(num_bands)
    
    print("Poids par bande (uniformes):")
    band_names = ["B2 - Bleu", "B3 - Vert", "B4 - Rouge", "B5 - RedEdge05", 
                  "B6 - RedEdge06", "B7 - RedEdge07", "B8 - PIR"]
    
    for i, band in enumerate(config.selected_bands):
        print(f"  {band_names[band-2]}: {uniform_weights[i]:.2f}")
    
    # Créer un répertoire pour les résultats avec un timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(config.output_dir, f"classification_unweighted_tif_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    
    # Configurer pour la validation
    custom_config = config.copy()
    custom_config.output_dir = output_dir
    custom_config.model_weights = uniform_weights
    
    # Générer un fichier TIF
    custom_config.save_raster = True
    custom_config.save_raster_path = os.path.join(output_dir, "classification_unweighted.tif")
    
    # Supprimer l'affichage des graphiques de pondération
    custom_config.show_weight_plots = False
    
    # Exécuter la classification
    print(f"\nRépertoire de sortie: {output_dir}")
    print("\nLancement de la classification...")
    
    # Lancer la classification
    results_dict = perform_classification(
        custom_config, 
        weights=uniform_weights, 
        compare_with_weights=False, 
        enable_validation=True
    )
    
    # Enregistrer la configuration utilisée
    config_path = os.path.join(output_dir, "config_utilisee.json")
    with open(config_path, 'w') as f:
        json.dump(custom_config.__dict__, f, indent=2)
    
    print(f"Configuration enregistrée dans: {config_path}")
    
    # Vérifier que le fichier TIF a été généré
    if os.path.exists(custom_config.save_raster_path):
        print(f"\nFichier TIF généré avec succès: {custom_config.save_raster_path}")
    else:
        print("\nAttention: Le fichier TIF n'a pas été généré correctement.")
    
    # Afficher les résultats
    if results_dict:
        print("\nRésultats de la classification:")
        if 'accuracy_standard' in results_dict:
            print(f"  Précision validation: {results_dict['accuracy_standard']:.4f}")
            print(f"  Kappa validation: {results_dict['kappa_standard']:.4f}")
    
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

if __name__ == "__main__":
    main() 