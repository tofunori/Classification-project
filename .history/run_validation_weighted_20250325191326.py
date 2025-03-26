"""
VALIDATION AVEC POIDS OPTIMISÉS PAR CLASSE
=========================================
Ce script exécute une classification avec poids optimisés et une 
validation basée uniquement sur les points de validation.
"""

import os
import sys
from datetime import datetime
from modules.config import Config
from main import run_classification
import numpy as np

def main():
    """Point d'entrée principal du programme."""
    print("=" * 70)
    print(" VALIDATION AVEC POIDS OPTIMISÉS (BASÉE SUR LES POINTS DE RÉFÉRENCE) ")
    print("=" * 70)
    
    # Charger la configuration
    config = Config()
    
    # Récupérer les poids optimisés par classe
    class_weights = config["class_optimized_weights"]["weights"]
    class_description = config["class_optimized_weights"]["description"]
    band_names = config["class_optimized_weights"]["band_names"]
    target_classes = config["class_optimized_weights"]["target_classes"]
    
    # Afficher les informations
    print(f"Utilisation des poids optimisés par classe:")
    print(f"  {class_description}")
    print(f"  Classes cibles: {', '.join([config['class_names'][c] for c in target_classes])}")
    
    print("\nPoids par bande:")
    for i, (band, weight) in enumerate(zip(band_names, class_weights)):
        print(f"  {band}: {weight}")
    
    # Créer un répertoire de sortie avec horodatage
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(config["output_dir"], f"validation_weighted_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\nRépertoire de sortie: {output_dir}")
    
    # S'assurer que la validation est activée et la comparaison désactivée
    custom_config = {
        "output_dir": output_dir,
        "validation": {
            "enabled": True,
            "shapefile_path": r"D:\UQTR\Hiver 2025\Télédétection\TP3\points_validation.shp",
            "class_column": "Class_code"
        },
        "comparison": {
            "enabled": False  # Désactiver la comparaison avec le raster de référence
        }
    }
    
    # Exécuter la classification avec les poids optimisés par classe
    print("\nLancement de la classification...")
    results = run_classification(output_dir=output_dir, weights=class_weights, custom_config=custom_config)
    
    if results:
        print("\nRésultats de la classification:")
        print(f"  Précision validation: {results.get('accuracy_weighted', 0):.4f}")
        print(f"  Kappa validation: {results.get('kappa_weighted', 0):.4f}")
    
    print("\nValidation terminée avec succès!")
    print("\nVoir la matrice de confusion en pourcentage dans:")
    print(f"{output_dir}/matrice_confusion_pourcent.png")

if __name__ == "__main__":
    main() 