"""
CLASSIFICATION SANS PONDÉRATION
===============================
Ce script exécute une classification standard sans pondération
des bandes pour servir de référence.
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
    print(" CLASSIFICATION SANS PONDÉRATION ")
    print("=" * 70)
    
    # Charger la configuration
    config = Config()
    
    # Créer des poids uniformes (tous à 1.0)
    uniform_weights = np.ones(7)  # 7 bandes avec des poids de 1.0
    band_names = ["B2 - Bleu", "B3 - Vert", "B4 - Rouge", "B5 - RedEdge05", 
                 "B6 - RedEdge06", "B7 - RedEdge07", "B8 - PIR"]
    
    # Afficher les informations
    print(f"Utilisation de poids uniformes (sans pondération):")
    print("\nPoids par bande:")
    for i, (band, weight) in enumerate(zip(band_names, uniform_weights)):
        print(f"  {band}: {weight}")
    
    # Créer un répertoire de sortie avec horodatage
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(config["output_dir"], f"unweighted_classification_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\nRépertoire de sortie: {output_dir}")
    
    # S'assurer que la validation est activée et correctement configurée
    custom_config = {
        "output_dir": output_dir,
        "validation": {
            "enabled": True,
            "shapefile_path": r"D:\UQTR\Hiver 2025\Télédétection\TP3\points_validation.shp",
            "class_column": "Class_code"
        },
        "comparison": {
            "enabled": True,
            "raster_path": r"D:\UQTR\Hiver 2025\Télédétection\TP3\resultats\classification_mlc.tif"
        }
    }
    
    # Exécuter la classification avec les poids uniformes
    print("\nLancement de la classification...")
    results = run_classification(output_dir=output_dir, weights=uniform_weights, custom_config=custom_config)
    
    if results:
        print("\nRésultats de la classification:")
        print(f"  Précision: {results.get('accuracy_weighted', 0):.4f}")
        print(f"  Kappa: {results.get('kappa_weighted', 0):.4f}")
    
    print("\nClassification terminée avec succès!")

if __name__ == "__main__":
    main() 