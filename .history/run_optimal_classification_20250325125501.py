"""
CLASSIFICATION AVEC POIDS OPTIMAUX
=================================
Ce script exécute une classification en utilisant les poids optimaux
déterminés par optimisation fine.
"""

import os
import sys
from datetime import datetime
from modules.config import Config
from main import run_classification

def main():
    """Point d'entrée principal du programme."""
    print("=" * 70)
    print(" CLASSIFICATION AVEC POIDS OPTIMAUX ")
    print("=" * 70)
    
    # Charger la configuration
    config = Config()
    
    # Récupérer les poids optimaux
    optimal_weights = config["optimal_weights"]["weights"]
    optimal_description = config["optimal_weights"]["description"]
    band_names = config["optimal_weights"]["band_names"]
    
    # Afficher les informations
    print(f"Utilisation des poids optimaux:")
    print(f"  {optimal_description}")
    print("\nPoids par bande:")
    for i, (band, weight) in enumerate(zip(band_names, optimal_weights)):
        print(f"  {band}: {weight}")
    
    # Créer un répertoire de sortie avec horodatage
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(config["output_dir"], f"optimal_classification_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\nRépertoire de sortie: {output_dir}")
    
    # Exécuter la classification avec les poids optimaux
    print("\nLancement de la classification...")
    results = run_classification(output_dir=output_dir, weights=optimal_weights)
    
    if results:
        print("\nRésultats de la classification:")
        print(f"  Précision: {results.get('accuracy_weighted', 0):.4f}")
        print(f"  Kappa: {results.get('kappa_weighted', 0):.4f}")
    
    print("\nClassification terminée avec succès!")

if __name__ == "__main__":
    main()
