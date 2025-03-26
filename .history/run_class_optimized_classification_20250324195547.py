"""
CLASSIFICATION AVEC POIDS OPTIMISÉS PAR CLASSE
=================================
Ce script exécute une classification en utilisant les poids optimisés
spécifiquement pour améliorer les classes problématiques (Tourbière et Champs).
"""

import os
import sys
from datetime import datetime
from modules.config import Config
from main import run_classification

def main():
    """Point d'entrée principal du programme."""
    print("=" * 70)
    print(" CLASSIFICATION AVEC POIDS OPTIMISÉS PAR CLASSE ")
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
    output_dir = os.path.join(config["output_dir"], f"class_optimized_classification_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\nRépertoire de sortie: {output_dir}")
    
    # Exécuter la classification avec les poids optimisés par classe
    print("\nLancement de la classification...")
    results = run_classification(output_dir=output_dir, weights=class_weights)
    
    if results:
        print("\nRésultats de la classification:")
        print(f"  Précision: {results.get('accuracy_weighted', 0):.4f}")
        print(f"  Kappa: {results.get('kappa_weighted', 0):.4f}")
        
        # Afficher les précisions par classe
        print("\nPrécision par classe:")
        for class_id in sorted(config["class_names"].keys()):
            class_name = config["class_names"][class_id]
            class_accuracy = results.get(f"class_{class_id}_accuracy", 0)
            print(f"  Classe {class_id} ({class_name}): {class_accuracy:.4f}")
            
            # Mettre en évidence les classes cibles
            if class_id in target_classes:
                print(f"    ^ Classe cible optimisée")
    
    print("\nClassification terminée avec succès!")

if __name__ == "__main__":
    main()
