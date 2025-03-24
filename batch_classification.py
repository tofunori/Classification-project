"""
BATCH CLASSIFICATION
==================
Ce script exécute des classifications avec différentes pondérations de bandes
et enregistre les résultats dans le fichier de log.
"""

import os
import sys
import time
import numpy as np
import random
import traceback
from datetime import datetime

# Importer les fonctions du script principal
from main import run_classification, log_classification_results

def generate_random_weights(num_bands=7, min_weight=0.5, max_weight=5.0):
    """
    Génère un tableau de poids aléatoires pour les bandes.
    
    Args:
        num_bands (int): Nombre de bandes à pondérer
        min_weight (float): Poids minimum
        max_weight (float): Poids maximum
        
    Returns:
        numpy.ndarray: Tableau de poids aléatoires
    """
    return np.array([round(random.uniform(min_weight, max_weight), 1) for _ in range(num_bands)])

def generate_grid_weights(num_bands=7, min_weight=0.5, max_weight=5.0, steps=3):
    """
    Génère des combinaisons de poids selon une grille.
    
    Args:
        num_bands (int): Nombre de bandes à pondérer
        min_weight (float): Poids minimum
        max_weight (float): Poids maximum
        steps (int): Nombre de valeurs différentes par bande
        
    Returns:
        list: Liste de tableaux de poids
    """
    # Générer les valeurs possibles pour chaque bande
    values = np.linspace(min_weight, max_weight, steps)
    
    # Générer toutes les combinaisons possibles (attention, peut être très grand!)
    # Pour limiter, on va sélectionner un sous-ensemble aléatoire
    
    # Calculer le nombre total de combinaisons
    total_combinations = steps ** num_bands
    print(f"Nombre total de combinaisons possibles: {total_combinations}")
    
    # Si trop de combinaisons, limiter à un nombre raisonnable
    max_combinations = 100
    if total_combinations > max_combinations:
        print(f"Limitation à {max_combinations} combinaisons aléatoires")
        combinations = []
        for _ in range(max_combinations):
            weights = np.array([random.choice(values) for _ in range(num_bands)])
            combinations.append(weights)
    else:
        # Générer toutes les combinaisons
        from itertools import product
        combinations = list(product(values, repeat=num_bands))
        combinations = [np.array(c) for c in combinations]
    
    return combinations

def batch_classification(num_iterations=100, method='random', output_dir=None):
    """
    Exécute plusieurs classifications avec différentes pondérations.
    
    Args:
        num_iterations (int): Nombre d'itérations à exécuter
        method (str): Méthode de génération des poids ('random' ou 'grid')
        output_dir (str, optional): Répertoire de sortie
        
    Returns:
        bool: True si le processus s'est terminé avec succès, False sinon
    """
    print("=" * 70)
    print(" CLASSIFICATION PAR LOT - DÉMARRAGE ")
    print("=" * 70)
    
    # Configuration par défaut
    if output_dir is None:
        output_dir = r"D:\UQTR\Hiver 2025\Télédétection\TP3\resultats"
    
    # Créer un sous-répertoire pour cette exécution par lot
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    batch_dir = os.path.join(output_dir, f"batch_{timestamp}")
    os.makedirs(batch_dir, exist_ok=True)
    
    # Générer les poids
    if method == 'random':
        # Générer des poids aléatoires pour chaque itération
        weight_sets = [generate_random_weights() for _ in range(num_iterations)]
    elif method == 'grid':
        # Générer des poids selon une grille
        weight_sets = generate_grid_weights(steps=int(np.cbrt(num_iterations)))
        # Limiter au nombre d'itérations demandé
        weight_sets = weight_sets[:num_iterations]
    else:
        print(f"Méthode de génération de poids inconnue: {method}")
        return False
    
    print(f"Nombre de combinaisons de poids à tester: {len(weight_sets)}")
    
    # Exécuter les classifications
    results = []
    for i, weights in enumerate(weight_sets):
        print(f"\n--- Itération {i+1}/{len(weight_sets)} ---")
        print(f"Poids: {weights}")
        
        try:
            # Créer un sous-répertoire pour cette itération
            iter_dir = os.path.join(batch_dir, f"iter_{i+1}")
            os.makedirs(iter_dir, exist_ok=True)
            
            # Configurer cette exécution
            custom_config = {
                "output_dir": iter_dir,
                # Désactiver certaines visualisations pour accélérer le traitement
                "skip_visualizations": True
            }
            
            # Exécuter la classification avec ces poids
            result = run_classification(output_dir=iter_dir, custom_config=custom_config, weights=weights)
            
            if result and isinstance(result, dict):
                # Enregistrer les résultats
                results.append({
                    "iteration": i+1,
                    "weights": weights.tolist(),
                    "accuracy_std": result.get("accuracy_std", 0),
                    "kappa_std": result.get("kappa_std", 0),
                    "accuracy_weighted": result.get("accuracy_weighted", 0),
                    "kappa_weighted": result.get("kappa_weighted", 0),
                    "variance_explained": result.get("variance_explained", [])
                })
                
                # Enregistrer dans le fichier de log principal
                log_classification_results(
                    output_dir,
                    weights,
                    result.get("accuracy_std", 0),
                    result.get("kappa_std", 0),
                    result.get("accuracy_weighted", 0),
                    result.get("kappa_weighted", 0),
                    result.get("variance_explained", [])
                )
            
        except Exception as e:
            print(f"Erreur lors de l'itération {i+1}: {str(e)}")
            print(traceback.format_exc())
    
    # Enregistrer un résumé des résultats
    try:
        import json
        summary_file = os.path.join(batch_dir, "batch_summary.json")
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\nRésumé des résultats enregistré dans: {summary_file}")
    except Exception as e:
        print(f"Erreur lors de l'enregistrement du résumé: {str(e)}")
    
    print("\n" + "=" * 70)
    print(" CLASSIFICATION PAR LOT - TERMINÉE ")
    print("=" * 70)
    
    return True

def main():
    """Point d'entrée principal du programme."""
    # Paramètres par défaut
    num_iterations = 100
    method = 'random'
    output_dir = r"D:\UQTR\Hiver 2025\Télédétection\TP3\resultats"
    
    # Traiter les arguments de ligne de commande
    if len(sys.argv) > 1:
        try:
            num_iterations = int(sys.argv[1])
        except ValueError:
            print(f"Nombre d'itérations invalide: {sys.argv[1]}")
            print(f"Utilisation du nombre par défaut: {num_iterations}")
    
    if len(sys.argv) > 2:
        method = sys.argv[2]
        if method not in ['random', 'grid']:
            print(f"Méthode invalide: {method}")
            print("Utilisation de la méthode par défaut: random")
            method = 'random'
    
    if len(sys.argv) > 3:
        output_dir = sys.argv[3]
    
    # Exécuter le traitement par lot
    batch_classification(num_iterations, method, output_dir)

if __name__ == "__main__":
    main()
