"""
OPTIMISATION FINE DES POIDS
=========================
Ce script effectue une optimisation fine des poids autour de valeurs proches de 1
pour trouver la meilleure combinaison de pondérations pour la classification.
"""

import os
import sys
import time
import numpy as np
import random
import json
import traceback
from datetime import datetime
import matplotlib.pyplot as plt

# Importer les fonctions du script principal
from main import run_classification, log_classification_results

def generate_weights_around_one(num_bands=7, deviation=0.3):
    """
    Génère des poids aléatoires autour de 1 avec une déviation contrôlée.
    
    Args:
        num_bands (int): Nombre de bandes
        deviation (float): Écart maximum par rapport à 1
        
    Returns:
        numpy.ndarray: Tableau de poids
    """
    return np.array([round(random.uniform(1 - deviation, 1 + deviation), 2) 
                    for _ in range(num_bands)])

def run_fine_tuning(num_iterations=10, 
                   num_bands=7, 
                   deviation=0.3,
                   output_dir=None,
                   fitness_metric="kappa"):
    """
    Exécute une optimisation fine des poids autour de 1.
    
    Args:
        num_iterations (int): Nombre d'itérations à exécuter
        num_bands (int): Nombre de bandes
        deviation (float): Écart maximum par rapport à 1
        output_dir (str): Répertoire de sortie
        fitness_metric (str): Métrique d'évaluation ("kappa" ou "precision")
        
    Returns:
        tuple: Meilleurs poids et meilleur score
    """
    print("=" * 70)
    print(" OPTIMISATION FINE DES POIDS - DÉMARRAGE ")
    print("=" * 70)
    print(f"Métrique d'optimisation: {fitness_metric}")
    print(f"Nombre d'itérations: {num_iterations}")
    print(f"Déviation autour de 1: ±{deviation}")
    
    # Préparer le répertoire de sortie
    if output_dir is None:
        output_dir = r"D:\UQTR\Hiver 2025\Télédétection\TP3\resultats"
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    opt_dir = os.path.join(output_dir, f"fine_tune_{timestamp}")
    os.makedirs(opt_dir, exist_ok=True)
    
    # Variables pour suivre les meilleurs résultats
    best_weights = None
    best_fitness = 0
    all_results = []
    
    # Exécuter les itérations
    for i in range(num_iterations):
        print(f"\n{'='*30} ITÉRATION {i+1}/{num_iterations} {'='*30}")
        
        # Générer des poids autour de 1
        weights = generate_weights_around_one(num_bands, deviation)
        print(f"Poids générés: {weights}")
        
        try:
            # Créer un sous-répertoire pour cette itération
            iter_dir = os.path.join(opt_dir, f"iter_{i+1}")
            os.makedirs(iter_dir, exist_ok=True)
            
            # Configurer cette exécution
            custom_config = {
                "output_dir": iter_dir,
                "skip_visualizations": True  # Désactiver les visualisations pour accélérer
            }
            
            # Exécuter la classification avec ces poids
            result = run_classification(output_dir=iter_dir, custom_config=custom_config, weights=weights)
            
            if result and isinstance(result, dict):
                # Déterminer le score de fitness selon la métrique choisie
                if fitness_metric == "kappa":
                    fitness = result.get("kappa_weighted", 0)
                else:  # precision
                    fitness = result.get("accuracy_weighted", 0)
                
                print(f"Fitness ({fitness_metric}): {fitness:.4f}")
                
                # Enregistrer ce résultat
                result_info = {
                    "iteration": i+1,
                    "weights": weights.tolist(),
                    "fitness": float(fitness),
                    "accuracy": float(result.get("accuracy_weighted", 0)),
                    "kappa": float(result.get("kappa_weighted", 0))
                }
                all_results.append(result_info)
                
                # Mettre à jour le meilleur résultat si nécessaire
                if fitness > best_fitness:
                    best_fitness = fitness
                    best_weights = weights.copy()
                    print(f"\n*** NOUVEAU MEILLEUR RÉSULTAT TROUVÉ ***")
                    print(f"Fitness: {fitness:.4f}")
                    print(f"Poids: {weights}")
            else:
                print("Échec de l'évaluation, ignoré")
                
        except Exception as e:
            print(f"Erreur lors de l'itération {i+1}: {str(e)}")
            print(traceback.format_exc())
    
    # Enregistrer tous les résultats
    results_file = os.path.join(opt_dir, "all_results.json")
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    
    # Enregistrer le meilleur résultat
    if best_weights is not None:
        best_file = os.path.join(opt_dir, "best_weights.json")
        with open(best_file, 'w', encoding='utf-8') as f:
            json.dump({
                "weights": best_weights.tolist(),
                "fitness": float(best_fitness),
                "metric": fitness_metric
            }, f, indent=2, ensure_ascii=False)
    
    # Générer un graphique des résultats
    plot_results(all_results, opt_dir, fitness_metric)
    
    print("\n" + "=" * 70)
    print(" OPTIMISATION FINE DES POIDS - TERMINÉE ")
    print("=" * 70)
    
    if best_weights is not None:
        print(f"\nMeilleur résultat trouvé:")
        print(f"  Fitness ({fitness_metric}): {best_fitness:.4f}")
        print(f"  Poids: {best_weights}")
        print(f"\nRésultats enregistrés dans: {opt_dir}")
        
        # Exécuter une classification finale avec les meilleurs poids
        print("\nExécution d'une classification finale avec les meilleurs poids...")
        final_dir = os.path.join(opt_dir, "final_classification")
        os.makedirs(final_dir, exist_ok=True)
        
        run_classification(output_dir=final_dir, weights=best_weights)
    else:
        print("\nAucun résultat valide n'a été trouvé.")
    
    return best_weights, best_fitness

def plot_results(results, output_dir, fitness_metric):
    """
    Génère des graphiques pour visualiser les résultats de l'optimisation.
    
    Args:
        results (list): Liste des résultats de chaque itération
        output_dir (str): Répertoire de sortie
        fitness_metric (str): Métrique d'évaluation utilisée
    """
    if not results:
        print("Aucun résultat à visualiser")
        return
    
    # Extraire les données
    iterations = [r["iteration"] for r in results]
    fitness_values = [r["fitness"] for r in results]
    weights_history = np.array([r["weights"] for r in results])
    
    # Créer la figure
    plt.figure(figsize=(12, 8))
    
    # Graphique des valeurs de fitness
    plt.subplot(2, 1, 1)
    plt.plot(iterations, fitness_values, 'bo-', linewidth=2)
    plt.axhline(y=max(fitness_values), color='r', linestyle='--', alpha=0.7)
    plt.text(iterations[-1], max(fitness_values), f' Max: {max(fitness_values):.4f}', 
             verticalalignment='bottom', horizontalalignment='right', color='r')
    plt.xlabel('Itération')
    plt.ylabel(f'Fitness ({fitness_metric})')
    plt.title('Évolution de la qualité de classification')
    plt.grid(True)
    
    # Graphique des poids
    plt.subplot(2, 1, 2)
    
    # Noms des bandes
    band_names = ["B2 - Bleu", "B3 - Vert", "B4 - Rouge", "B5 - RedEdge05", 
                  "B6 - RedEdge06", "B7 - RedEdge07", "B8 - PIR"]
    
    # Tracer chaque bande
    for i in range(weights_history.shape[1]):
        if i < len(band_names):
            label = band_names[i]
        else:
            label = f"Bande {i+1}"
        plt.plot(iterations, weights_history[:, i], 'o-', label=label)
    
    plt.axhline(y=1.0, color='k', linestyle='--', alpha=0.5)
    plt.xlabel('Itération')
    plt.ylabel('Poids')
    plt.title('Évolution des poids des bandes')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "optimization_results.png"), dpi=300)
    plt.close()

def main():
    """Point d'entrée principal du programme."""
    # Paramètres par défaut
    num_iterations = 10
    deviation = 0.3
    num_bands = 7
    fitness_metric = "kappa"  # "kappa" ou "precision"
    output_dir = r"D:\UQTR\Hiver 2025\Télédétection\TP3\resultats"
    
    # Traiter les arguments de ligne de commande
    if len(sys.argv) > 1:
        try:
            num_iterations = int(sys.argv[1])
        except ValueError:
            print(f"Nombre d'itérations invalide: {sys.argv[1]}")
            print(f"Utilisation de la valeur par défaut: {num_iterations}")
    
    if len(sys.argv) > 2:
        try:
            deviation = float(sys.argv[2])
        except ValueError:
            print(f"Déviation invalide: {sys.argv[2]}")
            print(f"Utilisation de la valeur par défaut: {deviation}")
    
    if len(sys.argv) > 3:
        fitness_metric = sys.argv[3].lower()
        if fitness_metric not in ["kappa", "precision"]:
            print(f"Métrique invalide: {fitness_metric}")
            print("Utilisation de la métrique par défaut: kappa")
            fitness_metric = "kappa"
    
    if len(sys.argv) > 4:
        output_dir = sys.argv[4]
    
    # Exécuter l'optimisation fine
    run_fine_tuning(
        num_iterations=num_iterations,
        num_bands=num_bands,
        deviation=deviation,
        output_dir=output_dir,
        fitness_metric=fitness_metric
    )

if __name__ == "__main__":
    main()
