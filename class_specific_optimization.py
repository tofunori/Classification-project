"""
OPTIMISATION SPÉCIFIQUE PAR CLASSE
=================================
Ce script implémente une optimisation des poids spécifique à chaque classe
pour améliorer la classification des classes problématiques.
"""

import os
import sys
import numpy as np
import random
import json
from datetime import datetime
import matplotlib.pyplot as plt

# Importer les fonctions du script principal
from main import run_classification
from modules.config import Config

def optimize_class_specific_weights(num_iterations=10, 
                                   target_classes=[3, 5],  # Tourbière et Champs
                                   output_dir=None):
    """
    Optimise les poids spécifiquement pour améliorer certaines classes.
    
    Args:
        num_iterations (int): Nombre d'itérations
        target_classes (list): Classes à améliorer en priorité
        output_dir (str): Répertoire de sortie
    """
    print("=" * 70)
    print(" OPTIMISATION SPÉCIFIQUE PAR CLASSE - DÉMARRAGE ")
    print("=" * 70)
    
    # Charger la configuration et les poids optimaux existants
    config = Config()
    base_weights = np.array(config["optimal_weights"]["weights"])
    band_names = config["optimal_weights"]["band_names"]
    class_names = config["class_names"]
    
    # Préparer le répertoire de sortie
    if output_dir is None:
        output_dir = r"D:\UQTR\Hiver 2025\Télédétection\TP3\resultats"
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    opt_dir = os.path.join(output_dir, f"class_opt_{timestamp}")
    os.makedirs(opt_dir, exist_ok=True)
    
    # Afficher les informations initiales
    print(f"Poids de base: {base_weights}")
    print(f"Classes cibles:")
    for class_id in target_classes:
        print(f"  - Classe {class_id}: {class_names[class_id]}")
    
    # Variables pour suivre les meilleurs résultats
    best_weights = None
    best_fitness = 0
    all_results = []
    
    # Fonction pour générer des poids avec focus sur certaines bandes
    def generate_class_focused_weights(base_weights, target_class):
        # Analyse PCA montre que différentes classes sont mieux séparées par différentes bandes
        class_band_focus = {
            3: [2, 3, 4],  # Tourbière: focus sur bandes visibles
            5: [5, 7, 8]   # Champs: focus sur RedEdge et PIR
        }
        
        # Copier les poids de base
        new_weights = base_weights.copy()
        
        # Augmenter légèrement les poids des bandes importantes pour cette classe
        if target_class in class_band_focus:
            focus_bands = class_band_focus[target_class]
            for i in range(len(new_weights)):
                band_num = i + 2  # Convertir l'index en numéro de bande
                if band_num in focus_bands:
                    # Augmenter le poids de 10-30%
                    new_weights[i] *= random.uniform(1.1, 1.3)
                else:
                    # Réduire légèrement les autres poids
                    new_weights[i] *= random.uniform(0.9, 1.0)
        
        # Arrondir les poids
        return np.round(new_weights, 2)
    
    # Exécuter les itérations
    for i in range(num_iterations):
        print(f"\n{'='*30} ITÉRATION {i+1}/{num_iterations} {'='*30}")
        
        # Sélectionner une classe cible aléatoire parmi les classes à améliorer
        target_class = random.choice(target_classes)
        
        # Générer des poids avec focus sur cette classe
        weights = generate_class_focused_weights(base_weights, target_class)
        print(f"Classe cible: {class_names[target_class]}")
        print(f"Poids générés: {weights}")
        
        try:
            # Créer un sous-répertoire pour cette itération
            iter_dir = os.path.join(opt_dir, f"iter_{i+1}_class_{target_class}")
            os.makedirs(iter_dir, exist_ok=True)
            
            # Configurer cette exécution
            custom_config = {
                "output_dir": iter_dir,
                "skip_visualizations": True  # Désactiver les visualisations pour accélérer
            }
            
            # Exécuter la classification avec ces poids
            result = run_classification(output_dir=iter_dir, custom_config=custom_config, weights=weights)
            
            if result and isinstance(result, dict):
                # Évaluer les résultats avec un focus sur la classe cible
                # Extraire la précision de la classe cible à partir de la matrice de confusion
                class_accuracy = result.get(f"class_{target_class}_accuracy", 0)
                
                # Calculer un score combiné (70% précision de la classe cible, 30% Kappa global)
                class_weight = 0.7
                global_weight = 0.3
                fitness = (class_weight * class_accuracy + 
                          global_weight * result.get("kappa_weighted", 0))
                
                print(f"Précision classe {target_class}: {class_accuracy:.4f}")
                print(f"Kappa global: {result.get('kappa_weighted', 0):.4f}")
                print(f"Score combiné: {fitness:.4f}")
                
                # Enregistrer ce résultat
                result_info = {
                    "iteration": i+1,
                    "target_class": target_class,
                    "weights": weights.tolist(),
                    "class_accuracy": float(class_accuracy),
                    "global_kappa": float(result.get("kappa_weighted", 0)),
                    "fitness": float(fitness)
                }
                all_results.append(result_info)
                
                # Mettre à jour le meilleur résultat si nécessaire
                if fitness > best_fitness:
                    best_fitness = fitness
                    best_weights = weights.copy()
                    print(f"\n*** NOUVEAU MEILLEUR RÉSULTAT TROUVÉ ***")
                    print(f"Score combiné: {fitness:.4f}")
                    print(f"Poids: {weights}")
            else:
                print("Échec de l'évaluation, ignoré")
                
        except Exception as e:
            print(f"Erreur lors de l'itération {i+1}: {str(e)}")
    
    # Enregistrer tous les résultats
    results_file = os.path.join(opt_dir, "all_results.json")
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    
    # Enregistrer le meilleur résultat
    if best_weights is not None:
        best_file = os.path.join(opt_dir, "best_class_weights.json")
        with open(best_file, 'w', encoding='utf-8') as f:
            json.dump({
                "weights": best_weights.tolist(),
                "fitness": float(best_fitness),
                "description": "Poids optimisés pour les classes spécifiques"
            }, f, indent=2, ensure_ascii=False)
    
    print("\n" + "=" * 70)
    print(" OPTIMISATION SPÉCIFIQUE PAR CLASSE - TERMINÉE ")
    print("=" * 70)
    
    if best_weights is not None:
        print(f"\nMeilleur résultat trouvé:")
        print(f"  Score combiné: {best_fitness:.4f}")
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

def main():
    """Point d'entrée principal du programme."""
    # Paramètres par défaut
    num_iterations = 10
    target_classes = [3, 5]  # Tourbière et Champs
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
            target_classes = [int(c) for c in sys.argv[2].split(',')]
        except ValueError:
            print(f"Classes cibles invalides: {sys.argv[2]}")
            print(f"Utilisation de la valeur par défaut: {target_classes}")
    
    if len(sys.argv) > 3:
        output_dir = sys.argv[3]
    
    # Exécuter l'optimisation spécifique par classe
    optimize_class_specific_weights(
        num_iterations=num_iterations,
        target_classes=target_classes,
        output_dir=output_dir
    )

if __name__ == "__main__":
    main()
