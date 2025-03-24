"""
OPTIMISATION DES POIDS
=====================
Ce script utilise un algorithme génétique pour optimiser les pondérations des bandes
afin d'améliorer les résultats de classification.
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

class GeneticOptimizer:
    """Classe pour l'optimisation des poids par algorithme génétique"""
    
    def __init__(self, 
                 num_bands=7, 
                 population_size=20, 
                 generations=10, 
                 mutation_rate=0.2, 
                 crossover_rate=0.7,
                 min_weight=0.5,
                 max_weight=5.0,
                 output_dir=None,
                 fitness_metric="kappa"):
        """
        Initialise l'optimiseur génétique.
        
        Args:
            num_bands (int): Nombre de bandes à pondérer
            population_size (int): Taille de la population
            generations (int): Nombre de générations
            mutation_rate (float): Taux de mutation (0-1)
            crossover_rate (float): Taux de croisement (0-1)
            min_weight (float): Poids minimum
            max_weight (float): Poids maximum
            output_dir (str): Répertoire de sortie
            fitness_metric (str): Métrique pour évaluer la qualité ("kappa" ou "precision")
        """
        self.num_bands = num_bands
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.min_weight = min_weight
        self.max_weight = max_weight
        self.output_dir = output_dir or r"D:\UQTR\Hiver 2025\Télédétection\TP3\resultats"
        self.fitness_metric = fitness_metric
        
        # Créer un sous-répertoire pour cette optimisation
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.opt_dir = os.path.join(self.output_dir, f"optimize_{timestamp}")
        os.makedirs(self.opt_dir, exist_ok=True)
        
        # Historique pour suivre l'évolution
        self.history = {
            "best_fitness": [],
            "avg_fitness": [],
            "best_weights": [],
            "generation": []
        }
        
        # Meilleur individu trouvé
        self.best_individual = None
        self.best_fitness = 0
        
    def generate_individual(self):
        """Génère un individu aléatoire (ensemble de poids)"""
        return np.array([round(random.uniform(self.min_weight, self.max_weight), 1) 
                        for _ in range(self.num_bands)])
    
    def initialize_population(self):
        """Initialise la population avec des individus aléatoires"""
        return [self.generate_individual() for _ in range(self.population_size)]
    
    def evaluate_individual(self, individual, generation, individual_idx):
        """
        Évalue un individu en exécutant la classification avec ses poids.
        
        Args:
            individual (numpy.ndarray): Ensemble de poids à évaluer
            generation (int): Numéro de génération
            individual_idx (int): Index de l'individu dans la population
            
        Returns:
            float: Score de fitness (précision ou kappa)
        """
        print(f"\n--- Génération {generation+1}/{self.generations}, Individu {individual_idx+1}/{self.population_size} ---")
        print(f"Poids: {individual}")
        
        try:
            # Créer un sous-répertoire pour cet individu
            indiv_dir = os.path.join(self.opt_dir, f"gen_{generation+1}_indiv_{individual_idx+1}")
            os.makedirs(indiv_dir, exist_ok=True)
            
            # Configurer cette exécution
            custom_config = {
                "output_dir": indiv_dir,
                "skip_visualizations": True  # Désactiver les visualisations pour accélérer
            }
            
            # Exécuter la classification avec ces poids
            result = run_classification(output_dir=indiv_dir, custom_config=custom_config, weights=individual)
            
            if result and isinstance(result, dict):
                # Déterminer le score de fitness selon la métrique choisie
                if self.fitness_metric == "kappa":
                    fitness = result.get("kappa_weighted", 0)
                else:  # precision
                    fitness = result.get("accuracy_weighted", 0)
                
                print(f"Fitness ({self.fitness_metric}): {fitness:.4f}")
                return fitness
            else:
                print("Échec de l'évaluation, retourne fitness 0")
                return 0
            
        except Exception as e:
            print(f"Erreur lors de l'évaluation: {str(e)}")
            print(traceback.format_exc())
            return 0
    
    def evaluate_population(self, population, generation):
        """
        Évalue tous les individus de la population.
        
        Args:
            population (list): Liste des individus à évaluer
            generation (int): Numéro de génération
            
        Returns:
            list: Liste des scores de fitness
        """
        fitness_scores = []
        for i, individual in enumerate(population):
            fitness = self.evaluate_individual(individual, generation, i)
            fitness_scores.append(fitness)
            
            # Mettre à jour le meilleur individu si nécessaire
            if fitness > self.best_fitness:
                self.best_fitness = fitness
                self.best_individual = individual.copy()
                print(f"\n*** NOUVEAU MEILLEUR INDIVIDU TROUVÉ ***")
                print(f"Fitness: {fitness:.4f}")
                print(f"Poids: {individual}")
        
        return fitness_scores
    
    def select_parents(self, population, fitness_scores):
        """
        Sélectionne des parents pour la reproduction en utilisant la sélection par tournoi.
        
        Args:
            population (list): Population actuelle
            fitness_scores (list): Scores de fitness correspondants
            
        Returns:
            tuple: Deux parents sélectionnés
        """
        # Sélection par tournoi
        def tournament_selection():
            # Sélectionner aléatoirement k individus et prendre le meilleur
            k = 3  # Taille du tournoi
            selected_indices = random.sample(range(len(population)), k)
            tournament_fitness = [fitness_scores[i] for i in selected_indices]
            winner_idx = selected_indices[np.argmax(tournament_fitness)]
            return population[winner_idx]
        
        parent1 = tournament_selection()
        parent2 = tournament_selection()
        
        return parent1, parent2
    
    def crossover(self, parent1, parent2):
        """
        Effectue un croisement entre deux parents.
        
        Args:
            parent1 (numpy.ndarray): Premier parent
            parent2 (numpy.ndarray): Deuxième parent
            
        Returns:
            tuple: Deux enfants issus du croisement
        """
        if random.random() < self.crossover_rate:
            # Point de croisement aléatoire
            crossover_point = random.randint(1, self.num_bands - 1)
            
            # Créer les enfants
            child1 = np.concatenate((parent1[:crossover_point], parent2[crossover_point:]))
            child2 = np.concatenate((parent2[:crossover_point], parent1[crossover_point:]))
            
            return child1, child2
        else:
            # Pas de croisement, retourner des copies des parents
            return parent1.copy(), parent2.copy()
    
    def mutate(self, individual):
        """
        Applique une mutation à un individu.
        
        Args:
            individual (numpy.ndarray): Individu à muter
            
        Returns:
            numpy.ndarray: Individu muté
        """
        mutated = individual.copy()
        
        for i in range(len(mutated)):
            if random.random() < self.mutation_rate:
                # Mutation: ajouter ou soustraire une valeur aléatoire
                delta = random.uniform(-0.5, 0.5)
                mutated[i] = round(max(self.min_weight, min(self.max_weight, mutated[i] + delta)), 1)
        
        return mutated
    
    def create_next_generation(self, population, fitness_scores):
        """
        Crée la prochaine génération à partir de la population actuelle.
        
        Args:
            population (list): Population actuelle
            fitness_scores (list): Scores de fitness correspondants
            
        Returns:
            list: Nouvelle génération
        """
        new_population = []
        
        # Élitisme: conserver le meilleur individu
        elite_idx = np.argmax(fitness_scores)
        new_population.append(population[elite_idx])
        
        # Créer le reste de la population par sélection, croisement et mutation
        while len(new_population) < self.population_size:
            # Sélectionner les parents
            parent1, parent2 = self.select_parents(population, fitness_scores)
            
            # Croisement
            child1, child2 = self.crossover(parent1, parent2)
            
            # Mutation
            child1 = self.mutate(child1)
            child2 = self.mutate(child2)
            
            # Ajouter à la nouvelle population
            new_population.append(child1)
            if len(new_population) < self.population_size:
                new_population.append(child2)
        
        return new_population
    
    def update_history(self, generation, population, fitness_scores):
        """
        Met à jour l'historique de l'optimisation.
        
        Args:
            generation (int): Numéro de génération
            population (list): Population actuelle
            fitness_scores (list): Scores de fitness correspondants
        """
        best_idx = np.argmax(fitness_scores)
        best_fitness = fitness_scores[best_idx]
        avg_fitness = np.mean(fitness_scores)
        
        self.history["generation"].append(generation + 1)
        self.history["best_fitness"].append(best_fitness)
        self.history["avg_fitness"].append(avg_fitness)
        self.history["best_weights"].append(population[best_idx].tolist())
        
        # Enregistrer l'historique dans un fichier JSON
        history_file = os.path.join(self.opt_dir, "optimization_history.json")
        with open(history_file, 'w', encoding='utf-8') as f:
            json.dump(self.history, f, indent=2, ensure_ascii=False)
    
    def plot_progress(self):
        """Génère un graphique montrant l'évolution de l'optimisation"""
        plt.figure(figsize=(12, 8))
        
        # Tracer l'évolution du fitness
        plt.subplot(2, 1, 1)
        plt.plot(self.history["generation"], self.history["best_fitness"], 'b-', label='Meilleur fitness')
        plt.plot(self.history["generation"], self.history["avg_fitness"], 'r--', label='Fitness moyen')
        plt.xlabel('Génération')
        plt.ylabel(f'Fitness ({self.fitness_metric})')
        plt.title('Évolution du fitness au cours des générations')
        plt.legend()
        plt.grid(True)
        
        # Tracer l'évolution des poids du meilleur individu
        plt.subplot(2, 1, 2)
        
        # Préparer les données
        generations = self.history["generation"]
        weights_history = np.array(self.history["best_weights"])
        
        # Tracer chaque bande
        band_names = ["B2 - Bleu", "B3 - Vert", "B4 - Rouge", "B5 - RedEdge05", 
                      "B6 - RedEdge06", "B7 - RedEdge07", "B8 - PIR"]
        
        for i in range(self.num_bands):
            if i < len(band_names):
                label = band_names[i]
            else:
                label = f"Bande {i+1}"
            plt.plot(generations, weights_history[:, i], marker='o', label=label)
        
        plt.xlabel('Génération')
        plt.ylabel('Poids')
        plt.title('Évolution des poids du meilleur individu')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.opt_dir, "optimization_progress.png"), dpi=300)
        plt.close()
    
    def run(self):
        """
        Exécute l'algorithme génétique d'optimisation.
        
        Returns:
            tuple: Meilleur individu trouvé et son score de fitness
        """
        print("=" * 70)
        print(" OPTIMISATION GÉNÉTIQUE DES POIDS - DÉMARRAGE ")
        print("=" * 70)
        print(f"Métrique d'optimisation: {self.fitness_metric}")
        print(f"Taille de la population: {self.population_size}")
        print(f"Nombre de générations: {self.generations}")
        print(f"Taux de mutation: {self.mutation_rate}")
        print(f"Taux de croisement: {self.crossover_rate}")
        print(f"Plage de poids: [{self.min_weight}, {self.max_weight}]")
        
        # Initialiser la population
        population = self.initialize_population()
        
        # Boucle principale de l'algorithme génétique
        for generation in range(self.generations):
            print(f"\n{'='*30} GÉNÉRATION {generation+1}/{self.generations} {'='*30}")
            
            # Évaluer la population
            fitness_scores = self.evaluate_population(population, generation)
            
            # Mettre à jour l'historique
            self.update_history(generation, population, fitness_scores)
            
            # Générer la prochaine génération (sauf pour la dernière itération)
            if generation < self.generations - 1:
                population = self.create_next_generation(population, fitness_scores)
            
            # Afficher les statistiques de cette génération
            best_idx = np.argmax(fitness_scores)
            avg_fitness = np.mean(fitness_scores)
            print(f"\nStatistiques de la génération {generation+1}:")
            print(f"  Meilleur fitness: {fitness_scores[best_idx]:.4f}")
            print(f"  Fitness moyen: {avg_fitness:.4f}")
            print(f"  Meilleurs poids: {population[best_idx]}")
        
        # Générer le graphique d'évolution
        self.plot_progress()
        
        # Enregistrer le meilleur individu trouvé
        best_file = os.path.join(self.opt_dir, "best_weights.json")
        with open(best_file, 'w', encoding='utf-8') as f:
            json.dump({
                "weights": self.best_individual.tolist(),
                "fitness": float(self.best_fitness),
                "metric": self.fitness_metric
            }, f, indent=2, ensure_ascii=False)
        
        print("\n" + "=" * 70)
        print(" OPTIMISATION GÉNÉTIQUE DES POIDS - TERMINÉE ")
        print("=" * 70)
        print(f"\nMeilleur individu trouvé:")
        print(f"  Fitness ({self.fitness_metric}): {self.best_fitness:.4f}")
        print(f"  Poids: {self.best_individual}")
        print(f"\nRésultats enregistrés dans: {self.opt_dir}")
        
        return self.best_individual, self.best_fitness

def main():
    """Point d'entrée principal du programme."""
    # Paramètres par défaut
    population_size = 10
    generations = 5
    mutation_rate = 0.2
    crossover_rate = 0.7
    min_weight = 0.5
    max_weight = 5.0
    num_bands = 7
    fitness_metric = "kappa"  # "kappa" ou "precision"
    output_dir = r"D:\UQTR\Hiver 2025\Télédétection\TP3\resultats"
    
    # Traiter les arguments de ligne de commande
    if len(sys.argv) > 1:
        try:
            population_size = int(sys.argv[1])
        except ValueError:
            print(f"Taille de population invalide: {sys.argv[1]}")
            print(f"Utilisation de la valeur par défaut: {population_size}")
    
    if len(sys.argv) > 2:
        try:
            generations = int(sys.argv[2])
        except ValueError:
            print(f"Nombre de générations invalide: {sys.argv[2]}")
            print(f"Utilisation de la valeur par défaut: {generations}")
    
    if len(sys.argv) > 3:
        fitness_metric = sys.argv[3].lower()
        if fitness_metric not in ["kappa", "precision"]:
            print(f"Métrique invalide: {fitness_metric}")
            print("Utilisation de la métrique par défaut: kappa")
            fitness_metric = "kappa"
    
    if len(sys.argv) > 4:
        output_dir = sys.argv[4]
    
    # Créer et exécuter l'optimiseur
    optimizer = GeneticOptimizer(
        num_bands=num_bands,
        population_size=population_size,
        generations=generations,
        mutation_rate=mutation_rate,
        crossover_rate=crossover_rate,
        min_weight=min_weight,
        max_weight=max_weight,
        output_dir=output_dir,
        fitness_metric=fitness_metric
    )
    
    best_weights, best_fitness = optimizer.run()
    
    # Exécuter une classification finale avec les meilleurs poids
    print("\nExécution d'une classification finale avec les meilleurs poids...")
    final_dir = os.path.join(optimizer.opt_dir, "final_classification")
    os.makedirs(final_dir, exist_ok=True)
    
    run_classification(output_dir=final_dir, weights=best_weights)
    
    print("\nOptimisation terminée avec succès!")

if __name__ == "__main__":
    main()
