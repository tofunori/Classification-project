"""
COMPARAISON DES MATRICES DE CONFUSION
=====================================
Ce script compare les matrices de confusion des classifications pondérée et non pondérée.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.colors import LinearSegmentedColormap
import json
from datetime import datetime

def load_json_results(file_path):
    """Charger les résultats de classification depuis un fichier JSON."""
    with open(file_path, 'r') as file:
        return json.load(file)

def compare_confusion_matrices():
    """Compare les matrices de confusion des deux approches."""
    # Chemins des répertoires de résultats
    base_dir = r"D:\UQTR\Hiver 2025\Télédétection\TP3\resultats"
    
    # Trouver les répertoires les plus récents pour chaque type
    validation_only_dirs = [d for d in os.listdir(base_dir) if d.startswith('validation_only_')]
    validation_weighted_dirs = [d for d in os.listdir(base_dir) if d.startswith('validation_weighted_')]
    
    if not validation_only_dirs or not validation_weighted_dirs:
        print("Répertoires de résultats introuvables.")
        return
    
    # Trier par date (supposant que le format timestamp est le même)
    validation_only_dir = sorted(validation_only_dirs)[-1]
    validation_weighted_dir = sorted(validation_weighted_dirs)[-1]
    
    validation_only_path = os.path.join(base_dir, validation_only_dir)
    validation_weighted_path = os.path.join(base_dir, validation_weighted_dir)
    
    print(f"Utilisation des répertoires de résultats:")
    print(f"  Non pondéré: {validation_only_path}")
    print(f"  Pondéré: {validation_weighted_path}")
    
    # Charger les matrices de confusion
    try:
        # Charger les images des matrices de confusion
        non_weighted_matrix_img = mpimg.imread(os.path.join(validation_only_path, "matrice_confusion_pourcent.png"))
        weighted_matrix_img = mpimg.imread(os.path.join(validation_weighted_path, "matrice_confusion_pourcent.png"))
        
        # Charger les statistiques de classification
        non_weighted_stats = load_json_results(os.path.join(validation_only_path, "classification_stats_log.json"))
        weighted_stats = load_json_results(os.path.join(validation_weighted_path, "classification_stats_log.json"))
        
        # Extraire les précisions et kappas
        non_weighted_accuracy = non_weighted_stats[0].get('accuracy_standard', 0)
        non_weighted_kappa = non_weighted_stats[0].get('kappa_standard', 0)
        weighted_accuracy = weighted_stats[0].get('accuracy_weighted', 0)
        weighted_kappa = weighted_stats[0].get('kappa_weighted', 0)
        
        # Créer une figure comparative
        fig, axes = plt.subplots(1, 2, figsize=(16, 8))
        
        # Afficher les matrices
        axes[0].imshow(non_weighted_matrix_img)
        axes[0].set_title(f"Matrice de confusion (Non pondérée)\nPrécision: {non_weighted_accuracy:.4f}, Kappa: {non_weighted_kappa:.4f}", fontsize=12)
        axes[0].axis('off')
        
        axes[1].imshow(weighted_matrix_img)
        axes[1].set_title(f"Matrice de confusion (Pondérée)\nPrécision: {weighted_accuracy:.4f}, Kappa: {weighted_kappa:.4f}", fontsize=12)
        axes[1].axis('off')
        
        # Ajouter un titre général
        plt.suptitle("Comparaison des matrices de confusion (en pourcentage)", fontsize=16)
        plt.tight_layout()
        
        # Créer un répertoire pour la comparaison
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        comparison_dir = os.path.join(base_dir, f"comparison_{timestamp}")
        os.makedirs(comparison_dir, exist_ok=True)
        
        # Sauvegarder la comparaison
        comparison_path = os.path.join(comparison_dir, "comparaison_matrices.png")
        plt.savefig(comparison_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"\nComparaison sauvegardée dans: {comparison_path}")
        
        # Créer un tableau comparatif des métriques
        fig, ax = plt.subplots(figsize=(10, 6))
        metrics = ['Précision', 'Kappa']
        non_weighted_values = [non_weighted_accuracy, non_weighted_kappa]
        weighted_values = [weighted_accuracy, weighted_kappa]
        
        x = np.arange(len(metrics))
        width = 0.35
        
        ax.bar(x - width/2, non_weighted_values, width, label='Non pondérée')
        ax.bar(x + width/2, weighted_values, width, label='Pondérée')
        
        ax.set_ylim(0, 1)
        ax.set_ylabel('Valeur')
        ax.set_title('Comparaison des métriques de performance')
        ax.set_xticks(x)
        ax.set_xticklabels(metrics)
        ax.legend()
        
        # Ajouter les valeurs sur les barres
        for i, v in enumerate(non_weighted_values):
            ax.text(i - width/2, v + 0.01, f'{v:.4f}', ha='center')
        
        for i, v in enumerate(weighted_values):
            ax.text(i + width/2, v + 0.01, f'{v:.4f}', ha='center')
        
        # Sauvegarder le graphique comparatif
        metrics_path = os.path.join(comparison_dir, "comparaison_metriques.png")
        plt.savefig(metrics_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Graphique des métriques sauvegardé dans: {metrics_path}")
        
        # Calculer les différences entre les deux approches
        diff_accuracy = non_weighted_accuracy - weighted_accuracy
        diff_kappa = non_weighted_kappa - weighted_kappa
        
        print("\nRésultats comparatifs:")
        print(f"  Différence de précision: {diff_accuracy:.4f} ({diff_accuracy*100:.2f}%)")
        print(f"  Différence de Kappa: {diff_kappa:.4f} ({diff_kappa*100:.2f}%)")
        
        if diff_accuracy > 0 and diff_kappa > 0:
            print("\nConclusion: La classification non pondérée a donné de meilleurs résultats globaux.")
        elif diff_accuracy < 0 and diff_kappa < 0:
            print("\nConclusion: La classification pondérée a donné de meilleurs résultats globaux.")
        else:
            print("\nConclusion: Les résultats sont mitigés entre les deux approches.")
        
    except Exception as e:
        print(f"Erreur lors de la comparaison: {e}")

if __name__ == "__main__":
    compare_confusion_matrices() 