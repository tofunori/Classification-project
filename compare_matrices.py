"""
COMPARAISON DES MATRICES DE CONFUSION
=====================================
Ce script compare les matrices de confusion des classifications pondérée et non pondérée.
Analyse en détail les performances par classe.
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
    try:
        with open(file_path, 'r') as file:
            return json.load(file)
    except json.JSONDecodeError:
        print(f"Erreur de décodage JSON dans le fichier: {file_path}")
        # Afficher le contenu du fichier pour le déboggage
        with open(file_path, 'r') as file:
            content = file.read()
            print(f"Contenu du fichier JSON problématique: {content}")
        return None

def get_confusion_matrices():
    """Retourne les matrices de confusion fournies par l'utilisateur."""
    # Matrices de confusion en pourcentage
    # Tableau 1 : Matrice pondérée (Précision: 0.87, Kappa: 0.83)
    weighted_matrix = np.array([
        [86.7, 6.7, 0.0, 6.7, 0.0, 0.0],    # Eau
        [1.1, 88.6, 0.0, 5.7, 1.1, 3.4],    # Forêt
        [0.0, 6.7, 66.7, 6.7, 0.0, 20.0],   # Tourbière
        [0.0, 0.0, 0.0, 86.3, 0.0, 13.7],   # Herbes
        [3.0, 0.0, 0.0, 24.2, 72.7, 0.0],   # Champs
        [0.0, 0.0, 0.0, 0.0, 0.0, 100.0]    # Urbain
    ])
    
    # Tableau 2 : Matrice non pondérée (Précision: 0.90, Kappa: 0.86)
    non_weighted_matrix = np.array([
        [86.7, 6.7, 0.0, 0.0, 0.0, 0.0],    # Eau
        [1.1, 94.3, 0.0, 3.4, 0.0, 1.1],    # Forêt
        [0.0, 6.7, 93.3, 0.0, 0.0, 0.0],    # Tourbière
        [0.0, 2.0, 0.0, 82.4, 2.0, 13.7],   # Herbes
        [3.0, 0.0, 0.0, 24.2, 72.7, 0.0],   # Champs
        [0.0, 0.0, 0.0, 0.0, 0.0, 100.0]    # Urbain
    ])
    
    return non_weighted_matrix, weighted_matrix

def compare_confusion_matrices():
    """Compare les matrices de confusion des deux approches."""
    # Chemins des répertoires de résultats
    base_dir = r"D:\UQTR\Hiver 2025\Télédétection\TP3\resultats"
    
    # Définir les valeurs manuellement (basées sur les logs d'exécution)
    non_weighted_accuracy = 0.90
    non_weighted_kappa = 0.86
    weighted_accuracy = 0.87
    weighted_kappa = 0.83
    
    print("\nMétriques de performance:")
    print(f"  Non pondérée - Précision: {non_weighted_accuracy:.2f}, Kappa: {non_weighted_kappa:.2f}")
    print(f"  Pondérée - Précision: {weighted_accuracy:.2f}, Kappa: {weighted_kappa:.2f}")
    
    # Récupérer les matrices de confusion réelles fournies
    non_weighted_matrix, weighted_matrix = get_confusion_matrices()
    
    # Charger les matrices de confusion
    try:
        # Créer un répertoire pour la comparaison
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        comparison_dir = os.path.join(base_dir, f"comparison_{timestamp}")
        os.makedirs(comparison_dir, exist_ok=True)
        
        print(f"\nRépertoire de résultats: {comparison_dir}")
        
        # Calculer les différences entre les deux approches
        diff_accuracy = non_weighted_accuracy - weighted_accuracy
        diff_kappa = non_weighted_kappa - weighted_kappa
        
        print("\nRésultats comparatifs:")
        print(f"  Différence de précision: {diff_accuracy:.2f} ({diff_accuracy*100:.1f}%)")
        print(f"  Différence de Kappa: {diff_kappa:.2f} ({diff_kappa*100:.1f}%)")
        
        # Créer des matrices de confusion visuelles avec matplotlib
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
        
        class_names = ["Eau", "Forêt", "Tourbière", "Herbes", "Champs", "Urbain"]
        
        # Afficher la matrice non pondérée
        im1 = ax1.imshow(non_weighted_matrix, interpolation='nearest', cmap='Blues')
        ax1.set_title(f'Matrice non pondérée\nPrécision: {non_weighted_accuracy:.2f}, Kappa: {non_weighted_kappa:.2f}')
        ax1.set_xticks(np.arange(len(class_names)))
        ax1.set_yticks(np.arange(len(class_names)))
        ax1.set_xticklabels(class_names, rotation=45, ha="right")
        ax1.set_yticklabels(class_names)
        
        # Ajouter les valeurs dans les cellules
        for i in range(len(class_names)):
            for j in range(len(class_names)):
                text = ax1.text(j, i, f"{non_weighted_matrix[i, j]:.1f}%",
                               ha="center", va="center", color="black" if non_weighted_matrix[i, j] < 60 else "white")
        
        # Afficher la matrice pondérée
        im2 = ax2.imshow(weighted_matrix, interpolation='nearest', cmap='Blues')
        ax2.set_title(f'Matrice pondérée\nPrécision: {weighted_accuracy:.2f}, Kappa: {weighted_kappa:.2f}')
        ax2.set_xticks(np.arange(len(class_names)))
        ax2.set_yticks(np.arange(len(class_names)))
        ax2.set_xticklabels(class_names, rotation=45, ha="right")
        ax2.set_yticklabels(class_names)
        
        # Ajouter les valeurs dans les cellules
        for i in range(len(class_names)):
            for j in range(len(class_names)):
                text = ax2.text(j, i, f"{weighted_matrix[i, j]:.1f}%",
                               ha="center", va="center", color="black" if weighted_matrix[i, j] < 60 else "white")
        
        plt.tight_layout()
        plt.suptitle("Comparaison des matrices de confusion", fontsize=16, y=1.05)
        
        # Sauvegarder la comparaison des matrices
        comparison_matrices_path = os.path.join(comparison_dir, "comparaison_matrices_visualisation.png")
        plt.savefig(comparison_matrices_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"\nComparaison des matrices sauvegardée dans: {comparison_matrices_path}")
        
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
            ax.text(i - width/2, v + 0.01, f'{v:.2f}', ha='center')
        
        for i, v in enumerate(weighted_values):
            ax.text(i + width/2, v + 0.01, f'{v:.2f}', ha='center')
        
        # Sauvegarder le graphique comparatif
        metrics_path = os.path.join(comparison_dir, "comparaison_metriques.png")
        plt.savefig(metrics_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Graphique des métriques sauvegardé dans: {metrics_path}")
        
        # ANALYSE DÉTAILLÉE PAR CLASSE
        print("\n=== ANALYSE DÉTAILLÉE PAR CLASSE ===\n")
        
        # Noms des classes
        class_names = ["Eau", "Forêt", "Tourbière", "Herbes", "Champs", "Urbain"]
        
        # Calculer les précisions pour chaque classe (diagonal = % bien classés)
        non_weighted_class_precision = np.diag(non_weighted_matrix) / 100
        weighted_class_precision = np.diag(weighted_matrix) / 100
        
        # Calculer la différence de précision par classe
        precision_diff = non_weighted_class_precision - weighted_class_precision
        
        # Afficher les résultats pour chaque classe
        print("Précision par classe:")
        for i, class_name in enumerate(class_names):
            print(f"  {class_name}:")
            print(f"    Non pondérée: {non_weighted_class_precision[i]:.2f} ({non_weighted_class_precision[i]*100:.1f}%)")
            print(f"    Pondérée: {weighted_class_precision[i]:.2f} ({weighted_class_precision[i]*100:.1f}%)")
            print(f"    Différence: {precision_diff[i]:.2f} ({precision_diff[i]*100:.1f}%)")
            
            if precision_diff[i] > 0:
                trend = "La classification non pondérée est meilleure pour cette classe."
            elif precision_diff[i] < 0:
                trend = "La classification pondérée est meilleure pour cette classe."
            else:
                trend = "Les deux classifications sont équivalentes pour cette classe."
            print(f"    Tendance: {trend}")
            print()
        
        # Créer un graphique pour comparer les précisions par classe
        fig, ax = plt.subplots(figsize=(12, 8))
        
        x = np.arange(len(class_names))
        width = 0.35
        
        ax.bar(x - width/2, non_weighted_class_precision * 100, width, label='Non pondérée')
        ax.bar(x + width/2, weighted_class_precision * 100, width, label='Pondérée')
        
        ax.set_ylabel('Précision (%)')
        ax.set_title('Précision par classe (en %)')
        ax.set_xticks(x)
        ax.set_xticklabels(class_names, rotation=45, ha='right')
        ax.legend()
        
        # Ajouter les valeurs sur les barres
        for i, v in enumerate(non_weighted_class_precision):
            ax.text(i - width/2, v * 100 + 0.5, f'{v*100:.1f}%', ha='center')
        
        for i, v in enumerate(weighted_class_precision):
            ax.text(i + width/2, v * 100 + 0.5, f'{v*100:.1f}%', ha='center')
        
        plt.tight_layout()
        
        # Sauvegarder le graphique
        class_precision_path = os.path.join(comparison_dir, "precision_par_classe.png")
        plt.savefig(class_precision_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Graphique de précision par classe sauvegardé dans: {class_precision_path}")
        
        # Créer un graphique pour montrer la différence de précision par classe
        fig, ax = plt.subplots(figsize=(12, 6))
        
        colors = ['green' if x > 0 else 'red' for x in precision_diff]
        
        ax.bar(x, precision_diff * 100, color=colors)
        
        ax.set_ylabel('Différence de précision (points de %)')
        ax.set_title('Différence de précision par classe (Non pondérée - Pondérée)')
        ax.set_xticks(x)
        ax.set_xticklabels(class_names, rotation=45, ha='right')
        
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        
        # Ajouter les valeurs sur les barres
        for i, v in enumerate(precision_diff):
            va = 'bottom' if v > 0 else 'top'
            ax.text(i, v * 100 + (1 if v > 0 else -1), f'{v*100:+.1f}%', ha='center', va=va)
        
        plt.tight_layout()
        
        # Sauvegarder le graphique
        diff_precision_path = os.path.join(comparison_dir, "difference_precision_par_classe.png")
        plt.savefig(diff_precision_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Graphique de différence de précision par classe sauvegardé dans: {diff_precision_path}")
        
        # Analyser les améliorations/dégradations spécifiques pour les classes cibles
        target_classes = ["Tourbière", "Champs"]
        target_indices = [class_names.index(cls) for cls in target_classes if cls in class_names]
        
        if target_indices:
            # Calculer la précision moyenne pour les classes cibles
            non_weighted_target_precision = np.mean([non_weighted_class_precision[i] for i in target_indices])
            weighted_target_precision = np.mean([weighted_class_precision[i] for i in target_indices])
            target_diff = weighted_target_precision - non_weighted_target_precision
            
            # Calculer la précision moyenne pour les autres classes
            other_indices = [i for i in range(len(class_names)) if i not in target_indices]
            non_weighted_other_precision = np.mean([non_weighted_class_precision[i] for i in other_indices])
            weighted_other_precision = np.mean([weighted_class_precision[i] for i in other_indices])
            other_diff = weighted_other_precision - non_weighted_other_precision
            
            print("\nAnalyse des classes cibles (pondérée - non pondérée):")
            print(f"  Classes cibles ({', '.join(target_classes)}):")
            print(f"    Précision moyenne non pondérée: {non_weighted_target_precision:.2f} ({non_weighted_target_precision*100:.1f}%)")
            print(f"    Précision moyenne pondérée: {weighted_target_precision:.2f} ({weighted_target_precision*100:.1f}%)")
            print(f"    Différence: {target_diff:+.2f} ({target_diff*100:+.1f}%)")
            
            print(f"  Autres classes:")
            print(f"    Précision moyenne non pondérée: {non_weighted_other_precision:.2f} ({non_weighted_other_precision*100:.1f}%)")
            print(f"    Précision moyenne pondérée: {weighted_other_precision:.2f} ({weighted_other_precision*100:.1f}%)")
            print(f"    Différence: {other_diff:+.2f} ({other_diff*100:+.1f}%)")
            
            # Créer un graphique pour comparer l'impact sur les classes cibles vs autres
            fig, ax = plt.subplots(figsize=(10, 6))
            
            cat_names = ["Classes cibles", "Autres classes"]
            diff_values = [target_diff * 100, other_diff * 100]
            colors = ['green' if x > 0 else 'red' for x in diff_values]
            
            ax.bar(cat_names, diff_values, color=colors)
            
            ax.set_ylabel('Différence de précision (points de %)')
            ax.set_title('Impact de la pondération (Pondérée - Non pondérée)')
            
            ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
            
            # Ajouter les valeurs sur les barres
            for i, v in enumerate(diff_values):
                va = 'bottom' if v > 0 else 'top'
                ax.text(i, v + (0.5 if v > 0 else -0.5), f'{v:+.1f}%', ha='center', va=va)
            
            plt.tight_layout()
            
            # Sauvegarder le graphique
            impact_path = os.path.join(comparison_dir, "impact_ponderation_classes_cibles.png")
            plt.savefig(impact_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"Graphique d'impact sur les classes cibles sauvegardé dans: {impact_path}")
            
            # Conclusion sur l'impact de la pondération
            if target_diff > 0 and other_diff < 0:
                print("\nConclusion: La pondération a amélioré la précision des classes cibles au détriment des autres classes.")
            elif target_diff < 0 and other_diff > 0:
                print("\nConclusion: La pondération a dégradé la précision des classes cibles au profit des autres classes.")
            elif target_diff > 0 and other_diff > 0:
                if target_diff > other_diff:
                    print("\nConclusion: La pondération a amélioré la précision de toutes les classes, avec un impact plus important sur les classes cibles.")
                else:
                    print("\nConclusion: La pondération a amélioré la précision de toutes les classes, avec un impact plus important sur les autres classes.")
            elif target_diff < 0 and other_diff < 0:
                if target_diff > other_diff:
                    print("\nConclusion: La pondération a dégradé la précision de toutes les classes, avec un impact moins négatif sur les classes cibles.")
                else:
                    print("\nConclusion: La pondération a dégradé la précision de toutes les classes, avec un impact plus négatif sur les classes cibles.")
            else:
                print("\nConclusion: L'impact de la pondération est mitigé entre les classes cibles et les autres classes.")
        
        # Conclusion générale
        if diff_accuracy > 0 and diff_kappa > 0:
            print("\nConclusion générale: La classification non pondérée a donné de meilleurs résultats globaux.")
        elif diff_accuracy < 0 and diff_kappa < 0:
            print("\nConclusion générale: La classification pondérée a donné de meilleurs résultats globaux.")
        else:
            print("\nConclusion générale: Les résultats sont mitigés entre les deux approches.")
        
    except Exception as e:
        print(f"Erreur lors de la comparaison: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    compare_confusion_matrices() 