"""
Script pour enregistrer et visualiser les résultats de classification avec différentes pondérations
"""
import os
import sys
import csv
import datetime
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import argparse

def log_classification_results(output_dir, weights, accuracy_std, kappa_std, accuracy_weighted, kappa_weighted, variance_explained=None):
    """
    Enregistre les résultats de classification dans un fichier CSV
    
    Args:
        output_dir (str): Répertoire de sortie
        weights (list): Liste des poids utilisés pour les bandes
        accuracy_std (float): Précision de la classification standard
        kappa_std (float): Coefficient Kappa de la classification standard
        accuracy_weighted (float): Précision de la classification pondérée
        kappa_weighted (float): Coefficient Kappa de la classification pondérée
        variance_explained (list, optional): Variance expliquée par les composantes principales
    """
    # Initialiser le fichier de log s'il n'existe pas
    log_file = os.path.join(output_dir, 'classification_stats_log.csv')
    header_needed = not os.path.exists(log_file)
    
    # Obtenir la date et l'heure actuelles
    now = datetime.datetime.now()
    date_str = now.strftime("%Y-%m-%d")
    time_str = now.strftime("%H:%M:%S")
    
    # Préparer les données
    row_data = [date_str, time_str]
    
    # Ajouter les poids
    band_names = ["B2 - Bleu", "B3 - Vert", "B4 - Rouge", "B5 - RedEdge05", 
                 "B6 - RedEdge06", "B7 - RedEdge07", "B8 - PIR"]
    
    for i, weight in enumerate(weights):
        row_data.append(f"{weight:.2f}")
    
    # Ajouter les métriques
    row_data.extend([
        f"{accuracy_std:.2f}",
        f"{kappa_std:.2f}",
        f"{accuracy_weighted:.2f}",
        f"{kappa_weighted:.2f}"
    ])
    
    # Ajouter les variances expliquées si disponibles
    if variance_explained is not None:
        for i in range(min(3, len(variance_explained))):
            row_data.append(f"{variance_explained[i]:.2f}")
    
    # Écrire dans le fichier
    try:
        with open(log_file, 'a', newline='') as f:
            writer = csv.writer(f)
            
            # Écrire l'en-tête si nécessaire
            if header_needed:
                header = ['Date', 'Heure']
                for i, name in enumerate(band_names):
                    header.append(f"Poids_{name}")
                header.extend(['Précision_Standard', 'Kappa_Standard', 
                              'Précision_Pondérée', 'Kappa_Pondéré'])
                if variance_explained is not None:
                    header.extend(['Variance_PC1(%)', 'Variance_PC2(%)', 'Variance_PC3(%)'])
                writer.writerow(header)
            
            # Écrire les données
            writer.writerow(row_data)
        
        print(f"Résultats enregistrés dans: {log_file}")
        return True
    except Exception as e:
        print(f"Erreur lors de l'enregistrement des résultats: {str(e)}")
        return False

def generate_comparison_chart(output_dir):
    """
    Génère un graphique comparatif des différentes pondérations
    
    Args:
        output_dir (str): Répertoire contenant le fichier de log
    """
    log_file = os.path.join(output_dir, 'classification_stats_log.csv')
    if not os.path.exists(log_file):
        print(f"Fichier de log introuvable: {log_file}")
        return False
    
    try:
        # Charger les données
        df = pd.read_csv(log_file)
        
        # Créer un identifiant unique pour chaque expérience
        df['Expérience'] = df['Date'] + ' ' + df['Heure']
        
        # Créer le graphique
        plt.figure(figsize=(15, 10))
        
        # Tracer la précision et le kappa pour chaque expérience
        x = range(len(df))
        
        # Sous-graphique 1: Précision et Kappa
        plt.subplot(2, 1, 1)
        plt.plot(x, df['Précision_Standard'], 'b-', marker='o', label='Précision Standard')
        plt.plot(x, df['Kappa_Standard'], 'b--', marker='s', label='Kappa Standard')
        plt.plot(x, df['Précision_Pondérée'], 'r-', marker='o', label='Précision Pondérée')
        plt.plot(x, df['Kappa_Pondéré'], 'r--', marker='s', label='Kappa Pondéré')
        
        plt.xticks(x, df['Expérience'], rotation=45, ha='right')
        plt.ylabel('Score')
        plt.title('Comparaison des résultats de classification avec différentes pondérations')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        
        # Sous-graphique 2: Poids des bandes
        plt.subplot(2, 1, 2)
        
        # Identifier les colonnes de poids
        weight_cols = [col for col in df.columns if col.startswith('Poids_')]
        
        # Tracer les poids pour chaque expérience
        for col in weight_cols:
            plt.plot(x, df[col].astype(float), marker='o', label=col.replace('Poids_', ''))
        
        plt.xticks(x, df['Expérience'], rotation=45, ha='right')
        plt.ylabel('Poids')
        plt.title('Poids appliqués aux bandes')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        
        plt.tight_layout()
        
        # Sauvegarder le graphique
        chart_file = os.path.join(output_dir, 'comparaison_pondérations.png')
        plt.savefig(chart_file)
        plt.close()
        
        print(f"Graphique de comparaison généré: {chart_file}")
        return True
    except Exception as e:
        print(f"Erreur lors de la génération du graphique: {str(e)}")
        return False

def manual_entry():
    """Interface pour saisir manuellement les résultats d'une classification"""
    parser = argparse.ArgumentParser(description='Enregistrer les résultats de classification')
    parser.add_argument('--output_dir', type=str, default="D:\\UQTR\\Hiver 2025\\Télédétection\\TP3\\resultats",
                        help='Répertoire de sortie')
    parser.add_argument('--generate_chart', action='store_true',
                        help='Générer un graphique de comparaison')
    
    args = parser.parse_args()
    
    if args.generate_chart:
        generate_comparison_chart(args.output_dir)
        return
    
    # Saisie des poids
    print("Saisie des poids pour les bandes:")
    weights = []
    band_names = ["B2 - Bleu", "B3 - Vert", "B4 - Rouge", "B5 - RedEdge05", 
                 "B6 - RedEdge06", "B7 - RedEdge07", "B8 - PIR"]
    
    for i, name in enumerate(band_names):
        weight = float(input(f"Poids pour {name}: "))
        weights.append(weight)
    
    # Saisie des métriques
    accuracy_std = float(input("Précision de la classification standard: "))
    kappa_std = float(input("Coefficient Kappa de la classification standard: "))
    accuracy_weighted = float(input("Précision de la classification pondérée: "))
    kappa_weighted = float(input("Coefficient Kappa de la classification pondérée: "))
    
    # Saisie des variances expliquées (optionnel)
    use_variance = input("Avez-vous des données de variance expliquée? (o/n): ").lower() == 'o'
    variance_explained = None
    
    if use_variance:
        variance_explained = []
        for i in range(3):
            variance = float(input(f"Variance expliquée par PC{i+1} (%): "))
            variance_explained.append(variance)
    
    # Enregistrer les résultats
    success = log_classification_results(
        "D:\\UQTR\\Hiver 2025\\Télédétection\\TP3\\resultats", 
        weights, 
        accuracy_std, 
        kappa_std, 
        accuracy_weighted, 
        kappa_weighted, 
        variance_explained
    )
    
    if success:
        generate_chart = input("Voulez-vous générer un graphique de comparaison? (o/n): ").lower() == 'o'
        if generate_chart:
            generate_comparison_chart("D:\\UQTR\\Hiver 2025\\Télédétection\\TP3\\resultats")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Enregistrer les résultats de classification')
    parser.add_argument('--output_dir', type=str, default="D:\\UQTR\\Hiver 2025\\Télédétection\\TP3\\resultats",
                        help='Répertoire de sortie')
    parser.add_argument('--generate_chart', action='store_true',
                        help='Générer un graphique de comparaison')
    parser.add_argument('--weights', type=float, nargs='+', 
                        help='Poids des bandes (B2 B3 B4 B5 B6 B7 B8)')
    parser.add_argument('--accuracy_std', type=float, help='Précision standard')
    parser.add_argument('--kappa_std', type=float, help='Kappa standard')
    parser.add_argument('--accuracy_weighted', type=float, help='Précision pondérée')
    parser.add_argument('--kappa_weighted', type=float, help='Kappa pondéré')
    parser.add_argument('--variance', type=float, nargs='+', help='Variance expliquée (PC1 PC2 PC3)')
    
    args = parser.parse_args()
    
    # Si seulement --generate_chart est spécifié, générer le graphique et sortir
    if args.generate_chart and not args.weights:
        generate_comparison_chart(args.output_dir)
        sys.exit(0)
    
    # Si les poids et métriques sont fournis, enregistrer directement
    if args.weights and args.accuracy_std and args.kappa_std and args.accuracy_weighted and args.kappa_weighted:
        log_classification_results(
            args.output_dir,
            args.weights,
            args.accuracy_std,
            args.kappa_std,
            args.accuracy_weighted,
            args.kappa_weighted,
            args.variance
        )
        
        # Générer automatiquement le graphique si demandé
        if args.generate_chart:
            generate_comparison_chart(args.output_dir)
    else:
        # Sinon, utiliser l'interface interactive
        manual_entry()
