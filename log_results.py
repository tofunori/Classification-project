"""
Script pour enregistrer les résultats de classification avec différentes pondérations
"""
import os
import csv
import datetime
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def init_log_file(output_dir):
    """Initialise le fichier de log s'il n'existe pas"""
    log_file = os.path.join(output_dir, 'classification_stats_log.csv')
    if not os.path.exists(log_file):
        with open(log_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Date', 'Heure', 'Poids_B2', 'Poids_B3', 'Poids_B4', 'Poids_B5', 
                            'Poids_B6', 'Poids_B7', 'Poids_B8', 'Précision_Standard', 
                            'Kappa_Standard', 'Précision_Pondérée', 'Kappa_Pondéré', 
                            'Variance_PC1(%)', 'Variance_PC2(%)', 'Variance_PC3(%)'])
    return log_file

def log_results(output_dir, weights, accuracy_std, kappa_std, accuracy_weighted, kappa_weighted, variance_explained):
    """Enregistre les résultats dans le fichier de log"""
    log_file = init_log_file(output_dir)
    
    now = datetime.datetime.now()
    date_str = now.strftime("%Y-%m-%d")
    time_str = now.strftime("%H:%M:%S")
    
    try:
        with open(log_file, 'a', newline='') as f:
            writer = csv.writer(f)
            row = [
                date_str, 
                time_str
            ]
            # Ajouter les poids
            for w in weights:
                row.append(f"{w:.2f}")
            
            # Ajouter les métriques
            row.extend([
                f"{accuracy_std:.2f}",
                f"{kappa_std:.2f}",
                f"{accuracy_weighted:.2f}",
                f"{kappa_weighted:.2f}"
            ])
            
            # Ajouter les variances expliquées
            for v in variance_explained[:3]:
                row.append(f"{v:.2f}")
                
            writer.writerow(row)
        print(f"\nStatistiques enregistrées dans: {log_file}")
        return True
    except Exception as e:
        print(f"\nErreur lors de l'enregistrement des statistiques: {str(e)}")
        return False

def generate_comparison_chart(output_dir):
    """Génère un graphique comparatif des différentes pondérations"""
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
        plt.figure(figsize=(12, 8))
        
        # Tracer la précision et le kappa pour chaque expérience
        x = range(len(df))
        plt.plot(x, df['Précision_Standard'], 'b-', label='Précision Standard')
        plt.plot(x, df['Kappa_Standard'], 'b--', label='Kappa Standard')
        plt.plot(x, df['Précision_Pondérée'], 'r-', label='Précision Pondérée')
        plt.plot(x, df['Kappa_Pondéré'], 'r--', label='Kappa Pondéré')
        
        # Ajouter les étiquettes et la légende
        plt.xticks(x, df['Expérience'], rotation=45, ha='right')
        plt.ylabel('Score')
        plt.title('Comparaison des résultats de classification avec différentes pondérations')
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

if __name__ == "__main__":
    # Exemple d'utilisation
    output_dir = "D:\\UQTR\\Hiver 2025\\Télédétection\\TP3\\resultats"
    
    # Si le fichier existe déjà, générer le graphique de comparaison
    if os.path.exists(os.path.join(output_dir, 'classification_stats_log.csv')):
        generate_comparison_chart(output_dir)
    else:
        print("Aucun fichier de log trouvé. Exécutez d'abord la classification avec différentes pondérations.")
