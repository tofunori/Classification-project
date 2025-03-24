"""
MAIN MODULE - CLASSIFICATION SUPERVISÉE
======================================
Point d'entrée principal pour le projet de classification d'images Sentinel-2.
Ce script orchestre le processus complet de classification par maximum de vraisemblance.
"""

import os
import sys
import time
import traceback
import numpy as np
import rasterio
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import csv
from datetime import datetime

# Importer les modules du projet
from modules.data_loader import create_output_directory, load_and_check_data, save_raster
from modules.train import extract_training_samples
from modules.model import perform_classification, perform_classification_weighted
from modules.evaluate import compare_classifications
from modules.visualize import generate_classification_map, create_scatterplots, create_pca_scatterplots

def log_classification_results(output_dir, weights, accuracy_std, kappa_std, accuracy_weighted, kappa_weighted, variance_explained=None):
    """
    Enregistre automatiquement les résultats de classification dans un fichier CSV
    
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
    temp_log_file = os.path.join(output_dir, 'classification_stats_log_temp.csv')
    header_needed = not os.path.exists(log_file)
    
    # Obtenir la date et l'heure actuelles
    now = datetime.now()
    date_str = now.strftime("%Y-%m-%d")
    time_str = now.strftime("%H:%M:%S")
    
    try:
        # Préparer les données
        row_data = [date_str, time_str]
        
        # Ajouter les poids
        for weight in weights:
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
        
        # Si le fichier existe déjà, essayer de le lire d'abord
        existing_data = []
        header = []
        
        if os.path.exists(log_file) and not header_needed:
            try:
                with open(log_file, 'r', newline='', encoding='utf-8') as f:
                    reader = csv.reader(f)
                    header = next(reader)  # Lire l'en-tête
                    existing_data = list(reader)  # Lire toutes les lignes existantes
            except Exception as e:
                print(f"Impossible de lire le fichier existant: {str(e)}")
                # Si on ne peut pas lire le fichier, on va le recréer
                header_needed = True
                existing_data = []
        
        # Créer l'en-tête si nécessaire
        if header_needed:
            header = ['Date', 'Heure']
            band_names = ["B2 - Bleu", "B3 - Vert", "B4 - Rouge", "B5 - RedEdge05", 
                         "B6 - RedEdge06", "B7 - RedEdge07", "B8 - PIR"]
            
            for name in band_names:
                header.append(f"Poids_{name}")
            
            header.extend(['Précision_Standard', 'Kappa_Standard', 
                          'Précision_Pondérée', 'Kappa_Pondéré'])
            
            if variance_explained is not None:
                header.extend(['Variance_PC1(%)', 'Variance_PC2(%)', 'Variance_PC3(%)'])
        
        # Essayer d'écrire directement dans le fichier
        try:
            with open(log_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(header)
                for row in existing_data:
                    writer.writerow(row)
                writer.writerow(row_data)
            
            print(f"\nRésultats enregistrés dans: {log_file}")
            return True
        except PermissionError:
            # Si le fichier est verrouillé (par Excel par exemple), créer un fichier temporaire
            print(f"\nLe fichier {log_file} est verrouillé (probablement ouvert dans Excel).")
            print(f"Création d'un fichier temporaire: {temp_log_file}")
            
            try:
                with open(temp_log_file, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    writer.writerow(header)
                    for row in existing_data:
                        writer.writerow(row)
                    writer.writerow(row_data)
                
                print(f"Résultats enregistrés dans le fichier temporaire: {temp_log_file}")
                print(f"IMPORTANT: Fermez Excel et exécutez le script suivant pour fusionner les fichiers:")
                print(f"python -c \"import os, shutil; shutil.copy2('{temp_log_file}', '{log_file}'); os.remove('{temp_log_file}')\"")
                return True
            except Exception as e:
                print(f"Erreur lors de la création du fichier temporaire: {str(e)}")
                return False
    except Exception as e:
        print(f"\nErreur lors de l'enregistrement des résultats: {str(e)}")
        return False

def generate_comparison_chart(output_dir):
    """
    Génère un graphique comparatif des différentes pondérations
    
    Args:
        output_dir (str): Répertoire contenant le fichier de log
    """
    log_file = os.path.join(output_dir, 'classification_stats_log.csv')
    temp_log_file = os.path.join(output_dir, 'classification_stats_log_temp.csv')
    
    # Vérifier d'abord le fichier principal
    file_to_use = log_file
    if not os.path.exists(log_file):
        # Si le fichier principal n'existe pas, vérifier le fichier temporaire
        if os.path.exists(temp_log_file):
            file_to_use = temp_log_file
            print(f"Fichier de log principal introuvable, utilisation du fichier temporaire: {temp_log_file}")
        else:
            print(f"Aucun fichier de log trouvé: ni {log_file} ni {temp_log_file}")
            return False
    
    try:
        # Charger les données
        import pandas as pd
        try:
            df = pd.read_csv(file_to_use, encoding='utf-8')
        except UnicodeDecodeError:
            # Essayer avec une autre encodage si utf-8 échoue
            df = pd.read_csv(file_to_use, encoding='latin1')
        
        # Créer un identifiant unique pour chaque expérience
        df['Expérience'] = df['Date'] + ' ' + df['Heure']
        
        # Créer le graphique
        plt.figure(figsize=(15, 10))
        
        # Sous-graphique 1: Précision et Kappa
        plt.subplot(2, 1, 1)
        plt.plot(range(len(df)), df['Précision_Standard'], 'b-', marker='o', label='Précision Standard')
        plt.plot(range(len(df)), df['Kappa_Standard'], 'b--', marker='s', label='Kappa Standard')
        plt.plot(range(len(df)), df['Précision_Pondérée'], 'r-', marker='o', label='Précision Pondérée')
        plt.plot(range(len(df)), df['Kappa_Pondéré'], 'r--', marker='s', label='Kappa Pondéré')
        
        plt.xticks(range(len(df)), df['Expérience'], rotation=45, ha='right')
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
            plt.plot(range(len(df)), df[col].astype(float), marker='o', label=col.replace('Poids_', ''))
        
        plt.xticks(range(len(df)), df['Expérience'], rotation=45, ha='right')
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
        import traceback
        print(traceback.format_exc())
        return False

def run_classification(input_path=None, output_dir=None, custom_config=None):
    """
    Exécute le workflow complet de classification par maximum de vraisemblance.
    
    Args:
        input_path (str, optional): Chemin vers le fichier raster d'entrée. Si None, utilise le chemin par défaut.
        output_dir (str, optional): Répertoire de sortie. Si None, utilise le répertoire par défaut.
        custom_config (dict, optional): Configuration personnalisée à utiliser.
        
    Returns:
        bool: True si le processus s'est terminé avec succès, False sinon.
    """
    try:
        start_time = time.time()
        print("=" * 70)
        print(" CLASSIFICATION PAR MAXIMUM DE VRAISEMBLANCE - DÉMARRAGE ")
        print("=" * 70)
        
        # Étape 1: Configuration et initialisation
        config = {
            "raster_path": r"D:\UQTR\Hiver 2025\Télédétection\TP3\tr_clip.tif",
            "output_dir": r"D:\UQTR\Hiver 2025\Télédétection\TP3\resultats",
            "shapefile_path": r"D:\UQTR\Hiver 2025\Télédétection\TP3\classes.shp",
            "class_column": "Classe",
            "selected_bands": [2, 3, 4, 5, 6, 7, 8],
            "class_params": {},
            "class_names": {
                1: "Eau", 
                2: "Forêt", 
                3: "Tourbière",
                4: "Herbes", 
                5: "Champs", 
                6: "Urbain"
            },
            "class_colors": {
                1: "#3288bd",  # Eau - bleu
                2: "#66c164",  # Forêt - vert
                3: "#87CEFA",  # Tourbière - bleu clair
                4: "#ffff00",  # Herbes - vert clair
                5: "#f39c12",  # Champs - orange
                6: "#7f8c8d"   # Urbain - gris
            },
            "comparison": {
                "enabled": True,
                "raster_path": r"D:\UQTR\Hiver 2025\Télédétection\TP3\resultats\classification_mlc.tif"
            }
        }
        
        if custom_config:
            config.update(custom_config)
        
        if input_path:
            config["raster_path"] = input_path
        if output_dir:
            config["output_dir"] = output_dir
            
        print(f"Utilisation du fichier d'entrée: {config['raster_path']}")
        print(f"Utilisation du répertoire de sortie: {config['output_dir']}")
        
        # Vérifier/créer le répertoire de sortie
        create_output_directory(config["output_dir"])
        
        # Initialiser le fichier de log des statistiques s'il n'existe pas
        log_file = os.path.join(config['output_dir'], 'classification_stats_log.csv')
        if not os.path.exists(log_file):
            with open(log_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['Date', 'Heure', 'Poids_B2', 'Poids_B3', 'Poids_B4', 'Poids_B5', 
                                'Poids_B6', 'Poids_B7', 'Poids_B8', 'Précision_Standard', 
                                'Kappa_Standard', 'Précision_Pondérée', 'Kappa_Pondéré', 
                                'Variance_PC1(%)', 'Variance_PC2(%)', 'Variance_PC3(%)'])
        
        # Étape 2: Chargement et vérification des données
        raster_data, meta, shapefile = load_and_check_data(config)
        bands, height, width = raster_data.shape
        
        # Étape 3: Extraction des échantillons d'entraînement
        print("\nExtraction des échantillons d'entraînement...")
        try:
            classes_info = extract_training_samples(raster_data, shapefile, config)
        except Exception as e:
            print(f"\nERREUR lors de l'extraction des échantillons: {str(e)}")
            print(traceback.format_exc())
            return False
        
        # Définir les poids optimisés pour les bandes
        optimized_weights = np.array([
            1.0,    # B2 - Bleu [16U]
            1.0,    # B3 - Vert [16U]
            2.0,    # B4 - Rouge [16U]
            2.0,    # B5 - RedEdge05 [16U]
            2.0,    # B6 - RedEdge06 [16U]
            2.0,    # B7 - RedEdge07 [16U]
            3.0     # B8 - PIR [16U]
        ])
        
        print("Poids définis pour les bandes:")
        for i, weight in enumerate(optimized_weights):
            print(f"  Bande {i+1}: {weight:.2f}")
        
        # Pour vérification des dimensions
        print(f"Dimensions du raster: {raster_data.shape}")
        print(f"Dimensions du tableau de poids: {optimized_weights.shape}")
        
        # Vérification d'une classe d'entraînement pour ses dimensions
        first_class = next(iter(classes_info.values()))
        print(f"Dimensions d'une classe d'entraînement: {first_class['training_data'].shape}")
        
        # Étape 4: Réaliser les classifications
        print("\n--- Classification ---")
        
        # Classification standard sans pondération
        print("\nClassification standard pour référence...")
        classification_std, _ = perform_classification(raster_data, classes_info, config)
        
        # Classification avec pondération des bandes
        print("\nClassification avec poids optimisés...")
        classification_ponderee, _ = perform_classification_weighted(
            raster_data, classes_info, config, optimized_weights
        )
        
        # Génération des visualisations avec les poids optimisés
        print("\nGénération des visualisations avec pondération...")
        create_scatterplots(classes_info, config)  # Les scatterplots standards ne peuvent pas être pondérés
        create_pca_scatterplots(classes_info, config, band_weights=optimized_weights)  # Passage des poids à PCA
        
        # Génération des scatterplots combinés avec PCA...
        print("\nGénération des scatterplots combinés avec PCA...")
        
        # Application des pondérations aux données avant PCA
        print("  Application des pondérations aux données avant PCA...")
        
        # Afficher les poids utilisés
        print("  Poids appliqués pour la PCA:")
        for i, weight in enumerate(optimized_weights):
            print(f"    Bande {i+1}: {weight:.2f}")
        
        # Créer une copie des données d'entraînement
        weighted_training_data = {}
        for class_id, class_data in classes_info.items():
            weighted_training_data[class_id] = class_data['training_data'].copy()
            # Appliquer les poids aux données
            for i, weight in enumerate(optimized_weights):
                weighted_training_data[class_id][:, i] *= weight
        
        # Combiner toutes les données pondérées pour la PCA
        all_weighted_data = np.vstack([weighted_training_data[class_id] for class_id in weighted_training_data])
        
        # Appliquer la PCA
        from sklearn.decomposition import PCA
        pca = PCA(n_components=3)
        pca_result = pca.fit_transform(all_weighted_data)
        
        # Variance expliquée
        variance_explained = pca.explained_variance_ratio_ * 100
        print(f"  Variance expliquée par les composantes principales: {variance_explained}")
        
        # Séparer les résultats par classe
        pca_by_class = {}
        start_idx = 0
        for class_id, class_data in classes_info.items():
            end_idx = start_idx + class_data['training_data'].shape[0]
            pca_by_class[class_id] = pca_result[start_idx:end_idx]
            start_idx = end_idx
        
        # Étape 5: Évaluation et comparaison
        print("\n--- Évaluation des résultats ---")
        resultats_tests = []
        
        if config["comparison"]["enabled"]:
            try:
                # Évaluation de la classification standard
                print("\nÉvaluation de la classification standard...")
                comp_std = compare_classifications(classification_std, config)
                if comp_std:
                    print(f"Précision standard: {comp_std['accuracy']:.2f}")
                    print(f"Kappa standard: {comp_std['kappa']:.2f}")
                    resultats_tests.append({
                        "nom": "Standard (sans poids)",
                        "precision": comp_std["accuracy"],
                        "kappa": comp_std["kappa"]
                    })
                
                # Évaluation de la classification pondérée
                print("\nÉvaluation de la classification avec poids optimisés...")
                comp_ponderee = compare_classifications(classification_ponderee, config)
                if comp_ponderee:
                    print(f"Précision avec pondération: {comp_ponderee['accuracy']:.2f}")
                    print(f"Kappa avec pondération: {comp_ponderee['kappa']:.2f}")
                    resultats_tests.append({
                        "nom": "Poids optimisés",
                        "precision": comp_ponderee["accuracy"],
                        "kappa": comp_ponderee["kappa"]
                    })
                
                # Afficher la comparaison des résultats
                if len(resultats_tests) > 0:
                    print("\n--- COMPARAISON DES RÉSULTATS ---")
                    print("Test\t\tPrécision\tKappa")
                    print("-" * 40)
                    for res in resultats_tests:
                        print(f"{res['nom']}\t{res['precision']:.2f}\t\t{res['kappa']:.2f}")
                    print("-" * 40)
            except Exception as e:
                print(f"\nERREUR lors de la comparaison: {str(e)}")
                print(traceback.format_exc())
        else:
            print("\nComparaison désactivée dans la configuration.")
        
        # Enregistrer les résultats dans le fichier de log
        try:
            log_classification_results(
                config['output_dir'], 
                optimized_weights, 
                comp_std['accuracy'], 
                comp_std['kappa'], 
                comp_ponderee['accuracy'], 
                comp_ponderee['kappa'], 
                variance_explained
            )
            generate_comparison_chart(config['output_dir'])
        except Exception as e:
            print(f"\nErreur lors de l'enregistrement des statistiques: {str(e)}")
        
        print("\n--- Sauvegarde des résultats ---")
        
        # Préparer les métadonnées pour la sauvegarde
        save_meta = meta.copy()
        save_meta.update({
            'count': 1,
            'dtype': rasterio.uint8
        })
        
        # Sauvegarde des résultats de la classification pondérée
        weighted_output_path = os.path.join(config["output_dir"], "classification_ponderee.tif")
        save_raster(classification_ponderee.astype(rasterio.uint8), save_meta, weighted_output_path)
        print(f"Classification pondérée sauvegardée: {weighted_output_path}")
        
        # Sauvegarde des résultats de la classification standard
        standard_output_path = os.path.join(config["output_dir"], "classification_standard.tif")
        save_raster(classification_std.astype(rasterio.uint8), save_meta, standard_output_path)
        print(f"Classification standard sauvegardée: {standard_output_path}")
        
        # Calculer et sauvegarder la différence entre les classifications
        print("\nCalcul de la différence entre classifications...")
        difference = (classification_ponderee != classification_std).astype(np.uint8)
        difference_output_path = os.path.join(config["output_dir"], "difference_classification.tif")
        save_raster(difference, save_meta, difference_output_path)
        print(f"Différence entre classifications sauvegardée: {difference_output_path}")
        
        # Générer une visualisation de la différence
        print("Génération de la visualisation de différence...")
        diff_image_path = os.path.join(config["output_dir"], "difference_classification.png")
        
        # Créer une figure pour visualiser la différence
        plt.figure(figsize=(12, 8), dpi=300)
        plt.imshow(difference, cmap='hot')
        plt.colorbar(label='Différence (0=identique, 1=différent)')
        plt.title('Différence entre Classification Standard et Pondérée', fontsize=16)
        plt.savefig(diff_image_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Visualisation de différence sauvegardée: {diff_image_path}")
        
        # Génération de la carte de classification avec overwrite
        print("\nGénération de la carte de classification...")
        # Générer la carte pour la classification pondérée
        carte_path = r"D:\UQTR\Hiver 2025\Télédétection\TP3\resultats\carte_classification.png"
        # Supprimer le fichier s'il existe déjà pour s'assurer qu'il est écrasé
        if os.path.exists(carte_path):
            try:
                os.remove(carte_path)
                print(f"Ancien fichier carte supprimé: {carte_path}")
            except Exception as e:
                print(f"Erreur lors de la suppression de l'ancienne carte: {e}")
        
        generate_classification_map(classification_ponderee, config)
        print(f"Carte de classification générée: {carte_path}")
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        print("\n" + "=" * 70)
        print(f" CLASSIFICATION TERMINÉE EN {execution_time:.2f} SECONDES ")
        print("=" * 70)
        
        return True
        
    except Exception as e:
        print(f"\nERREUR: {str(e)}")
        print(traceback.format_exc())
        print("La classification a échoué.")
        return False

def main():
    """Point d'entrée principal du programme."""
    run_classification()

if __name__ == "__main__":
    main()
