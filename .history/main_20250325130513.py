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
import json
from datetime import datetime

# Importer les modules du projet
from modules.data_loader import create_output_directory, load_and_check_data, save_raster
from modules.train import extract_training_samples
from modules.model import perform_classification, perform_classification_weighted
from modules.evaluate import compare_classifications
from modules.visualize import generate_classification_map, create_scatterplots, create_pca_scatterplots, visualize_spectral_signatures

def log_classification_results(output_dir, weights, accuracy_std, kappa_std, accuracy_weighted, kappa_weighted, variance_explained=None):
    """
    Enregistre automatiquement les résultats de classification dans un fichier JSON
    
    Args:
        output_dir (str): Répertoire de sortie
        weights (list): Liste des poids utilisés pour les bandes
        accuracy_std (float): Précision de la classification standard
        kappa_std (float): Coefficient Kappa de la classification standard
        accuracy_weighted (float): Précision de la classification pondérée
        kappa_weighted (float): Coefficient Kappa de la classification pondérée
        variance_explained (list, optional): Variance expliquée par les composantes principales
    """
    import json
    import numpy as np
    
    # Fonction pour convertir les types NumPy en types Python standard
    def convert_numpy_types(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, list):
            return [convert_numpy_types(item) for item in obj]
        elif isinstance(obj, dict):
            return {key: convert_numpy_types(value) for key, value in obj.items()}
        else:
            return obj
    
    # Initialiser le fichier de log
    log_file = os.path.join(output_dir, 'classification_stats_log.json')
    temp_log_file = os.path.join(output_dir, 'classification_stats_log_temp.json')
    
    # Obtenir la date et l'heure actuelles
    now = datetime.now()
    date_str = now.strftime("%Y-%m-%d")
    time_str = now.strftime("%H:%M:%S")
    
    try:
        # Préparer les données pour cette exécution
        band_names = ["B2 - Bleu", "B3 - Vert", "B4 - Rouge", "B5 - RedEdge05", 
                     "B6 - RedEdge06", "B7 - RedEdge07", "B8 - PIR"]
        
        # Convertir les poids en types Python standard
        weights = convert_numpy_types(weights)
        
        # Créer un dictionnaire pour les poids
        weights_dict = {}
        for i, name in enumerate(band_names[:len(weights)]):
            weights_dict[f"Poids_{name}"] = round(float(weights[i]), 2)
        
        # Créer l'entrée pour cette exécution
        entry = {
            "Date": date_str,
            "Heure": time_str,
            "Poids": weights_dict,
            "Precision_Standard": round(float(accuracy_std), 2),
            "Kappa_Standard": round(float(kappa_std), 2),
            "Precision_Ponderee": round(float(accuracy_weighted), 2),
            "Kappa_Pondere": round(float(kappa_weighted), 2)
        }
        
        # Ajouter les variances expliquées si disponibles
        if variance_explained is not None:
            variance_explained = convert_numpy_types(variance_explained)
            entry["Variance_PC"] = [round(float(v), 2) for v in variance_explained[:3]]
        
        # Charger les données existantes si le fichier existe
        existing_data = []
        if os.path.exists(log_file):
            try:
                with open(log_file, 'r', encoding='utf-8') as f:
                    existing_data = json.load(f)
                print(f"Fichier JSON existant chargé avec succès")
            except Exception as e:
                print(f"Impossible de lire le fichier JSON existant: {str(e)}")
                print(f"Création d'un nouveau fichier JSON")
        
        # Ajouter la nouvelle entrée
        existing_data.append(entry)
        
        # Essayer d'écrire directement dans le fichier
        try:
            with open(log_file, 'w', encoding='utf-8') as f:
                json.dump(existing_data, f, indent=2, ensure_ascii=False)
            
            print(f"\nRésultats enregistrés dans: {log_file}")
            return True
        except PermissionError:
            # Si le fichier est verrouillé, créer un fichier temporaire
            print(f"\nLe fichier {log_file} est verrouillé (probablement ouvert dans un autre programme).")
            print(f"Création d'un fichier temporaire: {temp_log_file}")
            
            try:
                with open(temp_log_file, 'w', encoding='utf-8') as f:
                    json.dump(existing_data, f, indent=2, ensure_ascii=False)
                
                print(f"Résultats enregistrés dans le fichier temporaire: {temp_log_file}")
                print(f"IMPORTANT: Fermez le fichier original et exécutez le script suivant pour fusionner les fichiers:")
                merge_cmd = f"python -c \"import os, shutil; shutil.copy2('{temp_log_file}', '{log_file}'); os.remove('{temp_log_file}')\""
                print(merge_cmd)
                return True
            except Exception as e:
                print(f"Erreur lors de la création du fichier temporaire: {str(e)}")
                return False
    except Exception as e:
        print(f"\nErreur lors de l'enregistrement des résultats: {str(e)}")
        import traceback
        print(traceback.format_exc())
        return False

def generate_comparison_chart(output_dir):
    """
    Génère un graphique comparatif des différentes pondérations à partir du fichier JSON
    
    Args:
        output_dir (str): Répertoire contenant le fichier de log
    """
    import json
    
    log_file = os.path.join(output_dir, 'classification_stats_log.json')
    temp_log_file = os.path.join(output_dir, 'classification_stats_log_temp.json')
    
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
        # Charger les données JSON
        with open(file_to_use, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if not data:
            print("Aucune donnée trouvée dans le fichier JSON")
            return False
        
        print(f"Données JSON chargées avec succès: {len(data)} entrées")
        
        # Créer un identifiant unique pour chaque expérience
        experiences = [f"{entry['Date']} {entry['Heure']}" for entry in data]
        
        # Créer le graphique
        plt.figure(figsize=(15, 10))
        
        # Sous-graphique 1: Précision et Kappa
        plt.subplot(2, 1, 1)
        
        # Extraire les données de précision et kappa
        precision_std = [entry.get('Precision_Standard', 0) for entry in data]
        kappa_std = [entry.get('Kappa_Standard', 0) for entry in data]
        precision_pond = [entry.get('Precision_Ponderee', 0) for entry in data]
        kappa_pond = [entry.get('Kappa_Pondere', 0) for entry in data]
        
        # Tracer les courbes
        plt.plot(range(len(data)), precision_std, 'b-', marker='o', label='Précision Standard')
        plt.plot(range(len(data)), kappa_std, 'b--', marker='s', label='Kappa Standard')
        plt.plot(range(len(data)), precision_pond, 'r-', marker='o', label='Précision Pondérée')
        plt.plot(range(len(data)), kappa_pond, 'r--', marker='s', label='Kappa Pondéré')
        
        plt.xticks(range(len(data)), experiences, rotation=45, ha='right')
        plt.ylabel('Score')
        plt.title('Comparaison des résultats de classification avec différentes pondérations')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        
        # Sous-graphique 2: Poids des bandes
        plt.subplot(2, 1, 2)
        
        # Extraire tous les noms de bandes uniques
        all_bands = set()
        for entry in data:
            if 'Poids' in entry:
                all_bands.update(entry['Poids'].keys())
        
        # Trier les noms de bandes
        all_bands = sorted(list(all_bands))
        
        # Tracer les poids pour chaque bande
        for band in all_bands:
            band_weights = []
            for entry in data:
                if 'Poids' in entry and band in entry['Poids']:
                    band_weights.append(entry['Poids'][band])
                else:
                    band_weights.append(0)  # Valeur par défaut si manquante
            
            plt.plot(range(len(data)), band_weights, marker='o', label=band.replace('Poids_', ''))
        
        plt.xticks(range(len(data)), experiences, rotation=45, ha='right')
        plt.ylabel('Poids')
        plt.title('Poids appliqués aux bandes')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        
        plt.tight_layout()
        
        # Sauvegarder le graphique
        chart_file = os.path.join(output_dir, 'comparaison_ponderations.png')
        plt.savefig(chart_file)
        plt.close()
        
        print(f"Graphique de comparaison généré: {chart_file}")
        return True
    except Exception as e:
        print(f"Erreur lors de la génération du graphique: {str(e)}")
        import traceback
        print(traceback.format_exc())
        return False

def run_classification(input_path=None, output_dir=None, custom_config=None, weights=None):
    """
    Exécute le workflow complet de classification par maximum de vraisemblance.
    
    Args:
        input_path (str, optional): Chemin vers le fichier raster d'entrée. Si None, utilise le chemin par défaut.
        output_dir (str, optional): Répertoire de sortie. Si None, utilise le répertoire par défaut.
        custom_config (dict, optional): Configuration personnalisée à utiliser.
        weights (numpy.ndarray, optional): Poids personnalisés pour les bandes.
        
    Returns:
        dict or bool: Dictionnaire contenant les résultats de la classification, ou False en cas d'erreur.
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
            },
            "skip_visualizations": False  # Option pour sauter les visualisations (utile pour le traitement par lot)
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
        log_file = os.path.join(config['output_dir'], 'classification_stats_log.json')
        if not os.path.exists(log_file):
            with open(log_file, 'w', newline='') as f:
                json.dump([], f)
        
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
            2.0,    # B2 - Bleu [16U]
            1.0,    # B3 - Vert [16U]
            1.0,    # B4 - Rouge [16U]
            5.0,    # B5 - RedEdge05 [16U]
            1.0,    # B6 - RedEdge06 [16U]
            1.0,    # B7 - RedEdge07 [16U]
            4.0     # B8 - PIR [16U]
        ])
        
        # Utiliser les poids personnalisés s'ils sont fournis
        if weights is not None:
            optimized_weights = weights
        
        print("Poids définis pour les bandes:")
        for i, weight in enumerate(optimized_weights):
            print(f"  Bande {i+1}: {weight:.2f}")
        
        # Convertir en tableau NumPy si c'est une liste
        if isinstance(optimized_weights, list):
            optimized_weights = np.array(optimized_weights)
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
        if not config.get("skip_visualizations", False):
            print("\nGénération des visualisations avec pondération...")
            visualize_spectral_signatures(classes_info, config, band_weights=optimized_weights)
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
        
        # Variables pour stocker les résultats
        accuracy_std = 0
        kappa_std = 0
        accuracy_weighted = 0
        kappa_weighted = 0
        
        if config["comparison"]["enabled"]:
            try:
                # Évaluation de la classification standard
                print("\nÉvaluation de la classification standard...")
                comp_std = compare_classifications(classification_std, config)
                if comp_std:
                    accuracy_std = comp_std['accuracy']
                    kappa_std = comp_std['kappa']
                    print(f"Précision standard: {accuracy_std:.2f}")
                    print(f"Kappa standard: {kappa_std:.2f}")
                    resultats_tests.append({
                        "nom": "Standard (sans poids)",
                        "precision": accuracy_std,
                        "kappa": kappa_std
                    })
                
                # Évaluation de la classification pondérée
                print("\nÉvaluation de la classification avec poids optimisés...")
                comp_ponderee = compare_classifications(classification_ponderee, config)
                if comp_ponderee:
                    accuracy_weighted = comp_ponderee['accuracy']
                    kappa_weighted = comp_ponderee['kappa']
                    print(f"Précision avec pondération: {accuracy_weighted:.2f}")
                    print(f"Kappa avec pondération: {kappa_weighted:.2f}")
                    resultats_tests.append({
                        "nom": "Poids optimisés",
                        "precision": accuracy_weighted,
                        "kappa": kappa_weighted
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
                config["output_dir"],
                optimized_weights,
                accuracy_std,
                kappa_std,
                accuracy_weighted,
                kappa_weighted,
                variance_explained
            )
            print("\nRésultats enregistrés dans le fichier de log.")
        except Exception as e:
            print(f"\nERREUR lors de l'enregistrement des résultats: {str(e)}")
            print(traceback.format_exc())
        
        # Étape 6: Enregistrement des résultats
        print("\n--- Enregistrement des résultats ---")
        
        # Enregistrer la classification standard
        save_raster(
            classification_std,
            os.path.join(config["output_dir"], "classification_standard.tif"),
            meta
        )
        
        # Enregistrer la classification pondérée
        save_raster(
            classification_ponderee,
            os.path.join(config["output_dir"], "classification_ponderee.tif"),
            meta
        )
        
        # Générer les cartes de classification si les visualisations ne sont pas désactivées
        if not config.get("skip_visualizations", False):
            # Carte de classification standard
            generate_classification_map(classification_std, config)
            
            # Carte de classification pondérée
            config["output_suffix"] = "_ponderee"
            generate_classification_map(classification_ponderee, config)
        
        # Génération du graphique de comparaison
        try:
            generate_comparison_chart(config["output_dir"])
        except Exception as e:
            print(f"\nERREUR lors de la génération du graphique de comparaison: {str(e)}")
            print(traceback.format_exc())
        
        # Afficher le temps d'exécution
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"\nTemps d'exécution total: {execution_time:.2f} secondes")
        
        print("\n" + "=" * 70)
        print(" CLASSIFICATION TERMINÉE AVEC SUCCÈS ")
        print("=" * 70)
        
        # Retourner les résultats
        return {
            "accuracy_std": accuracy_std,
            "kappa_std": kappa_std,
            "accuracy_weighted": accuracy_weighted,
            "kappa_weighted": kappa_weighted,
            "variance_explained": variance_explained.tolist() if hasattr(variance_explained, 'tolist') else variance_explained
        }
        
    except Exception as e:
        print(f"\nERREUR FATALE: {str(e)}")
        print(traceback.format_exc())
        return False

def main():
    """Point d'entrée principal du programme."""
    run_classification()

if __name__ == "__main__":
    main()
