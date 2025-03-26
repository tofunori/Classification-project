"""
CLASSIFICATION COMPLÈTE CORRIGÉE
=======================================
Ce script effectue une classification par maximum de vraisemblance sans pondération des bandes,
avec génération de fichier TIF et tous les graphiques (PCA, scatterplots, matrices de confusion)
en utilisant la bonne colonne de classe pour la validation.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import time
import json
import traceback
from sklearn.decomposition import PCA

# Importer les modules du projet
from modules.data_loader import load_and_check_data, create_output_directory, save_raster
from modules.model import perform_classification
from modules.evaluate import validate_classification
from modules.visualize import (
    visualize_spectral_signatures, 
    generate_classification_map, 
    create_scatterplots
)

def create_pca_scatterplots(classes_info, config):
    """Crée des scatterplots basés sur PCA."""
    print("Génération des scatterplots combinés avec PCA...")
    
    # Combiner toutes les données d'entraînement pour la PCA
    all_data = np.vstack([classes_info[class_id]['training_data'] for class_id in classes_info])
    
    # Appliquer la PCA
    pca = PCA(n_components=3)
    pca_result = pca.fit_transform(all_data)
    
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
    
    # Créer le dossier pour les scatterplots
    output_dir = os.path.join(config["output_dir"], "scatterplots")
    os.makedirs(output_dir, exist_ok=True)
    
    # Créer le graphique PCA
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Tracer les points pour chaque classe
    for class_id, pca_data in pca_by_class.items():
        class_name = config["class_names"].get(class_id, f"Classe {class_id}")
        class_color = config["class_colors"].get(class_id, "black")
        ax.scatter(
            pca_data[:, 0], 
            pca_data[:, 1], 
            pca_data[:, 2],
            c=class_color, 
            label=class_name, 
            alpha=0.7
        )
    
    # Ajouter les annotations
    ax.set_xlabel(f'PC1 ({variance_explained[0]:.1f}%)')
    ax.set_ylabel(f'PC2 ({variance_explained[1]:.1f}%)')
    ax.set_zlabel(f'PC3 ({variance_explained[2]:.1f}%)')
    ax.set_title('Analyse en Composantes Principales')
    ax.legend()
    
    # Sauvegarder le graphique
    pca_file = os.path.join(output_dir, "scatterplot_pca_combiné.png")
    plt.savefig(pca_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  Scatterplot PCA combiné sauvegardé dans: {pca_file}")
    return True

def main():
    """Fonction principale pour exécuter la classification sans pondération."""
    start_time = time.time()
    
    print("======================================================================")
    print(" CLASSIFICATION COMPLÈTE CORRIGÉE ")
    print("======================================================================")
    
    # Configuration de base
    config = {
        "raster_path": r"D:\UQTR\Hiver 2025\Télédétection\TP3\tr_clip.tif",
        "shapefile_path": r"D:\UQTR\Hiver 2025\Télédétection\TP3\classes.shp",
        "class_column": "Classe",
        "selected_bands": [2, 3, 4, 5, 6, 7, 8],
        "class_params": {
            1: [1e-5, 0],   # Eau - Suppression du buffer pour éviter le surclassement
            2: [1e-4, 0],   # Forêt - Suppression du buffer négatif pour conserver plus d'échantillons
            3: [3e-4, 0],   # Tourbière - Suppression du buffer négatif pour conserver plus d'échantillons
            4: [5e-4, 0],   # Herbes - Paramètres inchangés
            5: [1e-3, 0],   # Champs - Paramètres inchangés
            6: [1e-2, 0]    # Urbain - Réduction de la régularisation (de 5e-2 à 1e-2)
        },
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
        "validation": {
            "enabled": True,
            "shapefile_path": r"D:\UQTR\Hiver 2025\Télédétection\TP3\points_validation.shp",
            "class_column": "Class_code"  # CORRECTION: utiliser la bonne colonne de classe
        },
        "comparison": {
            "enabled": False
        },
        "show_weight_plots": False,  # Désactiver les graphiques de pondération
        "skip_visualizations": False  # Ne pas sauter les visualisations
    }
    
    # Utiliser des poids uniformes (1.0) pour chaque bande
    num_bands = len(config["selected_bands"])
    uniform_weights = np.ones(num_bands)
    
    print("Poids par bande (uniformes):")
    band_names = ["B2 - Bleu", "B3 - Vert", "B4 - Rouge", "B5 - RedEdge05", 
                  "B6 - RedEdge06", "B7 - RedEdge07", "B8 - PIR"]
    
    for i, band in enumerate(config["selected_bands"]):
        print(f"  {band_names[band-2]}: {uniform_weights[i]:.2f}")
    
    # Créer un répertoire pour les résultats avec un timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(r"D:\UQTR\Hiver 2025\Télédétection\TP3\resultats", f"classification_complete_corrigee_{timestamp}")
    create_output_directory(output_dir)
    config["output_dir"] = output_dir
    
    print(f"\nRépertoire de sortie: {output_dir}")
    print("\nLancement de la classification...")
    
    try:
        # Chargement et vérification des données
        raster_data, meta, shapefile = load_and_check_data(config)
        
        # Extraction des échantillons d'entraînement
        print("\nExtraction des échantillons d'entraînement...")
        from main import extract_training_samples
        classes_info = extract_training_samples(raster_data, shapefile, config)
        
        # Classification sans pondération
        print("\nClassification standard sans pondération...")
        classification, probabilities = perform_classification(raster_data, classes_info, config)
        
        # Génération des signatures spectrales
        print("\nGénération des signatures spectrales...")
        visualize_spectral_signatures(classes_info, config)
        
        # Génération des scatterplots standards
        print("\nGénération des scatterplots...")
        try:
            create_scatterplots(classes_info, config)
        except Exception as e:
            print(f"Erreur lors de la génération des scatterplots standards: {str(e)}")
        
        # Génération des scatterplots PCA
        print("\nGénération des scatterplots PCA...")
        try:
            create_pca_scatterplots(classes_info, config)
        except Exception as e:
            print(f"Erreur lors de la génération des scatterplots PCA: {str(e)}")
        
        # Enregistrement du fichier TIF
        tif_path = os.path.join(output_dir, "classification_unweighted.tif")
        save_result = save_raster(classification, meta, tif_path)
        
        if save_result:
            print(f"\nFichier TIF généré avec succès: {tif_path}")
        else:
            print("\nAttention: Le fichier TIF n'a pas été généré correctement.")
        
        # Génération de la carte de classification
        print("\nGénération de la carte de classification...")
        generate_classification_map(classification, config)
        
        # Validation de la classification
        print("\nValidation de la classification...")
        try:
            validation_results = validate_classification(classification, config)
            
            # Afficher les résultats de la validation
            if validation_results:
                print("\nRésultats de la classification:")
                print(f"  Précision validation: {validation_results['accuracy']:.4f}")
                print(f"  Kappa validation: {validation_results['kappa']:.4f}")
                
                # Enregistrer les résultats dans un fichier JSON
                results_dict = {
                    "timestamp": timestamp,
                    "precision": validation_results['accuracy'],
                    "kappa": validation_results['kappa'],
                    "weights": [float(w) for w in uniform_weights]
                }
                
                results_path = os.path.join(output_dir, "resultats_classification.json")
                with open(results_path, 'w') as f:
                    json.dump(results_dict, f, indent=2)
                
                print(f"\nRésultats enregistrés dans: {results_path}")
        except Exception as validation_error:
            print(f"\nAttention: Erreur lors de la validation : {str(validation_error)}")
            print(traceback.format_exc())
            print("La classification a été réalisée avec succès, mais la validation a échoué.")
        
        # Afficher le temps d'exécution
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"\nTemps d'exécution total: {execution_time:.2f} secondes")
        
        print("\nClassification terminée avec succès!")
        print(f"Les résultats sont disponibles dans: {output_dir}")
        print("\nLes fichiers suivants ont été générés:")
        print("- Carte de classification (.png)")
        print("- Classification au format TIF (.tif)")
        print("- Signatures spectrales (.png)")
        print("- Scatterplots (dossier scatterplots/)")
        print("- Scatterplots PCA (dossier scatterplots/)")
        
        # Afficher le chemin vers la matrice de confusion
        confusion_matrix_path = os.path.join(output_dir, "matrice_confusion.png")
        confusion_matrix_percent_path = os.path.join(output_dir, "matrice_confusion_pourcent.png")
        
        if os.path.exists(confusion_matrix_path):
            print(f"\nVoir la matrice de confusion dans:\n{confusion_matrix_path}")
        
        if os.path.exists(confusion_matrix_percent_path):
            print(f"\nVoir la matrice de confusion en pourcentage dans:\n{confusion_matrix_percent_path}")
    
    except Exception as e:
        print(f"\nErreur lors de l'exécution: {str(e)}")
        print(traceback.format_exc())

if __name__ == "__main__":
    main() 