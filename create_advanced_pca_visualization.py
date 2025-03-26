"""
VISUALISATION PCA AVANCÉE COMPLÈTE
=======================================
Ce script génère une visualisation avancée de l'analyse en composantes principales (PCA)
pour les données de classification, incluant les projections 2D et 3D ainsi que
la matrice des coefficients des composantes principales dans un format similaire
à l'image partagée.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as patches
from matplotlib.lines import Line2D
from sklearn.decomposition import PCA
from datetime import datetime
import time
import traceback
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import chi2

# Importer les modules du projet
from modules.data_loader import load_and_check_data, create_output_directory

def create_custom_pca_visualization(classes_info, config, output_dir):
    """
    Crée une visualisation PCA avancée personnalisée avec projections 2D, 3D 
    et heatmap des coefficients des composantes principales.
    
    Args:
        classes_info (dict): Informations sur les classes
        config (dict): Configuration contenant les paramètres de visualisation
        output_dir (str): Répertoire de sortie pour les visualisations
        
    Returns:
        bool: True si réussi, False sinon
    """
    print("\nCréation de la visualisation PCA avancée personnalisée...")
    
    try:
        # Collecter toutes les données d'entraînement
        all_data = []
        all_labels = []
        
        for class_id, info in classes_info.items():
            all_data.append(info['training_data'])
            all_labels.extend([class_id] * len(info['training_data']))
        
        # Concaténer toutes les données
        all_data = np.vstack(all_data)
        all_labels = np.array(all_labels)
        
        # Appliquer PCA pour réduire à 3 dimensions
        pca = PCA(n_components=3)
        pca_result = pca.fit_transform(all_data)
        
        # Variance expliquée
        explained_variance = pca.explained_variance_ratio_ * 100
        print(f"  Variance expliquée par les composantes principales:")
        print(f"    PC1: {explained_variance[0]:.1f}%")
        print(f"    PC2: {explained_variance[1]:.1f}%")
        print(f"    PC3: {explained_variance[2]:.1f}%")
        
        # Configuration de la figure
        plt.figure(figsize=(15, 12))
        
        # Titre principal
        plt.suptitle("Analyse en Composantes Principales (PCA) des Classes", 
                    fontsize=18, y=0.98)
        
        # Créer un layout avec GridSpec
        gs = gridspec.GridSpec(2, 2, height_ratios=[2, 1])
        
        # 1. Projection 2D (PC1 vs PC2)
        ax1 = plt.subplot(gs[0, 0])
        ax1.set_title("Projection 2D des classes (PC1 vs PC2)", fontsize=14)
        
        # Tracer les points pour chaque classe et ajouter des ellipses
        for class_id in np.unique(all_labels):
            class_mask = all_labels == class_id
            class_name = config["class_names"].get(class_id, f"Classe {class_id}")
            color = config["class_colors"].get(class_id, f"C{class_id%10}")
            
            # Tracer les points
            ax1.scatter(
                pca_result[class_mask, 0], 
                pca_result[class_mask, 1], 
                c=color, 
                alpha=0.6,
                label=class_name
            )
            
            # Ajouter une ellipse pour cette classe
            class_data = pca_result[class_mask, :2]
            if len(class_data) > 2:
                # Calculer la moyenne et la covariance
                mean = np.mean(class_data, axis=0)
                cov = np.cov(class_data, rowvar=False)
                
                # Vérifier que la matrice de covariance est valide
                if np.all(np.linalg.eigvals(cov) > 0):
                    # Calculer les valeurs et vecteurs propres
                    eigenvals, eigenvecs = np.linalg.eigh(cov)
                    
                    # Calculer l'angle de l'ellipse
                    angle = np.degrees(np.arctan2(eigenvecs[1, 0], eigenvecs[0, 0]))
                    
                    # Calculer la largeur et la hauteur (intervalle de confiance à 95%)
                    chi2_val = chi2.ppf(0.95, 2)
                    width = 2 * np.sqrt(chi2_val * eigenvals[0])
                    height = 2 * np.sqrt(chi2_val * eigenvals[1])
                    
                    # Créer et ajouter l'ellipse
                    ellipse = patches.Ellipse(
                        xy=(mean[0], mean[1]), 
                        width=width, 
                        height=height, 
                        angle=angle, 
                        fill=False, 
                        edgecolor=color, 
                        linewidth=2
                    )
                    ax1.add_patch(ellipse)
        
        ax1.set_xlabel(f'Composante Principale 1 ({explained_variance[0]:.1f}%)', fontsize=12)
        ax1.set_ylabel(f'Composante Principale 2 ({explained_variance[1]:.1f}%)', fontsize=12)
        ax1.grid(True, linestyle='--', alpha=0.7)
        
        # 2. Projection 3D
        ax2 = plt.subplot(gs[0, 1], projection='3d')
        ax2.set_title("Projection 3D des classes", fontsize=14)
        
        for class_id in np.unique(all_labels):
            class_mask = all_labels == class_id
            class_name = config["class_names"].get(class_id, f"Classe {class_id}")
            color = config["class_colors"].get(class_id, f"C{class_id%10}")
            
            ax2.scatter(
                pca_result[class_mask, 0], 
                pca_result[class_mask, 1], 
                pca_result[class_mask, 2],
                c=color, 
                alpha=0.6,
                label=class_name
            )
        
        ax2.set_xlabel(f'PC1 ({explained_variance[0]:.1f}%)', fontsize=10)
        ax2.set_ylabel(f'PC2 ({explained_variance[1]:.1f}%)', fontsize=10)
        ax2.set_zlabel(f'PC3 ({explained_variance[2]:.1f}%)', fontsize=10)
        
        # Ajuster la vue 3D
        ax2.view_init(30, 45)
        
        # 3. Heatmap des coefficients
        ax3 = plt.subplot(gs[1, :])
        ax3.set_title("Coefficients des Composantes Principales", fontsize=14)
        
        # Préparer les données pour la heatmap
        components = pca.components_[:3, :]  # Seulement les 3 premières composantes
        
        # Étiquettes pour les bandes
        band_names = [f"Bande {b}" for b in config["selected_bands"]]
        
        # Créer des étiquettes pour les composantes
        component_names = [
            f"PC1 ({explained_variance[0]:.1f}%)",
            f"PC2 ({explained_variance[1]:.1f}%)",
            f"PC3 ({explained_variance[2]:.1f}%)"
        ]
        
        # Créer la heatmap
        sns.heatmap(
            components, 
            annot=True, 
            fmt=".2f", 
            cmap="RdBu_r",
            center=0, 
            vmin=-0.6, 
            vmax=0.6,
            xticklabels=band_names,
            yticklabels=component_names,
            ax=ax3
        )
        
        # Légende commune en bas
        handles, labels = ax1.get_legend_handles_labels()
        plt.figlegend(
            handles, 
            labels, 
            loc='lower center', 
            ncol=6, 
            bbox_to_anchor=(0.5, 0.02),
            fontsize=12
        )
        
        # Ajuster la mise en page
        plt.tight_layout(rect=[0, 0.05, 1, 0.95])
        
        # Sauvegarder la figure
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, "pca_analyse_complete.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  Analyse PCA avancée sauvegardée dans: {output_path}")
        return True
        
    except Exception as e:
        print(f"  Erreur lors de la création de la visualisation PCA: {str(e)}")
        print(traceback.format_exc())
        return False

def main():
    """Fonction principale pour exécuter la visualisation PCA avancée."""
    start_time = time.time()
    
    print("======================================================================")
    print(" VISUALISATION PCA AVANCÉE COMPLÈTE ")
    print("======================================================================")
    
    # Configuration de base
    config = {
        "raster_path": r"D:\UQTR\Hiver 2025\Télédétection\TP3\tr_clip.tif",
        "shapefile_path": r"D:\UQTR\Hiver 2025\Télédétection\TP3\classes.shp",
        "class_column": "Classe",
        "selected_bands": [2, 3, 4, 5, 6, 7, 8],
        "class_params": {
            1: [1e-5, 0],   # Eau
            2: [1e-4, 0],   # Forêt
            3: [3e-4, 0],   # Tourbière
            4: [5e-4, 0],   # Herbes
            5: [1e-3, 0],   # Champs
            6: [1e-2, 0]    # Urbain
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
        }
    }
    
    # Créer un répertoire pour les résultats avec un timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(r"D:\UQTR\Hiver 2025\Télédétection\TP3\resultats", f"pca_complet_{timestamp}")
    create_output_directory(output_dir)
    config["output_dir"] = output_dir
    
    print(f"\nRépertoire de sortie: {output_dir}")
    print("\nChargement des données pour l'analyse PCA...")
    
    try:
        # Chargement et vérification des données
        raster_data, meta, shapefile = load_and_check_data(config)
        
        # Extraction des échantillons d'entraînement
        print("\nExtraction des échantillons d'entraînement...")
        from main import extract_training_samples
        classes_info = extract_training_samples(raster_data, shapefile, config)
        
        # Génération de la visualisation PCA avancée
        success = create_custom_pca_visualization(classes_info, config, output_dir)
        
        if success:
            print("\nAnalyse PCA complète terminée avec succès!")
            print(f"La visualisation est disponible dans: {output_dir}")
        else:
            print("\nErreur lors de la génération de la visualisation PCA.")
        
        # Afficher le temps d'exécution
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"\nTemps d'exécution total: {execution_time:.2f} secondes")
        
    except Exception as e:
        print(f"\nErreur lors de l'exécution: {str(e)}")
        print(traceback.format_exc())

if __name__ == "__main__":
    main() 