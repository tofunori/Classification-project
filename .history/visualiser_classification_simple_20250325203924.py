"""
VISUALISATION DE CLASSIFICATION NON SUPERVISÉE (VERSION SIMPLIFIÉE)
=================================================================
Script simplifié pour visualiser un fichier TIFF de classification non supervisée existant
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import rasterio
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches

# Paramètres à modifier
FICHIER_CLASSIFICATION = "chemin/vers/classification.tif"  # Remplacer par le chemin de votre fichier TIFF
FICHIER_DONNEES_ORIGINALES = "chemin/vers/donnees.tif"  # Remplacer par le chemin de votre fichier de données originales
REPERTOIRE_SORTIE = "resultats_visualisation"  # Dossier où seront enregistrés les résultats

def main():
    # Créer le répertoire de sortie
    os.makedirs(REPERTOIRE_SORTIE, exist_ok=True)
    print(f"Résultats enregistrés dans: {REPERTOIRE_SORTIE}")
    
    # Vérifier que les fichiers existent
    if not os.path.isfile(FICHIER_CLASSIFICATION):
        print(f"ERREUR: Le fichier de classification {FICHIER_CLASSIFICATION} n'existe pas.")
        print("Veuillez modifier la variable FICHIER_CLASSIFICATION dans le script.")
        sys.exit(1)
    
    # Charger la classification
    try:
        with rasterio.open(FICHIER_CLASSIFICATION) as src:
            classification = src.read(1)  # Lire la première bande
            meta = src.meta.copy()
            print(f"Classification chargée depuis {FICHIER_CLASSIFICATION}")
            print(f"Dimensions: {classification.shape}")
            print(f"Valeurs uniques: {np.unique(classification)}")
    except Exception as e:
        print(f"Erreur lors du chargement du fichier de classification: {e}")
        sys.exit(1)
    
    # Générer la carte de classification
    print("\nGénération de la carte de classification...")
    
    # Obtenir les classes uniques
    classes = np.unique(classification)
    classes = classes[classes != 0]  # Exclure 0 s'il est présent (souvent NoData)
    
    # Créer un colormap
    cmap = plt.cm.get_cmap('tab10', len(classes))
    
    # Créer un colormap personnalisé
    color_list = [(0,0,0,0)]  # Couleur pour la classe 0 (transparente)
    for i in range(1, int(classes.max())+1):
        if i in classes:
            idx = np.where(classes == i)[0][0]
            color_list.append(cmap(idx % 10))
        else:
            color_list.append((0,0,0,0))  # Transparent pour les classes manquantes
    
    custom_cmap = ListedColormap(color_list)
    
    plt.figure(figsize=(12, 10))
    plt.imshow(classification, cmap=custom_cmap)
    
    # Créer la légende
    patches = []
    for i, cls in enumerate(classes):
        patches.append(mpatches.Patch(color=cmap(i % 10), label=f"Classe {cls}"))
    
    plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.title('Carte de Classification Non Supervisée', fontsize=16)
    plt.colorbar(label='Classes')
    plt.axis('off')
    
    # Enregistrer la carte
    output_path = os.path.join(REPERTOIRE_SORTIE, 'carte_classification.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Carte de classification enregistrée: {output_path}")
    
    # Si les données originales sont disponibles, générer les analyses supplémentaires
    if os.path.isfile(FICHIER_DONNEES_ORIGINALES):
        # Charger les données originales
        try:
            with rasterio.open(FICHIER_DONNEES_ORIGINALES) as src:
                print(f"Chargement de toutes les bandes depuis {FICHIER_DONNEES_ORIGINALES}")
                donnees_originales = src.read()
                print(f"Dimensions: {donnees_originales.shape}")
        except Exception as e:
            print(f"Erreur lors du chargement des données originales: {e}")
            print("Seule la carte de classification a été générée.")
            sys.exit(0)
        
        # Analyse PCA
        print("\nRéalisation de l'analyse en composantes principales (PCA)...")
        
        # Restructurer les données pour la PCA
        n_bands, height, width = donnees_originales.shape
        data_reshaped = donnees_originales.reshape(n_bands, -1).T
        
        # Créer un masque pour les pixels valides (sans NaN)
        valid_mask = ~np.isnan(data_reshaped).any(axis=1)
        data_valid = data_reshaped[valid_mask]
        
        # Classification correspondante pour les pixels valides
        class_flat = classification.flatten()[valid_mask]
        
        # Standardiser les données
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(data_valid)
        
        # Appliquer PCA
        n_components = min(3, n_bands)
        pca = PCA(n_components=n_components)
        pca_result = pca.fit_transform(data_scaled)
        
        # Afficher la variance expliquée
        explained_variance = pca.explained_variance_ratio_
        print(f"Variance expliquée par les composantes principales:")
        for i, var in enumerate(explained_variance):
            print(f"  PC{i+1}: {var*100:.2f}%")
        
        # Créer la figure pour les visualisations PCA
        fig = plt.figure(figsize=(18, 16))
        
        # 1. Scatter plot 2D (PC1 vs PC2)
        ax1 = plt.subplot(221)
        
        # Obtenir les classes uniques dans les données valides
        classes_valid = np.unique(class_flat)
        classes_valid = classes_valid[classes_valid != 0]
        
        # Tracer les points pour chaque classe (échantillonner si trop de points)
        for i, cls in enumerate(classes_valid):
            mask = class_flat == cls
            
            # Échantillonner si trop de points (max 5000 points par classe)
            idx = np.where(mask)[0]
            if len(idx) > 5000:
                np.random.seed(42)  # Pour reproductibilité
                idx = np.random.choice(idx, 5000, replace=False)
                mask = np.zeros_like(mask, dtype=bool)
                mask[idx] = True
            
            ax1.scatter(
                pca_result[mask, 0], 
                pca_result[mask, 1],
                c=[cmap(i % 10)],
                label=f"Classe {cls}",
                alpha=0.5,
                s=10
            )
        
        ax1.set_xlabel(f'PC1 ({explained_variance[0]*100:.1f}%)', fontsize=14)
        ax1.set_ylabel(f'PC2 ({explained_variance[1]*100:.1f}%)', fontsize=14)
        ax1.set_title('Projection 2D des classes (PC1 vs PC2)', fontsize=16)
        ax1.grid(True, linestyle='--', alpha=0.7)
        ax1.legend()
        
        # 2. Graphique 3D si nous avons 3 composantes
        if pca_result.shape[1] >= 3:
            from mpl_toolkits.mplot3d import Axes3D
            
            ax2 = plt.subplot(222, projection='3d')
            
            for i, cls in enumerate(classes_valid):
                mask = class_flat == cls
                
                # Échantillonner si trop de points
                idx = np.where(mask)[0]
                if len(idx) > 5000:
                    np.random.seed(42)
                    idx = np.random.choice(idx, 5000, replace=False)
                    mask = np.zeros_like(mask, dtype=bool)
                    mask[idx] = True
                
                ax2.scatter(
                    pca_result[mask, 0], 
                    pca_result[mask, 1],
                    pca_result[mask, 2],
                    c=[cmap(i % 10)],
                    label=f"Classe {cls}",
                    alpha=0.5,
                    s=10
                )
            
            ax2.set_xlabel(f'PC1 ({explained_variance[0]*100:.1f}%)', fontsize=14)
            ax2.set_ylabel(f'PC2 ({explained_variance[1]*100:.1f}%)', fontsize=14)
            ax2.set_zlabel(f'PC3 ({explained_variance[2]*100:.1f}%)', fontsize=14)
            ax2.set_title('Projection 3D des classes', fontsize=16)
            
            # Ajuster la vue 3D
            ax2.view_init(30, 45)
        
        # 3. Matrice des coefficients
        ax3 = plt.subplot(212)
        
        components = pca.components_
        columns = [f'Bande {i+1}' for i in range(n_bands)]
        
        sns.heatmap(
            components, 
            annot=True, 
            cmap='coolwarm', 
            xticklabels=columns,
            yticklabels=[f'PC{i+1} ({var*100:.1f}%)' for i, var in enumerate(explained_variance)],
            ax=ax3
        )
        
        ax3.set_title('Coefficients des Composantes Principales', fontsize=16)
        
        # Titre global
        plt.suptitle('Analyse en Composantes Principales (PCA)', fontsize=20, y=0.98)
        
        # Ajuster le layout
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        
        # Sauvegarder
        output_path = os.path.join(REPERTOIRE_SORTIE, 'analyse_pca.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Visualisation PCA enregistrée: {output_path}")
        
        # Matrice de similarité
        print("\nGénération de la matrice de similarité entre classes...")
        
        # Calculer les centres des clusters
        centers = []
        for cls in classes_valid:
            mask = class_flat == cls
            center = np.mean(data_scaled[mask], axis=0)
            centers.append(center)
        
        centers = np.array(centers)
        
        # Calculer les distances euclidiennes entre centres
        from sklearn.metrics.pairwise import euclidean_distances
        distances = euclidean_distances(centers)
        
        # Normaliser pour obtenir une mesure de similarité
        similarities = 1 / (1 + distances)
        
        # Visualiser la matrice de similarité
        plt.figure(figsize=(10, 8))
        
        ax = sns.heatmap(
            similarities, 
            annot=True, 
            fmt='.2f', 
            cmap='viridis',
            xticklabels=[f'Classe {cls}' for cls in classes_valid],
            yticklabels=[f'Classe {cls}' for cls in classes_valid]
        )
        
        plt.title('Matrice de Similarité entre Classes', fontsize=16)
        
        # Enregistrer la matrice
        output_path = os.path.join(REPERTOIRE_SORTIE, 'matrice_similarite.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Matrice de similarité enregistrée: {output_path}")
    else:
        print(f"\nLe fichier de données originales {FICHIER_DONNEES_ORIGINALES} n'existe pas.")
        print("Pour générer l'analyse PCA et la matrice de similarité, modifiez la variable FICHIER_DONNEES_ORIGINALES.")

if __name__ == "__main__":
    main() 