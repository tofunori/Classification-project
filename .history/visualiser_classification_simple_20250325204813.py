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
import pandas as pd
import rasterio
import geopandas as gpd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score, cohen_kappa_score, classification_report
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches

# Paramètres à modifier
FICHIER_CLASSIFICATION = "chemin/vers/classification.tif"  # Remplacer par le chemin de votre fichier TIFF
FICHIER_DONNEES_ORIGINALES = r"D:\UQTR\Hiver 2025\Télédétection\TP3\TP3\printemps_automne.tif"  # Chemin vers les données originales
FICHIER_VALIDATION = "chemin/vers/points_validation.shp"  # Remplacer par le chemin de votre shapefile de validation
COLONNE_CLASSE = "classvalue"  # Nom de la colonne contenant les classes dans le shapefile
REPERTOIRE_SORTIE = "resultats_visualisation"  # Dossier où seront enregistrés les résultats

def valide_avec_shapefile(classification, meta, shapefile_path):
    """
    Valide la classification en utilisant un shapefile de points de validation
    """
    if not os.path.exists(shapefile_path):
        print(f"ERREUR: Le fichier shapefile {shapefile_path} n'existe pas.")
        return None
    
    print(f"\nValidation de la classification avec le shapefile: {shapefile_path}")
    
    try:
        # Charger le shapefile de validation
        shapefile = gpd.read_file(shapefile_path)
        print(f"Shapefile chargé: {len(shapefile)} points")
        
        # Vérifier si la colonne de classe existe
        if COLONNE_CLASSE not in shapefile.columns:
            # Rechercher d'autres colonnes potentielles
            potential_cols = [col for col in shapefile.columns if 'class' in col.lower() or 'value' in col.lower()]
            if potential_cols:
                class_column = potential_cols[0]
                print(f"Colonne de classe '{class_column}' utilisée à la place")
            else:
                print(f"ERREUR: Colonne de classe '{COLONNE_CLASSE}' non trouvée")
                print(f"Colonnes disponibles: {list(shapefile.columns)}")
                return
        else:
            class_column = COLONNE_CLASSE
        
        # Initialiser les listes pour stocker les données
        y_true = []  # Valeurs de référence (du shapefile)
        y_pred = []  # Valeurs prédites (de la classification)
        
        # Pour chaque point de validation, extraire la classe prédite
        for idx, row in shapefile.iterrows():
            # Ignorer les entrées avec des valeurs de classe nulles
            if pd.isna(row[class_column]):
                continue
            
            # Obtenir la classe de référence
            ref_class = row[class_column]
            if isinstance(ref_class, float):
                ref_class = int(ref_class)
            
            # Obtenir les coordonnées du point
            geom = row.geometry
            
            # Convertir en coordonnées pixels
            px, py = rasterio.transform.rowcol(meta['transform'], geom.x, geom.y)
            
            # Vérifier si le point est dans les limites de l'image
            if 0 <= px < classification.shape[0] and 0 <= py < classification.shape[1]:
                # Obtenir la classe prédite
                pred_class = classification[px, py]
                
                # Ajouter aux listes
                y_true.append(ref_class)
                y_pred.append(pred_class)
        
        if len(y_true) == 0:
            print("ERREUR: Aucun point valide pour la validation")
            return
        
        # Calculer les métriques
        cm = confusion_matrix(y_true, y_pred)
        accuracy = accuracy_score(y_true, y_pred)
        kappa = cohen_kappa_score(y_true, y_pred)
        report = classification_report(y_true, y_pred, output_dict=True)
        
        print(f"Validation - Points utilisés: {len(y_true)}")
        print(f"Validation - Précision globale: {accuracy:.2f}")
        print(f"Validation - Coefficient Kappa: {kappa:.2f}")
        
        # Générer la matrice de confusion
        plt.figure(figsize=(10, 8))
        
        # Obtenir les classes uniques
        classes = np.unique(y_true + y_pred)  # Combiner pour avoir toutes les classes
        
        # Créer des labels pour les classes
        class_names = [f"Classe {cls}" for cls in classes]
        
        # Créer la matrice de confusion
        sns.heatmap(
            cm, 
            annot=True, 
            fmt='d', 
            cmap='Blues',
            xticklabels=class_names,
            yticklabels=class_names
        )
        
        plt.title(f'Matrice de Confusion\nPrécision: {accuracy:.2f}, Kappa: {kappa:.2f}', fontsize=16)
        plt.xlabel('Classe Prédite')
        plt.ylabel('Classe de Référence')
        
        # Sauvegarder
        output_path = os.path.join(REPERTOIRE_SORTIE, 'matrice_confusion.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Matrice de confusion enregistrée: {output_path}")
        
        # Créer une version normalisée (pourcentages)
        plt.figure(figsize=(10, 8))
        
        # Normaliser la matrice
        cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
        
        sns.heatmap(
            cm_percent, 
            annot=True, 
            fmt='.1f', 
            cmap='Blues',
            xticklabels=class_names,
            yticklabels=class_names
        )
        
        plt.title(f'Matrice de Confusion (Pourcentages)\nPrécision: {accuracy:.2f}, Kappa: {kappa:.2f}', fontsize=16)
        plt.xlabel('Classe Prédite')
        plt.ylabel('Classe de Référence')
        
        # Sauvegarder
        output_path = os.path.join(REPERTOIRE_SORTIE, 'matrice_confusion_pourcent.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Matrice de confusion (pourcentages) enregistrée: {output_path}")
        
        # Générer le rapport de classification
        report_data = []
        for cls in sorted(report.keys()):
            if cls not in ['accuracy', 'macro avg', 'weighted avg']:
                metrics = report[cls]
                report_data.append([
                    f"Classe {cls}",
                    f"{metrics['precision']:.2f}",
                    f"{metrics['recall']:.2f}",
                    f"{metrics['f1-score']:.2f}",
                    f"{metrics['support']}"
                ])
        
        # Ajouter les moyennes
        for avg_type in ['macro avg', 'weighted avg']:
            if avg_type in report:
                metrics = report[avg_type]
                report_data.append([
                    avg_type,
                    f"{metrics['precision']:.2f}",
                    f"{metrics['recall']:.2f}",
                    f"{metrics['f1-score']:.2f}",
                    f"{metrics['support']}"
                ])
        
        # Créer le tableau du rapport
        fig, ax = plt.figure(figsize=(10, 6), dpi=150), plt.subplot(111)
        ax.axis('tight')
        ax.axis('off')
        table = ax.table(
            cellText=report_data,
            colLabels=['Classe', 'Précision', 'Rappel', 'F1-score', 'Support'],
            cellLoc='center',
            loc='center',
            colWidths=[0.3, 0.15, 0.15, 0.15, 0.15]
        )
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.5)
        
        # Ajuster l'apparence
        for (i, j), cell in table.get_celld().items():
            if i == 0:  # Header
                cell.set_text_props(weight='bold', color='white')
                cell.set_facecolor('#4472C4')
            elif i > len(report_data) - 3:  # Moyennes
                cell.set_facecolor('#B4C7E7')
            elif i % 2 == 1:  # Lignes impaires
                cell.set_facecolor('#D9E1F2')
            else:  # Lignes paires
                cell.set_facecolor('#E9EDF4')
        
        plt.title(f'Rapport de Classification\nPrécision: {accuracy:.2f}, Kappa: {kappa:.2f}', fontsize=14, pad=20)
        plt.tight_layout()
        
        # Sauvegarder
        output_path = os.path.join(REPERTOIRE_SORTIE, 'rapport_classification.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Rapport de classification enregistré: {output_path}")
        
    except Exception as e:
        print(f"ERREUR lors de la validation: {e}")
        import traceback
        traceback.print_exc()

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
    
    # Valider avec le shapefile si disponible
    if os.path.isfile(FICHIER_VALIDATION):
        valide_avec_shapefile(classification, meta, FICHIER_VALIDATION)
    else:
        print(f"\nLe fichier de validation {FICHIER_VALIDATION} n'existe pas.")
        print("Pour générer la matrice de confusion et le rapport, modifiez la variable FICHIER_VALIDATION.")
    
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