"""
VISUALISATION DE CLASSIFICATION NON SUPERVISÉE
=============================================
Script pour visualiser un fichier TIFF de classification non supervisée existant
et générer des visualisations : graphiques PCA, matrice de confusion, carte de classification
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import rasterio
from rasterio.mask import mask
import geopandas as gpd
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, accuracy_score, cohen_kappa_score, classification_report
from sklearn.preprocessing import StandardScaler
from matplotlib.colors import ListedColormap
import argparse

def charge_classification(chemin_fichier):
    """
    Charge un fichier TIFF de classification
    
    Args:
        chemin_fichier (str): Chemin vers le fichier TIFF de classification
        
    Returns:
        tuple: Classification et métadonnées
    """
    try:
        with rasterio.open(chemin_fichier) as src:
            classification = src.read(1)  # Lire la première bande
            meta = src.meta.copy()
            print(f"Classification chargée depuis {chemin_fichier}")
            print(f"Dimensions: {classification.shape}")
            print(f"Valeurs uniques: {np.unique(classification)}")
            return classification, meta
    except Exception as e:
        print(f"Erreur lors du chargement du fichier: {e}")
        sys.exit(1)

def charge_donnees_originales(chemin_fichier, bands=None):
    """
    Charge un fichier TIFF de données multispectrales originales
    
    Args:
        chemin_fichier (str): Chemin vers le fichier TIFF multispectral
        bands (list): Liste des bandes à charger (indice base 1)
        
    Returns:
        numpy.ndarray: Données multispectrales
    """
    try:
        with rasterio.open(chemin_fichier) as src:
            if bands:
                print(f"Chargement des bandes sélectionnées: {bands}")
                data = np.stack([src.read(b) for b in bands])
            else:
                print(f"Chargement de toutes les bandes")
                data = src.read()
                
            print(f"Données originales chargées depuis {chemin_fichier}")
            print(f"Dimensions: {data.shape}")
            return data
    except Exception as e:
        print(f"Erreur lors du chargement des données originales: {e}")
        return None

def genere_carte_classification(classification, meta, output_dir, class_colors=None, class_names=None):
    """
    Génère une carte de la classification
    
    Args:
        classification (numpy.ndarray): Raster de classification
        meta (dict): Métadonnées du raster
        output_dir (str): Répertoire de sortie
        class_colors (dict): Couleurs par classe
        class_names (dict): Noms des classes
    """
    print("\nGénération de la carte de classification...")
    
    # Obtenir les classes uniques
    classes = np.unique(classification)
    classes = classes[classes != 0]  # Exclure 0 s'il est présent (souvent NoData)
    
    # Créer des couleurs par défaut si non fournies
    if class_colors is None:
        class_colors = {}
        cmap = plt.cm.get_cmap('tab10', len(classes))
        for i, cls in enumerate(classes):
            class_colors[cls] = cmap(i)
    
    # Créer des noms par défaut si non fournis
    if class_names is None:
        class_names = {cls: f"Classe {cls}" for cls in classes}
    
    # Créer un colormap personnalisé
    color_list = [class_colors.get(cls, (0,0,0,0)) for cls in range(int(classes.max())+1)]
    custom_cmap = ListedColormap(color_list)
    
    plt.figure(figsize=(12, 10))
    plt.imshow(classification, cmap=custom_cmap)
    
    # Créer la légende
    patches = []
    for cls in classes:
        from matplotlib.patches import Patch
        patches.append(Patch(color=class_colors[cls], label=class_names.get(cls, f"Classe {cls}")))
    
    plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.title('Carte de Classification Non Supervisée', fontsize=16)
    plt.colorbar(label='Classes')
    plt.axis('off')
    
    # Enregistrer la carte
    output_path = os.path.join(output_dir, 'carte_classification.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Carte de classification enregistrée: {output_path}")

def genere_analyse_pca(donnees_originales, classification, output_dir):
    """
    Réalise une analyse en composantes principales (PCA) et génère des visualisations
    
    Args:
        donnees_originales (numpy.ndarray): Données multispectrales originales
        classification (numpy.ndarray): Résultat de la classification
        output_dir (str): Répertoire de sortie
    """
    print("\nRéalisation de l'analyse en composantes principales (PCA)...")
    
    # Restructurer les données pour la PCA (nbandes, nlignes, ncols) -> (nbandes, npixels)
    n_bands, height, width = donnees_originales.shape
    data_reshaped = donnees_originales.reshape(n_bands, -1).T  # (npixels, nbandes)
    
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
    
    # Obtenir les classes uniques
    classes = np.unique(class_flat)
    classes = classes[classes != 0]  # Exclure 0 s'il est présent (souvent NoData)
    
    # Créer un colormap
    cmap = plt.cm.get_cmap('tab10', len(classes))
    
    # Tracer les points pour chaque classe
    for i, cls in enumerate(classes):
        mask = class_flat == cls
        ax1.scatter(
            pca_result[mask, 0], 
            pca_result[mask, 1],
            c=[cmap(i)],
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
        
        for i, cls in enumerate(classes):
            mask = class_flat == cls
            ax2.scatter(
                pca_result[mask, 0], 
                pca_result[mask, 1],
                pca_result[mask, 2],
                c=[cmap(i)],
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
    output_path = os.path.join(output_dir, 'analyse_pca.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Visualisation PCA enregistrée: {output_path}")
    
    return pca, explained_variance

def genere_matrice_similarite(classification, donnees_originales, output_dir, pca=None):
    """
    Génère une matrice de similarité entre les classes
    
    Args:
        classification (numpy.ndarray): Résultat de la classification
        donnees_originales (numpy.ndarray): Données multispectrales
        output_dir (str): Répertoire de sortie
        pca (object): Objet PCA si disponible
    """
    print("\nGénération de la matrice de similarité entre classes...")
    
    # Restructurer les données
    n_bands, height, width = donnees_originales.shape
    data_reshaped = donnees_originales.reshape(n_bands, -1).T
    
    # Masque pour les pixels valides
    valid_mask = ~np.isnan(data_reshaped).any(axis=1)
    data_valid = data_reshaped[valid_mask]
    
    # Classification correspondante
    class_flat = classification.flatten()[valid_mask]
    
    # Classes uniques
    classes = np.unique(class_flat)
    classes = classes[classes != 0]  # Exclure 0 s'il est présent (souvent NoData)
    
    # Standardiser les données
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data_valid)
    
    # Appliquer PCA si fourni
    if pca is not None:
        data_scaled = pca.transform(data_scaled)
    
    # Calculer les centres des clusters
    centers = []
    for cls in classes:
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
        xticklabels=[f'Classe {cls}' for cls in classes],
        yticklabels=[f'Classe {cls}' for cls in classes]
    )
    
    plt.title('Matrice de Similarité entre Classes', fontsize=16)
    
    # Enregistrer la matrice
    output_path = os.path.join(output_dir, 'matrice_similarite.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Matrice de similarité enregistrée: {output_path}")

def valide_avec_shapefile(classification, meta, shapefile_path, output_dir, class_column='classvalue'):
    """
    Valide la classification en utilisant un shapefile de points de validation
    
    Args:
        classification (numpy.ndarray): Classification à valider
        meta (dict): Métadonnées du raster
        shapefile_path (str): Chemin vers le shapefile de validation
        output_dir (str): Répertoire de sortie
        class_column (str): Nom de la colonne contenant les classes de référence
        
    Returns:
        dict: Métriques de validation ou None en cas d'erreur
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
        if class_column not in shapefile.columns:
            # Rechercher d'autres colonnes potentielles
            potential_cols = [col for col in shapefile.columns if 'class' in col.lower() or 'value' in col.lower()]
            if potential_cols:
                class_column = potential_cols[0]
                print(f"Colonne de classe '{class_column}' utilisée à la place")
            else:
                print(f"ERREUR: Colonne de classe '{class_column}' non trouvée")
                print(f"Colonnes disponibles: {list(shapefile.columns)}")
                return None
        
        # Initialiser les listes pour stocker les données
        y_true = []  # Valeurs de référence (du shapefile)
        y_pred = []  # Valeurs prédites (de la classification)
        
        # Pour chaque point de validation, extraire la classe prédite
        with rasterio.open(meta['transform'].to_gdal(), meta['crs']) as src:
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
            return None
        
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
        output_path = os.path.join(output_dir, 'matrice_confusion.png')
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
        output_path = os.path.join(output_dir, 'matrice_confusion_pourcent.png')
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
        output_path = os.path.join(output_dir, 'rapport_classification.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Rapport de classification enregistré: {output_path}")
        
        return {
            'accuracy': accuracy,
            'kappa': kappa,
            'confusion_matrix': cm,
            'report': report
        }
    
    except Exception as e:
        print(f"ERREUR lors de la validation: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    parser = argparse.ArgumentParser(description='Visualiser un fichier TIFF de classification non supervisée')
    parser.add_argument('--classification', required=True, help='Chemin vers le fichier TIFF de classification')
    parser.add_argument('--donnees_originales', help='Chemin vers le fichier TIFF des données originales')
    parser.add_argument('--output_dir', default='./resultats_visualisation', help='Répertoire de sortie pour les résultats')
    parser.add_argument('--bands', nargs='+', type=int, help='Indices des bandes à utiliser (base 1)')
    parser.add_argument('--validation', help='Chemin vers le fichier shapefile de validation')
    parser.add_argument('--classe_colonne', default='classvalue', help='Nom de la colonne contenant les classes dans le shapefile')
    
    args = parser.parse_args()
    
    # Créer le répertoire de sortie
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Résultats enregistrés dans: {args.output_dir}")
    
    # Charger la classification
    classification, meta = charge_classification(args.classification)
    
    # Générer la carte de classification
    genere_carte_classification(classification, meta, args.output_dir)
    
    # Si un shapefile de validation est fourni, l'utiliser
    if args.validation:
        valide_avec_shapefile(classification, meta, args.validation, args.output_dir, args.classe_colonne)
    
    # Si les données originales sont fournies, générer les analyses supplémentaires
    if args.donnees_originales:
        donnees_originales = charge_donnees_originales(args.donnees_originales, args.bands)
        
        if donnees_originales is not None:
            # Analyse PCA
            pca, _ = genere_analyse_pca(donnees_originales, classification, args.output_dir)
            
            # Matrice de similarité
            genere_matrice_similarite(classification, donnees_originales, args.output_dir, pca)
    else:
        print("\nAucune donnée originale fournie. Seule la carte de classification est générée.")
        print("Pour générer l'analyse PCA et la matrice de similarité, fournissez le chemin vers les données originales.")

if __name__ == "__main__":
    main() 