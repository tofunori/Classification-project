"""
VISUALISATION DE CLASSIFICATION NON SUPERVISÉE (VERSION AVANCÉE)
===============================================================
Script pour visualiser un fichier TIFF de classification non supervisée existant
et générer des visualisations PCA et scatterplots similaires à la classification supervisée
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from matplotlib.colors import ListedColormap, BoundaryNorm
import seaborn as sns
import pandas as pd
import rasterio
import geopandas as gpd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score, cohen_kappa_score, classification_report
from scipy.stats import chi2

# Paramètres à modifier
FICHIER_CLASSIFICATION = r"D:\UQTR\Hiver 2025\Télédétection\TP3\classification_unsupervised.tif"  # IMPORTANT: Remplacez par le chemin de votre fichier TIFF de classification non supervisée
FICHIER_DONNEES_ORIGINALES = r"D:\UQTR\Hiver 2025\Télédétection\TP3\TP3\printemps_automne.tif"  # Chemin vers les données originales
FICHIER_VALIDATION = r"D:\UQTR\Hiver 2025\Télédétection\TP3\points_validation.shp"  # Chemin vers le shapefile de validation
COLONNE_CLASSE = "Class_code"  # Nom de la colonne contenant les classes dans le shapefile
REPERTOIRE_SORTIE = "resultats_visualisation"  # Dossier où seront enregistrés les résultats
BANDES_SELECTIONNEES = [2, 3, 4, 5, 6, 7, 8]  # Bandes à utiliser (indices commençant à 1)

# Définition des couleurs et noms de classes (pour un affichage cohérent)
COULEURS_CLASSES = {
    1: "#3288bd",  # Eau - bleu
    2: "#66c164",  # Forêt - vert
    3: "#87CEFA",  # Tourbière - bleu clair
    4: "#ffff00",  # Herbes - jaune
    5: "#f39c12",  # Champs - orange
    6: "#7f8c8d"   # Urbain - gris
}

NOMS_CLASSES = {
    1: "Eau", 
    2: "Forêt", 
    3: "Tourbière",
    4: "Herbes", 
    5: "Champs", 
    6: "Urbain"
}

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
        
        # Afficher les colonnes disponibles
        print(f"Colonnes disponibles dans le shapefile: {list(shapefile.columns)}")
        
        # Vérifier si la colonne de classe existe
        if COLONNE_CLASSE not in shapefile.columns:
            # Rechercher d'autres colonnes potentielles
            potential_cols = [col for col in shapefile.columns if 'class' in col.lower() or 'code' in col.lower() or 'value' in col.lower()]
            if potential_cols:
                class_column = potential_cols[0]
                print(f"Colonne de classe '{class_column}' utilisée à la place")
            else:
                print(f"ERREUR: Colonne de classe '{COLONNE_CLASSE}' non trouvée")
                print(f"Colonnes disponibles: {list(shapefile.columns)}")
                return None
        else:
            class_column = COLONNE_CLASSE
        
        # Afficher les valeurs uniques de la colonne de classe pour le débogage
        unique_values = shapefile[class_column].unique()
        print(f"Valeurs uniques dans la colonne '{class_column}': {unique_values}")
        
        # Initialiser les listes pour stocker les données
        y_true = []  # Valeurs de référence (du shapefile)
        y_pred = []  # Valeurs prédites (de la classification)
        
        # Pour chaque point de validation, extraire la classe prédite
        for idx, row in shapefile.iterrows():
            # Ignorer les entrées avec des valeurs de classe nulles
            if pd.isna(row[class_column]):
                continue
            
            # Obtenir la classe de référence et la convertir en entier si possible
            ref_class = row[class_column]
            try:
                if isinstance(ref_class, (float, int)):
                    ref_class = int(ref_class)
                elif isinstance(ref_class, str) and ref_class.isdigit():
                    ref_class = int(ref_class)
            except:
                # Si on ne peut pas convertir en entier, utiliser la valeur telle quelle
                pass
            
            # Obtenir les coordonnées du point
            geom = row.geometry
            
            # Convertir en coordonnées pixels
            try:
                px, py = rasterio.transform.rowcol(meta['transform'], geom.x, geom.y)
            
                # Vérifier si le point est dans les limites de l'image
                if 0 <= px < classification.shape[0] and 0 <= py < classification.shape[1]:
                    # Obtenir la classe prédite
                    pred_class = classification[px, py]
                    
                    # Ajouter aux listes
                    y_true.append(ref_class)
                    y_pred.append(pred_class)
            except Exception as e:
                print(f"Erreur lors de la conversion des coordonnées pour le point {idx}: {e}")
        
        if len(y_true) == 0:
            print("ERREUR: Aucun point valide pour la validation")
            return None
        
        # S'assurer que toutes les classes sont du même type pour éviter les erreurs
        y_true = [str(cls) for cls in y_true]
        y_pred = [str(cls) for cls in y_pred]
        
        print(f"Classes de référence: {np.unique(y_true)}")
        print(f"Classes prédites: {np.unique(y_pred)}")
        
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
        class_names = []
        for cls in classes:
            cls_id = int(cls) if cls.isdigit() else cls
            class_names.append(NOMS_CLASSES.get(cls_id, f"Classe {cls}"))
        
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
                cls_id = int(cls) if cls.isdigit() else cls
                cls_name = NOMS_CLASSES.get(cls_id, f"Classe {cls}")
                report_data.append([
                    cls_name,
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

def genere_carte_classification(classification, meta):
    """
    Génère une carte de classification avec légende dans le même style que la classification supervisée
    
    Args:
        classification (numpy.ndarray): Résultat de la classification
        meta (dict): Métadonnées du raster
    """
    print("\nGénération de la carte de classification...")
    
    # Obtenir les classes uniques dans le résultat
    unique_classes = np.unique(classification)
    unique_classes = unique_classes[unique_classes > 0]  # Ignorer la classe 0 (non classé)
    
    if len(unique_classes) > 0:
        # Préparer les couleurs pour la colormap
        cmap_colors = []
        for cls in range(int(max(unique_classes)) + 1):
            if cls == 0:  # Classe non classifiée (noir)
                cmap_colors.append((0, 0, 0, 0))  # Transparent
            else:
                # Utiliser les couleurs prédéfinies si disponibles, sinon utiliser les couleurs par défaut
                color = COULEURS_CLASSES.get(cls, plt.cm.tab10(cls % 10))
                cmap_colors.append(color)
        
        # Créer la colormap personnalisée
        cmap = ListedColormap(cmap_colors)
        
        # Créer une figure
        plt.figure(figsize=(12, 10), dpi=300)
        
        # Masquer les zones non classifiées (classe 0)
        masked_result = np.ma.masked_where(classification == 0, classification)
        
        # Définir les limites pour la colormap
        bounds = np.arange(0, max(unique_classes) + 2) - 0.5
        norm = BoundaryNorm(bounds, cmap.N)
        
        # Afficher la classification
        plt.imshow(masked_result, cmap=cmap, norm=norm)
        plt.title('Carte de Classification Non Supervisée', fontsize=16, fontweight='bold')
        
        # Ajouter la légende
        legend_elements = []
        for cls in sorted(unique_classes):
            cls_int = int(cls)
            color = COULEURS_CLASSES.get(cls_int, plt.cm.tab10(cls_int % 10))
            name = NOMS_CLASSES.get(cls_int, f"Classe {cls}")
            legend_elements.append(mpatches.Patch(facecolor=color, edgecolor='black', label=name))
        
        # Positionner la légende à droite de la carte
        plt.legend(handles=legend_elements, 
                  loc='center left', 
                  bbox_to_anchor=(1, 0.5),
                  title="CLASSES", 
                  fontsize=12,
                  title_fontsize=14,
                  frameon=True)
        
        plt.axis('off')  # Masquer les axes
        plt.tight_layout()
        
        # Sauvegarder
        output_path = os.path.join(REPERTOIRE_SORTIE, 'carte_classification.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Carte de classification sauvegardée: {output_path}")
        return True
    else:
        print("Aucune classe trouvée dans la classification")
        return False 

def genere_donnees_par_classe(classification, donnees):
    """
    Extrait les données spectrales pour chaque classe
    
    Args:
        classification (numpy.ndarray): Résultat de la classification
        donnees (numpy.ndarray): Données multispectrales originales
        
    Returns:
        dict: Dictionnaire contenant les données par classe
    """
    # Restructurer les données (nbandes, nlignes, ncols) -> (nbandes, npixels)
    n_bands, height, width = donnees.shape
    data_reshaped = donnees.reshape(n_bands, -1).T  # (npixels, nbandes)
    
    # Créer un masque pour les pixels valides (sans NaN)
    valid_mask = ~np.isnan(data_reshaped).any(axis=1)
    data_valid = data_reshaped[valid_mask]
    
    # Classification correspondante pour les pixels valides
    class_flat = classification.flatten()[valid_mask]
    
    # Obtenir les classes uniques
    classes = np.unique(class_flat)
    classes = classes[classes > 0]  # Exclure 0 s'il est présent (souvent NoData)
    
    # Créer un dictionnaire pour stocker les données par classe
    classes_info = {}
    
    # Pour chaque classe, extraire les données
    for cls in classes:
        mask = class_flat == cls
        class_data = data_valid[mask]
        
        # Si trop de points, échantillonner pour éviter des problèmes de mémoire
        if len(class_data) > 10000:
            np.random.seed(42)  # Pour reproductibilité
            sample_idx = np.random.choice(len(class_data), 10000, replace=False)
            class_data = class_data[sample_idx]
        
        # Calculer statistiques
        mean = np.mean(class_data, axis=0)
        std = np.std(class_data, axis=0)
        cov = np.cov(class_data, rowvar=False)
        
        # Stocker les informations
        classes_info[cls] = {
            'training_data': class_data,
            'mean': mean,
            'std': std,
            'cov': cov,
            'samples': len(class_data)
        }
    
    return classes_info

def genere_analyse_pca(classes_info):
    """
    Génère l'analyse PCA des classes comme dans l'image fournie
    
    Args:
        classes_info (dict): Informations sur les données par classe
    """
    print("\nGénération de l'analyse PCA...")
    
    try:
        # Collecter toutes les données
        all_data = []
        all_labels = []
        
        for class_id, info in classes_info.items():
            # Limiter le nombre d'échantillons par classe pour l'affichage
            data = info['training_data']
            if len(data) > 5000:
                np.random.seed(42)
                sample_idx = np.random.choice(len(data), 5000, replace=False)
                data = data[sample_idx]
            
            all_data.append(data)
            all_labels.extend([class_id] * len(data))
        
        # Concaténer toutes les données
        all_data = np.vstack(all_data)
        all_labels = np.array(all_labels)
        
        # Standardiser les données
        scaler = StandardScaler()
        all_data_scaled = scaler.fit_transform(all_data)
        
        # Appliquer PCA
        n_components = min(3, all_data.shape[1])
        pca = PCA(n_components=n_components)
        pca_result = pca.fit_transform(all_data_scaled)
        
        # Afficher la variance expliquée
        explained_variance = pca.explained_variance_ratio_
        print(f"Variance expliquée par les composantes principales:")
        for i, var in enumerate(explained_variance):
            print(f"  PC{i+1}: {var*100:.2f}%")
        
        # Créer la figure avec layout similaire à l'image fournie
        fig = plt.figure(figsize=(18, 16))
        
        # Titre global
        plt.suptitle('Analyse en Composantes Principales (PCA) des Classes', fontsize=20, y=0.98)
        
        # Organisation de la figure
        gs = gridspec.GridSpec(2, 2, height_ratios=[1, 0.6])
        
        # 1. Scatterplot 2D (PC1 vs PC2) en haut à gauche
        ax1 = fig.add_subplot(gs[0, 0])
        
        # Tracer les points pour chaque classe
        for class_id in np.unique(all_labels):
            mask = all_labels == class_id
            class_name = NOMS_CLASSES.get(int(class_id), f"Classe {class_id}")
            color = COULEURS_CLASSES.get(int(class_id), plt.cm.tab10(int(class_id) % 10))
            
            ax1.scatter(
                pca_result[mask, 0], 
                pca_result[mask, 1], 
                c=[color], 
                alpha=0.6,
                label=class_name,
                s=15
            )
            
            # Ajouter des ellipses de confiance pour chaque classe
            class_data = pca_result[mask, :2]
            
            if len(class_data) > 2:
                # Calculer la moyenne et la covariance
                mean = np.mean(class_data, axis=0)
                cov = np.cov(class_data, rowvar=False)
                
                # Calculer les valeurs propres et vecteurs propres
                eigenvals, eigenvecs = np.linalg.eigh(cov)
                
                # Calculer l'angle de l'ellipse
                angle = np.degrees(np.arctan2(eigenvecs[1, 0], eigenvecs[0, 0]))
                
                # Calculer la largeur et la hauteur de l'ellipse (intervalle de confiance à 95%)
                chi2_val = chi2.ppf(0.95, 2)
                width = 2 * np.sqrt(chi2_val * eigenvals[0])
                height = 2 * np.sqrt(chi2_val * eigenvals[1])
                
                # Créer et ajouter l'ellipse
                ellipse = mpatches.Ellipse(xy=(mean[0], mean[1]), 
                                         width=width, 
                                         height=height, 
                                         angle=angle, 
                                         fill=False, 
                                         edgecolor=color, 
                                         linewidth=2)
                ax1.add_patch(ellipse)
        
        ax1.set_xlabel(f'Composante Principale 1 ({explained_variance[0]*100:.1f}%)', fontsize=14)
        ax1.set_ylabel(f'Composante Principale 2 ({explained_variance[1]*100:.1f}%)', fontsize=14)
        ax1.set_title('Projection 2D des classes (PC1 vs PC2)', fontsize=16)
        ax1.grid(True, linestyle='--', alpha=0.7)
        
        # 2. Graphique 3D en haut à droite (si nous avons 3 composantes)
        if pca_result.shape[1] >= 3:
            from mpl_toolkits.mplot3d import Axes3D
            
            ax2 = fig.add_subplot(gs[0, 1], projection='3d')
            
            for class_id in np.unique(all_labels):
                mask = all_labels == class_id
                class_name = NOMS_CLASSES.get(int(class_id), f"Classe {class_id}")
                color = COULEURS_CLASSES.get(int(class_id), plt.cm.tab10(int(class_id) % 10))
                
                ax2.scatter(
                    pca_result[mask, 0], 
                    pca_result[mask, 1], 
                    pca_result[mask, 2],
                    c=[color], 
                    alpha=0.6,
                    s=15
                )
            
            ax2.set_xlabel(f'PC1 ({explained_variance[0]*100:.1f}%)', fontsize=14)
            ax2.set_ylabel(f'PC2 ({explained_variance[1]*100:.1f}%)', fontsize=14)
            ax2.set_zlabel(f'PC3 ({explained_variance[2]*100:.1f}%)', fontsize=14)
            ax2.set_title('Projection 3D des classes', fontsize=16)
            
            # Ajuster la vue 3D pour une meilleure visualisation
            ax2.view_init(30, 45)
        
        # 3. Matrice des coefficients en bas
        ax3 = fig.add_subplot(gs[1, :])
        
        components = pca.components_
        columns = [f'Bande {b}' for b in BANDES_SELECTIONNEES]
        
        sns.heatmap(
            components, 
            annot=True, 
            cmap='coolwarm', 
            xticklabels=columns,
            yticklabels=[f'PC{i+1} ({var*100:.1f}%)' for i, var in enumerate(explained_variance)],
            ax=ax3
        )
        
        ax3.set_title('Coefficients des Composantes Principales', fontsize=16)
        
        # Légende commune
        legend_elements = []
        for class_id in sorted(np.unique(all_labels)):
            class_name = NOMS_CLASSES.get(int(class_id), f"Classe {class_id}")
            color = COULEURS_CLASSES.get(int(class_id), plt.cm.tab10(int(class_id) % 10))
            legend_elements.append(Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=10, label=class_name))
            
        # Ajouter la légende en bas de la figure
        fig.legend(handles=legend_elements, 
                  loc='lower center', 
                  bbox_to_anchor=(0.5, 0.02),
                  ncol=min(6, len(legend_elements)),
                  fontsize=12,
                  title="Classes",
                  title_fontsize=14,
                  frameon=True)
        
        # Ajuster le layout
        plt.tight_layout(rect=[0, 0.05, 1, 0.95])
        
        # Sauvegarder
        output_path = os.path.join(REPERTOIRE_SORTIE, 'analyse_pca.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Analyse PCA enregistrée: {output_path}")
        
        return pca, explained_variance
        
    except Exception as e:
        print(f"ERREUR lors de la génération de l'analyse PCA: {e}")
        import traceback
        traceback.print_exc()
        return None, None

def genere_scatterplots(classes_info):
    """
    Génère des scatterplots pour visualiser la séparation des classes dans l'espace spectral
    comme dans l'image fournie
    
    Args:
        classes_info (dict): Informations sur les données par classe
    """
    print("\nGénération des scatterplots pour l'analyse de la séparation des classes...")
    
    # Créer un sous-répertoire pour les scatterplots
    scatterplot_dir = os.path.join(REPERTOIRE_SORTIE, "scatterplots")
    os.makedirs(scatterplot_dir, exist_ok=True)
    
    try:
        # Créer manuellement des paires d'indices de bandes à visualiser
        band_pairs = []
        
        # Limiter à un nombre raisonnable de paires (comme dans l'image)
        band_pairs.append((0, 1))  # Bandes 2 vs 3
        band_pairs.append((0, 2))  # Bandes 2 vs 4
        band_pairs.append((1, 3))  # Bandes 3 vs 5
        band_pairs.append((2, 4))  # Bandes 4 vs 6
        band_pairs.append((3, 5))  # Bandes 5 vs 7
        band_pairs.append((4, 6))  # Bandes 6 vs 8
        band_pairs.append((1, 5))  # Bandes 3 vs 7
        band_pairs.append((2, 6))  # Bandes 4 vs 8
        
        print(f"  {len(band_pairs)} combinaisons de bandes à visualiser")
        
        # Calculer le nombre de lignes et colonnes pour le layout
        n_plots = len(band_pairs)
        n_cols = min(3, n_plots)  # Maximum 3 colonnes
        n_rows = (n_plots + n_cols - 1) // n_cols  # Arrondi supérieur
        
        # Créer une figure avec des sous-graphiques
        fig = plt.figure(figsize=(18, 16))
        plt.suptitle('Analyse de la séparation des classes par paires de bandes', fontsize=16, y=0.98)
        
        # Préparer une légende commune
        legend_elements = []
        
        # Pour chaque classe, créer un élément de légende
        for class_id, info in classes_info.items():
            class_name = NOMS_CLASSES.get(int(class_id), f"Classe {class_id}")
            color = COULEURS_CLASSES.get(int(class_id), plt.cm.tab10(int(class_id) % 10))
            
            # Ajouter un élément à la légende commune
            legend_elements.append(
                Line2D([0], [0], marker='o', color='w', markerfacecolor=color, 
                       markersize=10, label=class_name)
            )
        
        # Pour chaque paire de bandes, créer un scatterplot dans son propre sous-graphique
        for i, band_pair in enumerate(band_pairs):
            ax = plt.subplot(n_rows, n_cols, i+1)
            
            # Extraire les indices de bandes
            band_idx1, band_idx2 = band_pair
            
            # Obtenir les numéros réels des bandes
            band1 = BANDES_SELECTIONNEES[band_idx1]
            band2 = BANDES_SELECTIONNEES[band_idx2]
            
            # Pour chaque classe, tracer les échantillons
            for class_id, info in classes_info.items():
                color = COULEURS_CLASSES.get(int(class_id), plt.cm.tab10(int(class_id) % 10))
                
                # Extraire les données pour les deux bandes sélectionnées
                data = info['training_data']
                
                # Échantillonner si trop de points
                if len(data) > 1000:
                    np.random.seed(42)
                    sample_idx = np.random.choice(len(data), 1000, replace=False)
                    data = data[sample_idx]
                
                x = data[:, band_idx1]
                y = data[:, band_idx2]
                
                # Tracer les échantillons
                ax.scatter(x, y, c=[color], alpha=0.5, edgecolors='none', s=10)
                
                # Calculer et tracer l'ellipse de confiance (intervalle de confiance à 95%)
                if len(data) > 2:  # Au moins 3 points pour calculer l'ellipse
                    # Extraire la moyenne et la covariance pour ces deux bandes
                    mean = info['mean'][[band_idx1, band_idx2]]
                    cov = np.array([[info['cov'][band_idx1, band_idx1], info['cov'][band_idx1, band_idx2]],
                                   [info['cov'][band_idx2, band_idx1], info['cov'][band_idx2, band_idx2]]])
                    
                    # Calculer les valeurs propres et vecteurs propres
                    eigenvals, eigenvecs = np.linalg.eigh(cov)
                    
                    # Calculer l'angle de l'ellipse
                    angle = np.degrees(np.arctan2(eigenvecs[1, 0], eigenvecs[0, 0]))
                    
                    # Calculer la largeur et la hauteur de l'ellipse (intervalle de confiance à 95%)
                    chi2_val = chi2.ppf(0.95, 2)  # 2 degrés de liberté pour un ellipse 2D
                    width = 2 * np.sqrt(chi2_val * eigenvals[0])
                    height = 2 * np.sqrt(chi2_val * eigenvals[1])
                    
                    # Créer et ajouter l'ellipse
                    ellipse = mpatches.Ellipse(xy=(mean[0], mean[1]), 
                                             width=width, 
                                             height=height, 
                                             angle=angle, 
                                             fill=False, 
                                             edgecolor=color, 
                                             linewidth=2)
                    ax.add_patch(ellipse)
            
            # Définir les labels et le titre pour ce sous-graphique
            ax.set_xlabel(f'Bande {band1}', fontsize=12)
            ax.set_ylabel(f'Bande {band2}', fontsize=12)
            ax.set_title(f'Bandes {band1} vs {band2}', fontsize=14)
            ax.grid(True, linestyle='--', alpha=0.7)
        
        # Ajouter une légende commune au bas de la figure
        fig.legend(handles=legend_elements, 
                  loc='lower center', 
                  bbox_to_anchor=(0.5, 0.02),
                  ncol=min(6, len(legend_elements)),
                  fontsize=12,
                  title="Classes",
                  title_fontsize=14,
                  frameon=True)
        
        plt.tight_layout(rect=[0, 0.05, 1, 0.95])  # Laisser de l'espace pour la légende
        
        # Sauvegarder
        output_path = os.path.join(scatterplot_dir, 'scatterplots_bandes.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Scatterplots des bandes enregistrés: {output_path}")
        
        return True
    
    except Exception as e:
        print(f"ERREUR lors de la génération des scatterplots: {e}")
        import traceback
        traceback.print_exc()
        return False 