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