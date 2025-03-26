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
from sklearn.metrics import confusion_matrix, accuracy_score, cohen_kappa_score, classification_report
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches

# Paramètres à modifier
FICHIER_CLASSIFICATION = r"D:\UQTR\Hiver 2025\Télédétection\TP3\classification_unsupervised.tif"  # IMPORTANT: Remplacez par le chemin de votre fichier TIFF de classification non supervisée
FICHIER_VALIDATION = r"D:\UQTR\Hiver 2025\Télédétection\TP3\points_validation.shp"  # Chemin vers le shapefile de validation
COLONNE_CLASSE = "Class_code"  # Nom de la colonne contenant les classes dans le shapefile
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

if __name__ == "__main__":
    main() 