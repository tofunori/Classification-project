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
REPERTOIRE_SORTIE = r"D:\UQTR\Hiver 2025\Télédétection\TP3\resultats\Classification non-superviser"  # Dossier où seront enregistrés les résultats

def valide_avec_shapefile(classification, meta, shapefile_path):
    """
    Valide la classification en utilisant un shapefile de points de validation
    
    Args:
        classification (numpy.ndarray): Résultat de la classification
        meta (dict): Métadonnées du raster
        shapefile_path (str): Chemin vers le shapefile de validation
        
    Returns:
        dict: Métriques de validation ou None en cas d'erreur
    """
    if not os.path.exists(shapefile_path):
        print(f"ERREUR: Le fichier shapefile {shapefile_path} n'existe pas.")
        return None
    
    print(f"\nValidation de la classification avec le shapefile: {shapefile_path}")
    
    # Créer un sous-répertoire pour les matrices de confusion
    confusion_dir = os.path.join(REPERTOIRE_SORTIE, "matrices_confusion")
    os.makedirs(confusion_dir, exist_ok=True)
    
    try:
        # Charger le shapefile de validation
        shapefile = gpd.read_file(shapefile_path)
        print(f"Shapefile chargé: {len(shapefile)} points")
        
        # Vérifier si la colonne de classe existe
        if COLONNE_CLASSE not in shapefile.columns:
            # Rechercher d'autres colonnes potentielles
            potential_cols = [col for col in shapefile.columns if 'class' in col.lower() or 'code' in col.lower()]
            if potential_cols:
                class_column = potential_cols[0]
                print(f"Colonne de classe '{class_column}' utilisée à la place")
            else:
                print(f"ERREUR: Colonne de classe '{COLONNE_CLASSE}' non trouvée")
                return None
        else:
            class_column = COLONNE_CLASSE
        
        # Initialiser les listes pour stocker les données
        y_true = []  # Valeurs de référence (du shapefile)
        y_pred = []  # Valeurs prédites (de la classification)
        
        # Pour chaque point de validation, extraire la classe prédite
        for idx, row in shapefile.iterrows():
            # Obtenir la classe de référence
            ref_class = row[class_column]
            
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
        
        print(f"Validation - Points utilisés: {len(y_true)}")
        print(f"Validation - Précision globale: {accuracy:.2f}")
        
        # Générer la matrice de confusion
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Matrice de Confusion\nPrécision: {accuracy:.2f}', fontsize=16)
        plt.xlabel('Classe Prédite')
        plt.ylabel('Classe de Référence')
        
        # Sauvegarder
        output_path = os.path.join(confusion_dir, 'matrice_confusion.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Matrice de confusion enregistrée: {output_path}")
        
        return {
            'accuracy': accuracy,
            'confusion_matrix': cm
        }
        
    except Exception as e:
        print(f"ERREUR lors de la validation: {e}")
        import traceback
        traceback.print_exc()
        return None

def visualise_classification(fichier_classification):
    """
    Visualise un fichier de classification existant
    
    Args:
        fichier_classification (str): Chemin vers le fichier TIFF de classification
        
    Returns:
        tuple: Classification et métadonnées
    """
    try:
        # Ouvrir le fichier de classification
        with rasterio.open(fichier_classification) as src:
            classification = src.read(1)  # Lire la première bande
            meta = src.meta.copy()
            
            print(f"Classification chargée depuis {fichier_classification}")
            print(f"Dimensions: {classification.shape}")
            print(f"Valeurs uniques: {np.unique(classification)}")
            
            # Créer un sous-répertoire pour la carte
            carte_dir = os.path.join(REPERTOIRE_SORTIE, "cartes")
            os.makedirs(carte_dir, exist_ok=True)
            
            # Visualiser la classification
            plt.figure(figsize=(12, 10))
            
            # Masquer les zones non classifiées (classe 0 si présente)
            masque = classification != 0 if 0 in np.unique(classification) else None
            
            # Afficher avec une palette de couleurs
            plt.imshow(classification, cmap='viridis')
            plt.colorbar(label='Classe')
            plt.title('Carte de Classification')
            plt.axis('off')
            
            # Sauvegarder
            output_path = os.path.join(carte_dir, 'carte_classification_simple.png')
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"Carte de classification enregistrée: {output_path}")
            
            return classification, meta
            
    except Exception as e:
        print(f"Erreur lors de la visualisation: {e}")
        import traceback
        traceback.print_exc()
        return None, None

def main():
    """
    Fonction principale qui exécute la visualisation
    """
    try:
        # Créer le répertoire de sortie principal
        os.makedirs(REPERTOIRE_SORTIE, exist_ok=True)
        print(f"Répertoire de sortie créé: {REPERTOIRE_SORTIE}")
        
        # Vérifier que les fichiers existent
        if not os.path.isfile(FICHIER_CLASSIFICATION):
            print(f"ERREUR: Le fichier de classification {FICHIER_CLASSIFICATION} n'existe pas.")
            print("Veuillez modifier la variable FICHIER_CLASSIFICATION dans le script.")
            sys.exit(1)
            
        # Visualiser la classification
        classification, meta = visualise_classification(FICHIER_CLASSIFICATION)
        
        if classification is None:
            print("Impossible de continuer sans classification valide.")
            sys.exit(1)
            
        # Valider avec le shapefile si disponible
        if os.path.isfile(FICHIER_VALIDATION):
            valide_avec_shapefile(classification, meta, FICHIER_VALIDATION)
        else:
            print(f"\nLe fichier de validation {FICHIER_VALIDATION} n'existe pas.")
            print("Pour générer la matrice de confusion, modifiez la variable FICHIER_VALIDATION.")
            
        print("\nTraitement terminé. Vérifiez les résultats dans le répertoire:", REPERTOIRE_SORTIE)
            
    except Exception as e:
        print(f"Erreur lors de l'exécution: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main() 