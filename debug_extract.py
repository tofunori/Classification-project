"""
Script de diagnostic pour identifier l'erreur dans extract_training_samples
"""

import os
import traceback
import geopandas as gpd
from modules.data_loader import load_and_check_data

def debug_extraction():
    try:
        # Configuration simplifiée pour le diagnostic
        config = {
            "raster_path": r"D:\UQTR\Hiver 2025\Télédétection\TP3\tr_clip.tif",
            "shapefile_path": r"D:\UQTR\Hiver 2025\Télédétection\TP3\classes.shp",
            "class_column": "CLASS",
            "selected_bands": [2, 3, 4, 5, 6, 7, 8],
            "class_params": {},
            "output_dir": r"D:\UQTR\Hiver 2025\Télédétection\TP3\resultats"
        }
        
        print("Chargement des données...")
        raster_data, meta, shapefile = load_and_check_data(config)
        
        print("Vérification du contenu du shapefile...")
        print(f"Colonnes du shapefile: {list(shapefile.columns)}")
        print(f"Nombre de lignes: {len(shapefile)}")
        
        # Vérifier si la colonne de classe existe
        if config["class_column"] not in shapefile.columns:
            print(f"ERREUR: La colonne '{config['class_column']}' n'existe pas dans le shapefile!")
            print(f"Colonnes disponibles: {list(shapefile.columns)}")
            return
        
        # Vérifier les valeurs uniques dans la colonne de classe
        unique_classes = shapefile[config["class_column"]].unique()
        print(f"Classes uniques trouvées: {unique_classes}")
        
        # Tester le groupby manuellement
        print("\nTest du groupby:")
        try:
            for class_id, group in shapefile.groupby(config["class_column"]):
                print(f"Classe: {class_id}, Nombre d'éléments: {len(group)}")
        except Exception as e:
            print(f"ERREUR lors du groupby: {e}")
            print(traceback.format_exc())
        
    except Exception as e:
        print(f"ERREUR GÉNÉRALE: {e}")
        print(traceback.format_exc())
        
if __name__ == "__main__":
    debug_extraction()
