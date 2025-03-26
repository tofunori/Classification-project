"""
Vérification des métadonnées du fichier TIF généré.
"""

import os
import rasterio

tif_path = r"D:\UQTR\Hiver 2025\Télédétection\TP3\resultats\classification_unweighted_tif_20250325_192931\classification_unweighted.tif"

try:
    with rasterio.open(tif_path) as src:
        print(f"Informations sur le fichier TIF :")
        print(f"Format : {src.driver}")
        print(f"Taille : {src.width}x{src.height} pixels")
        print(f"Nombre de bandes : {src.count}")
        print(f"CRS : {src.crs}")
        print(f"Emprise : {src.bounds}")
        print(f"Type de données : {src.dtypes}")
        
        # Lire un petit échantillon pour vérifier les valeurs
        sample = src.read(1, window=((0, 10), (0, 10)))
        print(f"\nÉchantillon des valeurs (10x10) :")
        print(sample)
        
        # Vérifier les valeurs uniques (classes)
        unique_values = set(src.read(1).flatten())
        print(f"\nClasses présentes : {sorted(list(unique_values))}")
        
except Exception as e:
    print(f"Erreur lors de la lecture du fichier TIF : {e}") 