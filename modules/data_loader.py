"""
DATA LOADER MODULE
=================
Ce module gère le chargement, la vérification et la sauvegarde des données.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import box
import rasterio
from rasterio.mask import mask
import geopandas as gpd

def create_output_directory(output_dir):
    """Crée le répertoire de sortie s'il n'existe pas."""
    try:
        os.makedirs(output_dir, exist_ok=True)
        print(f"Répertoire de sortie créé/vérifié: {output_dir}")
        return True
    except Exception as e:
        print(f"Erreur lors de la création du répertoire: {e}")
        return False

def save_raster(data, meta, output_path):
    """Sauvegarde les données sous forme de fichier raster."""
    try:
        with rasterio.open(output_path, 'w', **meta) as dst:
            dst.write(data, 1)
        print(f"Raster sauvegardé avec succès: {output_path}")
        return True
    except Exception as e:
        print(f"Erreur lors de la sauvegarde du raster: {e}")
        return False

def load_and_check_data(config):
    """
    Charge les données raster et vectorielles, vérifie leur compatibilité.
    Retourne les données raster, les métadonnées et le shapefile.
    """
    print("Chargement et vérification des données...")
    
    # Chargement du shapefile
    shapefile = gpd.read_file(config["shapefile_path"])
    print(f"Shapefile chargé: {len(shapefile)} polygones")
    print(f"Système de coordonnées du shapefile: {shapefile.crs}")
    
    # Chargement des métadonnées du raster
    with rasterio.open(config["raster_path"]) as src:
        raster_crs = src.crs
        raster_bounds = src.bounds
        raster_meta = src.meta.copy()
        print(f"Système de coordonnées du raster: {raster_crs}")
        print(f"Étendue du raster: {raster_bounds}")
        
        # Chargement des données raster
        if config["selected_bands"]:
            raster_data = np.stack([src.read(b) for b in config["selected_bands"]])
            print(f"Bandes sélectionnées: {config['selected_bands']}")
        else:
            raster_data = src.read()
            print(f"Toutes les bandes utilisées: {src.count}")
        
        print(f"Dimensions du raster: {raster_data.shape}")
    
    # Vérification de la compatibilité des systèmes de coordonnées
    if shapefile.crs != raster_crs:
        print(f"ATTENTION: Les systèmes de coordonnées sont différents.")
        print(f"Reprojection du shapefile vers {raster_crs}...")
        try:
            shapefile = shapefile.to_crs(raster_crs)
            print("Reprojection réussie.")
        except Exception as e:
            print(f"Erreur lors de la reprojection: {e}")
            print("Tentative de continuer sans reprojection...")
    
    # Vérification du chevauchement spatial
    raster_bbox = box(raster_bounds.left, raster_bounds.bottom, 
                      raster_bounds.right, raster_bounds.top)
    
    intersects = False
    for geom in shapefile.geometry:
        if geom.intersects(raster_bbox):
            intersects = True
            break
    
    if not intersects:
        print("ERREUR CRITIQUE: Le shapefile et le raster ne se chevauchent pas!")
        # Visualisation pour le débogage
        raster_gdf = gpd.GeoDataFrame(geometry=[raster_bbox], crs=raster_crs)
        fig, ax = plt.subplots(figsize=(10, 10))
        raster_gdf.boundary.plot(ax=ax, color='red', label='Raster')
        shapefile.boundary.plot(ax=ax, color='blue', label='Shapefile')
        ax.legend()
        ax.set_title('Comparaison des étendues spatiales')
        debug_path = os.path.join(config["output_dir"], 'debug_extents.png')
        plt.savefig(debug_path)
        print(f"Image de débogage sauvegardée dans {debug_path}")
        
        raise ValueError("Les données shapefile et raster ne se chevauchent pas!")
    
    print("Les données shapefile et raster se chevauchent correctement.")
    return raster_data, raster_meta, shapefile
