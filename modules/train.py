"""
TRAIN MODULE
===========
Ce module gère l'extraction des échantillons d'entraînement
et le calcul des statistiques pour les classes.
"""

import numpy as np
import rasterio
from rasterio.mask import mask

def extract_training_samples(raster_data, shapefile, config):
    """
    Extrait les échantillons d'entraînement pour chaque classe à partir du shapefile.
    Retourne un dictionnaire contenant les informations statistiques par classe.
    """
    print("\nExtraction des échantillons d'entraînement...")
    classes_info = {}
    
    for class_id, group in shapefile.groupby(config["class_column"]):
        # Obtenir les paramètres pour cette classe
        class_reg, class_buffer = config["class_params"].get(class_id, [1e-5, 0])
        print(f"Traitement de la classe {class_id} ({len(group)} polygones)...")
        
        # Appliquer un buffer si nécessaire
        if class_buffer != 0:
            group['geometry'] = group.geometry.buffer(class_buffer)
            print(f"  Buffer de {class_buffer} appliqué")
        
        # Extraire les échantillons pour cette classe
        class_samples = []
        
        # Si les données raster sont déjà chargées, utiliser ces données
        if raster_data is not None:
            src = rasterio.open(config["raster_path"])
            for _, row in group.iterrows():
                try:
                    # Extraire les pixels à l'intérieur du polygone
                    geom = [row.geometry.__geo_interface__]
                    out_image, _ = mask(src, geom, crop=True, all_touched=True)
                    
                    # Sélectionner les bandes spécifiques si demandé
                    if config["selected_bands"]:
                        out_image = np.stack([out_image[i-1] for i in config["selected_bands"]])
                    
                    # Filtrer les pixels valides
                    valid_pixels = []
                    for i in range(out_image.shape[1]):
                        for j in range(out_image.shape[2]):
                            pixel = out_image[:, i, j]
                            if not np.isnan(pixel).any() and not (pixel == 0).all():
                                valid_pixels.append(pixel)
                    
                    if valid_pixels:
                        class_samples.extend(valid_pixels)
                except Exception as e:
                    print(f"  Erreur lors du traitement d'un polygone: {e}")
                    continue
            src.close()
        else:
            with rasterio.open(config["raster_path"]) as src:
                for _, row in group.iterrows():
                    try:
                        # Extraire les pixels à l'intérieur du polygone
                        geom = [row.geometry.__geo_interface__]
                        out_image, _ = mask(src, geom, crop=True, all_touched=True)
                        
                        # Sélectionner les bandes spécifiques si demandé
                        if config["selected_bands"]:
                            out_image = np.stack([out_image[i-1] for i in config["selected_bands"]])
                        
                        # Filtrer les pixels valides
                        valid_pixels = []
                        for i in range(out_image.shape[1]):
                            for j in range(out_image.shape[2]):
                                pixel = out_image[:, i, j]
                                if not np.isnan(pixel).any() and not (pixel == 0).all():
                                    valid_pixels.append(pixel)
                        
                        if valid_pixels:
                            class_samples.extend(valid_pixels)
                    except Exception as e:
                        print(f"  Erreur lors du traitement d'un polygone: {e}")
                        continue
        
        # Calculer les statistiques de classe si suffisamment d'échantillons
        if len(class_samples) >= 2:
            class_samples = np.array(class_samples)
            print(f"  Classe {class_id}: {len(class_samples)} échantillons")
            
            # Calcul de la moyenne, de la covariance et de l'écart-type
            mean = np.mean(class_samples, axis=0)
            cov = np.cov(class_samples, rowvar=False)
            std = np.std(class_samples, axis=0)
            
            # Ajouter la régularisation
            cov += np.eye(cov.shape[0]) * class_reg
            
            # Stocker les informations de classe
            classes_info[class_id] = {
                'mean': mean,
                'cov': cov,
                'std': std,
                'samples': len(class_samples),
                'training_data': class_samples  # Stocker les données pour sklearn
            }
        else:
            print(f"  AVERTISSEMENT: Classe {class_id} ignorée - pas assez d'échantillons ({len(class_samples)})")
    
    # Vérifier qu'au moins une classe a été traitée
    if not classes_info:
        raise ValueError("Aucune classe n'a pu être traitée. Vérifiez vos données.")
    
    print(f"Statistiques calculées pour {len(classes_info)} classes")
    return classes_info
