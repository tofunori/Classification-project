"""
CONFIG MODULE
============
Ce module contient les configurations et paramètres pour le projet de classification.
"""

import os

class Config:
    """Classe de configuration pour le projet de classification."""
    
    def __init__(self, custom_config=None):
        # Paramètres par défaut
        self.config = self.get_default_config()
        
        # Mettre à jour avec la configuration personnalisée si fournie
        if custom_config:
            self.update_config(custom_config)
    
    def get_default_config(self):
        """Définit la configuration par défaut pour la classification."""
        return {
            # Chemins de fichiers - à modifier selon vos besoins
            "raster_path": r"D:\UQTR\Hiver 2025\Télédétection\TP3\tr_clip.tif",
            "shapefile_path": r"D:\UQTR\Hiver 2025\Télédétection\TP3\classes.shp",
            "output_dir": r"D:\UQTR\Hiver 2025\Télédétection\TP3\resultats",
            "output_filename": "classification_mlc_2.tif",
            
            # Paramètres généraux
            "class_column": "Classe",        # Colonne des classes dans le shapefile
            "selected_bands": [2, 3, 4, 5, 6, 7, 8],  # Bandes spécifiques à utiliser
            
            "class_params": {
                1: [1e-5, 0],   # Eau - Suppression du buffer pour éviter le surclassement
                2: [1e-4, 0],   # Forêt - Suppression du buffer négatif pour conserver plus d'échantillons
                3: [3e-4, 0],   # Tourbière - Suppression du buffer négatif pour conserver plus d'échantillons
                4: [5e-4, 0],   # Herbes - Paramètres inchangés
                5: [1e-3, 0],   # Champs - Paramètres inchangés
                6: [1e-2, 0]    # Urbain - Réduction de la régularisation (de 5e-2 à 1e-2)
            },
            
            # Informations pour la carte
            "class_colors": {
                1: "#3288bd",  # Eau - bleu
                2: "#66c164",  # Forêt - vert
                3: "#87CEFA",  # Tourbière - bleu clair
                4: "#ffff00",  # Herbes - vert clair
                5: "#f39c12",  # Champs - orange
                6: "#7f8c8d"   # Urbain - gris
            },
            "class_names": {
                1: "Eau", 
                2: "Forêt", 
                3: "Tourbière",
                4: "Herbes", 
                5: "Champs", 
                6: "Urbain"
            },
            
            # Paramètres de validation
            "validation": {
                "enabled": False,  # Activer/désactiver la validation
                "shapefile_path": r"D:\UQTR\Hiver 2025\Télédétection\TP3\validation.shp"
            },
            "comparison": {
                "enabled": True,  # Activez cette option pour comparer avec une classification de référence
                "raster_path": r"D:\UQTR\Hiver 2025\Télédétection\TP3\raster_polygone_merge.tif"
            }
        }
    
    def update_config(self, custom_config):
        """Met à jour la configuration avec des paramètres personnalisés."""
        for key, value in custom_config.items():
            if key in self.config:
                if isinstance(value, dict) and isinstance(self.config[key], dict):
                    # Mise à jour récursive pour les dictionnaires imbriqués
                    self.config[key].update(value)
                else:
                    self.config[key] = value
            else:
                self.config[key] = value
    
    def __getitem__(self, key):
        """Permet d'accéder aux valeurs de configuration avec la notation de dictionnaire."""
        return self.config[key]
    
    def __setitem__(self, key, value):
        """Permet de définir des valeurs de configuration avec la notation de dictionnaire."""
        self.config[key] = value
    
    def get(self, key, default=None):
        """Obtient une valeur de configuration avec une valeur par défaut en cas d'absence."""
        return self.config.get(key, default)
    
    def keys(self):
        """Renvoie les clés de configuration."""
        return self.config.keys()
    
    def items(self):
        """Renvoie les paires clé-valeur de configuration."""
        return self.config.items()
