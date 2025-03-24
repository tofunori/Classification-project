"""
Script pour examiner le contenu du shapefile
"""

import geopandas as gpd

def check_shapefile():
    shapefile_path = r"D:\UQTR\Hiver 2025\Télédétection\TP3\classes.shp"
    try:
        print(f"Chargement du shapefile: {shapefile_path}")
        gdf = gpd.read_file(shapefile_path)
        
        # Afficher les informations de base
        print(f"\nInformations du shapefile:")
        print(f"Nombre d'entités: {len(gdf)}")
        print(f"Système de coordonnées: {gdf.crs}")
        
        # Afficher les colonnes disponibles
        print(f"\nColonnes disponibles:")
        for col in gdf.columns:
            print(f"  - {col}")
        
        # Afficher un échantillon des valeurs
        print(f"\nPremière ligne du shapefile:")
        print(gdf.iloc[0])
        
        # Si une colonne ressemble à la colonne de classe, afficher les valeurs uniques
        potential_class_columns = []
        for col in gdf.columns:
            if col.upper() in ['CLASS', 'CLASSE', 'TYPE', 'CATEGORY', 'CATEGORIE', 'CODE', 'ID', 'VALUE']:
                potential_class_columns.append(col)
            # Chercher aussi des colonnes contenant ces mots-clés
            elif any(keyword in col.upper() for keyword in ['CLASS', 'CLASSE', 'TYPE', 'CATEGORY', 'CATEGORIE']):
                potential_class_columns.append(col)
        
        # Afficher les valeurs uniques pour chaque colonne potentielle de classe
        if potential_class_columns:
            print("\nValeurs uniques dans les colonnes potentielles de classe:")
            for col in potential_class_columns:
                unique_values = gdf[col].unique()
                print(f"  - {col}: {unique_values}")
        else:
            print("\nAucune colonne potentielle de classe n'a été identifiée.")
    
    except Exception as e:
        print(f"Erreur lors de l'examen du shapefile: {e}")

if __name__ == "__main__":
    check_shapefile()
