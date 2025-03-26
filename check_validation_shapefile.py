"""
Script pour vérifier les colonnes du fichier shapefile de validation
"""
import geopandas as gpd
import os

def main():
    # Chemin du fichier shapefile
    shapefile_path = r"D:\UQTR\Hiver 2025\Télédétection\TP3\points_validation.shp"
    
    # Vérifier si le fichier existe
    if not os.path.exists(shapefile_path):
        print(f"Le fichier {shapefile_path} n'existe pas!")
        return
    
    try:
        # Lire le shapefile
        shapefile = gpd.read_file(shapefile_path)
        
        # Afficher les colonnes
        print("Colonnes disponibles:")
        for col in shapefile.columns:
            print(f"  - {col}")
        
        # Afficher le nombre de points
        print(f"\nNombre de points: {len(shapefile)}")
        
        # Afficher les premières lignes
        print("\nAperçu des données:")
        print(shapefile.head())
        
        # Si la colonne classvalue existe, afficher ses valeurs uniques
        if 'classvalue' in shapefile.columns:
            unique_values = shapefile['classvalue'].unique()
            print("\nValeurs uniques dans la colonne 'classvalue':")
            print(unique_values)
        else:
            print("\nLa colonne 'classvalue' n'existe pas!")
            
            # Rechercher d'autres colonnes potentielles de classe
            potential_class_cols = [col for col in shapefile.columns if 'class' in col.lower() or 'value' in col.lower()]
            if potential_class_cols:
                print("\nColonnes potentielles contenant des classes:")
                for col in potential_class_cols:
                    unique_values = shapefile[col].unique()
                    print(f"  - {col}: {unique_values}")
    
    except Exception as e:
        print(f"Erreur lors de la lecture du shapefile: {str(e)}")

if __name__ == "__main__":
    main() 