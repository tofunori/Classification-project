"""
EXEMPLE D'UTILISATION DU SCRIPT DE VISUALISATION
===============================================
Ce script montre comment utiliser le script visualiser_classification.py
"""

import os
import sys
import argparse

def main():
    parser = argparse.ArgumentParser(description='Exemple de visualisation de classification non supervisée')
    parser.add_argument('--classification', required=True, help='Chemin vers le fichier TIFF de classification')
    parser.add_argument('--donnees_originales', help='Chemin vers le fichier TIFF des données originales')
    parser.add_argument('--output_dir', default='./resultats_visualisation', help='Répertoire de sortie pour les résultats')
    
    args = parser.parse_args()
    
    # Vérifier que les fichiers existent
    if not os.path.isfile(args.classification):
        print(f"ERREUR: Le fichier de classification {args.classification} n'existe pas.")
        sys.exit(1)
    
    if args.donnees_originales and not os.path.isfile(args.donnees_originales):
        print(f"ERREUR: Le fichier de données originales {args.donnees_originales} n'existe pas.")
        sys.exit(1)
    
    # Créer la commande pour exécuter le script de visualisation
    cmd = f"python visualiser_classification.py --classification \"{args.classification}\""
    
    if args.donnees_originales:
        cmd += f" --donnees_originales \"{args.donnees_originales}\""
    
    if args.output_dir:
        cmd += f" --output_dir \"{args.output_dir}\""
    
    # Ajouter quelques bandes par défaut si les données originales sont fournies
    if args.donnees_originales:
        cmd += " --bands 2 3 4 5 6 7 8"  # Bandes courantes pour Sentinel-2
    
    print(f"Exécution de la commande : {cmd}")
    os.system(cmd)
    
    print("\nTraitement terminé.")
    if os.path.exists(args.output_dir):
        print(f"Résultats disponibles dans : {args.output_dir}")
        
        # Lister les fichiers générés
        fichiers = os.listdir(args.output_dir)
        print("\nFichiers générés :")
        for fichier in fichiers:
            print(f" - {fichier}")

if __name__ == "__main__":
    main() 