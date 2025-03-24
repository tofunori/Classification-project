"""
ANALYSE DES BANDES RASTER
========================
Ce script analyse les bandes d'un fichier raster et génère un rapport détaillé
des statistiques pour chaque bande dans un fichier texte.
"""

import os
import sys
import numpy as np
import rasterio
from rasterio.plot import show
import matplotlib.pyplot as plt
from datetime import datetime

def analyze_raster_bands(raster_path, output_dir=None):
    """
    Analyse les bandes d'un fichier raster et génère un rapport détaillé.
    
    Args:
        raster_path (str): Chemin vers le fichier raster à analyser
        output_dir (str, optional): Répertoire de sortie pour le rapport
    
    Returns:
        str: Chemin vers le fichier de rapport généré
    """
    # Vérifier si le fichier existe
    if not os.path.exists(raster_path):
        print(f"ERREUR: Le fichier {raster_path} n'existe pas.")
        return None
    
    # Définir le répertoire de sortie
    if output_dir is None:
        output_dir = os.path.dirname(raster_path)
    
    # Créer le répertoire de sortie s'il n'existe pas
    os.makedirs(output_dir, exist_ok=True)
    
    # Définir le nom du fichier de rapport
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    raster_name = os.path.splitext(os.path.basename(raster_path))[0]
    report_path = os.path.join(output_dir, f"rapport_bandes_{raster_name}_{timestamp}.txt")
    
    # Définir les noms des bandes Sentinel-2
    sentinel2_band_names = {
        1: "B2 - Bleu [16U]",
        2: "B3 - Vert [16U]",
        3: "B4 - Rouge [16U]",
        4: "B5 - RedEdge05 [16U]",
        5: "B6 - RedEdge06 [16U]",
        6: "B7 - RedEdge07 [16U]",
        7: "B8 - PIR [16U]",
        8: "B11 - NIR11 [16U]",
        9: "B12 - NIR12 [16U]",
    }
    
    # Ouvrir le fichier raster
    print(f"Analyse du fichier raster: {raster_path}")
    with rasterio.open(raster_path) as src:
        # Récupérer les métadonnées
        profile = src.profile
        bands_count = src.count
        width = src.width
        height = src.height
        crs = src.crs
        transform = src.transform
        
        # Tenter de détecter le type d'image (Sentinel-2, etc.)
        image_type = "Inconnu"
        if "sentinel" in raster_path.lower() or "s2" in raster_path.lower() or "tr_clip" in raster_path.lower():
            image_type = "Sentinel-2"
        
        # Préparer le rapport
        report = []
        report.append("=" * 80)
        report.append(f"RAPPORT D'ANALYSE DES BANDES RASTER - {timestamp}")
        report.append("=" * 80)
        report.append(f"\nFichier analysé: {raster_path}")
        report.append(f"Type d'image détecté: {image_type}")
        report.append(f"Dimensions: {width} x {height} pixels")
        report.append(f"Nombre de bandes: {bands_count}")
        report.append(f"Système de coordonnées: {crs}")
        report.append(f"Type de données: {profile['dtype']}")
        report.append(f"Transformation: {transform}")
        report.append("\n" + "=" * 80)
        report.append("STATISTIQUES PAR BANDE")
        report.append("=" * 80)
        
        # Histogramme des valeurs pour chaque bande
        histograms = []
        
        # Analyser chaque bande
        for band_idx in range(1, bands_count + 1):
            # Déterminer le nom de la bande
            if image_type == "Sentinel-2" and band_idx in sentinel2_band_names:
                band_name = sentinel2_band_names[band_idx]
            else:
                band_name = f"Bande {band_idx}"
            
            print(f"Analyse de {band_name}...")
            band_data = src.read(band_idx)
            
            # Filtrer les valeurs non-data
            mask = band_data != src.nodata if src.nodata is not None else np.ones_like(band_data, dtype=bool)
            valid_data = band_data[mask]
            
            if len(valid_data) == 0:
                report.append(f"\n{band_name}: Aucune donnée valide")
                continue
            
            # Calculer les statistiques
            min_val = np.min(valid_data)
            max_val = np.max(valid_data)
            mean_val = np.mean(valid_data)
            median_val = np.median(valid_data)
            std_val = np.std(valid_data)
            
            # Calculer les percentiles
            p1 = np.percentile(valid_data, 1)
            p5 = np.percentile(valid_data, 5)
            p95 = np.percentile(valid_data, 95)
            p99 = np.percentile(valid_data, 99)
            
            # Calculer l'histogramme
            hist, bin_edges = np.histogram(valid_data, bins=50)
            histograms.append((hist, bin_edges, band_name))
            
            # Ajouter les statistiques au rapport
            report.append(f"\n{band_name}:")
            report.append(f"  Minimum: {min_val:.4f}")
            report.append(f"  Maximum: {max_val:.4f}")
            report.append(f"  Moyenne: {mean_val:.4f}")
            report.append(f"  Médiane: {median_val:.4f}")
            report.append(f"  Écart-type: {std_val:.4f}")
            report.append(f"  Percentile 1%: {p1:.4f}")
            report.append(f"  Percentile 5%: {p5:.4f}")
            report.append(f"  Percentile 95%: {p95:.4f}")
            report.append(f"  Percentile 99%: {p99:.4f}")
            
            # Calculer les valeurs uniques si peu nombreuses
            unique_values = np.unique(valid_data)
            if len(unique_values) < 20:
                report.append(f"  Valeurs uniques ({len(unique_values)}): {unique_values}")
            else:
                report.append(f"  Nombre de valeurs uniques: {len(unique_values)}")
        
        # Ajouter des informations sur les corrélations entre bandes
        if bands_count > 1:
            report.append("\n" + "=" * 80)
            report.append("CORRÉLATIONS ENTRE BANDES")
            report.append("=" * 80)
            
            # Créer une matrice pour stocker les données de toutes les bandes
            all_bands_data = np.zeros((bands_count, np.sum(mask)))
            
            # Remplir la matrice avec les données valides de chaque bande
            for band_idx in range(1, bands_count + 1):
                band_data = src.read(band_idx)[mask]
                all_bands_data[band_idx-1] = band_data
            
            # Calculer la matrice de corrélation
            corr_matrix = np.corrcoef(all_bands_data)
            
            # Ajouter la matrice de corrélation au rapport
            report.append("\nMatrice de corrélation:")
            
            # Créer les en-têtes avec les noms des bandes
            band_labels = []
            for i in range(bands_count):
                if image_type == "Sentinel-2" and i+1 in sentinel2_band_names:
                    # Utiliser uniquement la première partie du nom (ex: "B2" au lieu de "B2 - Bleu (490nm)")
                    short_name = sentinel2_band_names[i+1].split(" - ")[0]
                    band_labels.append(short_name)
                else:
                    band_labels.append(f"B{i+1}")
            
            # Ajouter l'en-tête
            header = "      " + "  ".join([f"{label:5s}" for label in band_labels])
            report.append(header)
            
            # Ajouter les lignes de la matrice
            for i in range(bands_count):
                line = f"{band_labels[i]:5s} "
                for j in range(bands_count):
                    line += f"{corr_matrix[i, j]:5.2f}  "
                report.append(line)
        
        # Générer des visualisations
        print("Génération des visualisations...")
        fig_path = os.path.join(output_dir, f"histogrammes_{raster_name}_{timestamp}.png")
        
        # Créer la figure pour les histogrammes
        n_rows = (bands_count + 2) // 3  # 3 histogrammes par ligne
        fig, axes = plt.subplots(n_rows, 3, figsize=(15, 4 * n_rows))
        axes = axes.flatten() if n_rows > 1 else [axes] if bands_count == 1 else axes
        
        # Tracer les histogrammes
        for i, (hist, bin_edges, title) in enumerate(histograms):
            if i < len(axes):
                axes[i].bar(bin_edges[:-1], hist, width=bin_edges[1] - bin_edges[0], alpha=0.7)
                axes[i].set_title(title)
                axes[i].set_xlabel('Valeur')
                axes[i].set_ylabel('Fréquence')
        
        # Cacher les axes inutilisés
        for i in range(len(histograms), len(axes)):
            axes[i].set_visible(False)
        
        # Ajuster la mise en page et sauvegarder
        plt.tight_layout()
        plt.savefig(fig_path)
        plt.close()
        
        # Ajouter le chemin de la figure au rapport
        report.append("\n" + "=" * 80)
        report.append("VISUALISATIONS")
        report.append("=" * 80)
        report.append(f"\nHistogrammes des bandes: {fig_path}")
        
        # Écrire le rapport dans un fichier
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report))
        
        print(f"Rapport généré: {report_path}")
        return report_path

def main():
    """Point d'entrée principal du script."""
    # Vérifier les arguments
    if len(sys.argv) < 2:
        print("Usage: python analyze_bands.py <chemin_raster> [répertoire_sortie]")
        return
    
    # Récupérer les arguments
    raster_path = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else None
    
    # Analyser le raster
    report_path = analyze_raster_bands(raster_path, output_dir)
    
    if report_path:
        print(f"\nAnalyse terminée avec succès. Rapport disponible à: {report_path}")
    else:
        print("\nL'analyse a échoué.")

if __name__ == "__main__":
    main()
