"""
EVALUATE MODULE
==============
Ce module contient les fonctions pour évaluer la qualité des classifications,
calculer les métriques de précision et comparer différentes classifications.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import rasterio
import geopandas as gpd
from sklearn.metrics import confusion_matrix, accuracy_score, cohen_kappa_score, classification_report

def validate_classification(classification, config):
    """
    Valide la classification en utilisant un shapefile de validation.
    Retourne les métriques de validation.
    
    Args:
        classification (numpy.ndarray): Résultat de la classification
        config (dict): Configuration du processus de validation
        
    Returns:
        dict: Métriques de validation ou None si la validation est désactivée
    """
    if not config["validation"]["enabled"]:
        print("\nValidation désactivée dans la configuration")
        return None
    
    print("\nValidation de la classification...")
    validation_shapefile = config["validation"]["shapefile_path"]
    
    # Utiliser la colonne de classe spécifiée dans la validation s'il y en a une, sinon utiliser la colonne par défaut
    class_column = config["validation"].get("class_column", config["class_column"])
    
    output_dir = config["output_dir"]
    
    # Charger le shapefile de validation
    validation_data = gpd.read_file(validation_shapefile)
    
    # Initialiser les listes pour stocker les données de référence et prédites
    y_true = []
    y_pred = []
    
    # Pour chaque polygone de validation
    for _, row in validation_data.iterrows():
        # Ignorer les entrées avec des valeurs de classe nulles ou NaN
        if pd.isna(row[class_column]):
            continue
            
        class_id = row[class_column]
        geom = row.geometry
        
        # Convertir la géométrie en coordonnées pixels
        with rasterio.open(config["raster_path"]) as src:
            geom_mask = rasterio.features.geometry_mask(
                [geom.__geo_interface__], 
                out_shape=(src.height, src.width),
                transform=src.transform, 
                invert=True
            )
            
            # Obtenir les pixels dans ce polygone
            predicted_classes = classification[geom_mask]
            
            # Ajouter à nos listes
            if len(predicted_classes) > 0:
                for pred_class in predicted_classes:
                    y_true.append(class_id)
                    y_pred.append(pred_class)
    
    # Calculer les métriques seulement si nous avons des données
    if len(y_true) > 0:
        # Calculer la matrice de confusion
        cm = confusion_matrix(y_true, y_pred)
        
        # Autres métriques
        accuracy = accuracy_score(y_true, y_pred)
        kappa = cohen_kappa_score(y_true, y_pred)
        report = classification_report(y_true, y_pred, output_dict=True)
        
        print(f"Précision globale: {accuracy:.2f}")
        print(f"Coefficient Kappa: {kappa:.2f}")
        
        # Générer la visualisation du rapport de classification
        report_data = []
        for cls, metrics in report.items():
            if cls not in ['accuracy', 'macro avg', 'weighted avg']:
                cls_name = config["class_names"].get(int(cls), f"Classe {cls}")
                report_data.append([
                    cls_name,
                    f"{metrics['precision']:.2f}",
                    f"{metrics['recall']:.2f}",
                    f"{metrics['f1-score']:.2f}",
                    f"{metrics['support']}"
                ])
        
        # Ajouter les moyennes
        for avg_type in ['macro avg', 'weighted avg']:
            report_data.append([
                avg_type,
                f"{report[avg_type]['precision']:.2f}",
                f"{report[avg_type]['recall']:.2f}",
                f"{report[avg_type]['f1-score']:.2f}",
                f"{report[avg_type]['support']}"
            ])
        
        # Créer le tableau du rapport
        fig, ax = plt.figure(figsize=(10, 6), dpi=150), plt.subplot(111)
        ax.axis('tight')
        ax.axis('off')
        table = ax.table(
            cellText=report_data,
            colLabels=['Classe', 'Précision', 'Rappel', 'F1-score', 'Support'],
            cellLoc='center',
            loc='center',
            colWidths=[0.3, 0.15, 0.15, 0.15, 0.15]
        )
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.5)
        
        # Ajuster l'apparence
        for (i, j), cell in table.get_celld().items():
            if i == 0:  # Header
                cell.set_text_props(weight='bold', color='white')
                cell.set_facecolor('#4472C4')
            elif i > len(report_data) - 3:  # moyennes
                cell.set_facecolor('#B4C7E7')
            elif i % 2 == 1:  # Odd rows
                cell.set_facecolor('#D9E1F2')
            else:  # Even rows
                cell.set_facecolor('#E9EDF4')
        
        plt.title(f'Rapport de Classification\nPrécision: {accuracy:.2f}, Kappa: {kappa:.2f}', fontsize=14, pad=20)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "rapport_classification.png"), bbox_inches='tight')
        plt.close()
        
        # Générer un graphique de la matrice de confusion
        plt.figure(figsize=(10, 8))
        class_names = [config["class_names"].get(cls, f"Classe {cls}") for cls in np.unique(y_true)]
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=class_names, 
                    yticklabels=class_names)
        plt.title(f'Matrice de Confusion\nPrécision: {accuracy:.2f}, Kappa: {kappa:.2f}')
        plt.ylabel('Classe réelle')
        plt.xlabel('Classe prédite')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "matrice_confusion.png"), dpi=300)
        plt.close()
        
        print("Validation terminée et résultats sauvegardés")
        return {
            'accuracy': accuracy,
            'kappa': kappa,
            'confusion_matrix': cm,
            'report': report
        }
    else:
        print("Pas assez de données pour la validation")
        return None

def compare_classifications(classification, config):
    """
    Compare la classification générée avec une classification raster de référence.
    Calcule la matrice de confusion, la précision, le coefficient kappa et le rapport de classification.
    
    Args:
        classification (numpy.ndarray): Classification à évaluer
        config (dict): Configuration contenant les paramètres de comparaison
        
    Returns:
        dict: Métriques de comparaison ou None en cas d'erreur
    """
    if not config["comparison"]["enabled"]:
        print("\nComparaison désactivée dans la configuration")
        return None
        
    print("\nComparaison avec la classification de référence...")
    ref_raster_path = config["comparison"]["raster_path"]
    output_dir = config["output_dir"]
    
    try:
        with rasterio.open(ref_raster_path) as src:
            ref_classification = src.read(1)
            if ref_classification.shape != classification.shape:
                print("Les dimensions du raster de référence ne correspondent pas à la classification générée.")
                return None
    except Exception as e:
        print(f"Erreur lors du chargement du raster de référence: {e}")
        return None
    
    # Aplatir les tableaux pour une comparaison pixel par pixel
    y_true = ref_classification.flatten()
    y_pred = classification.flatten()
    
    # Exclure les pixels non classifiés (valeur 0)
    valid_mask = (y_true > 0) & (y_pred > 0)
    y_true = y_true[valid_mask]
    y_pred = y_pred[valid_mask]
    
    if len(y_true) == 0:
        print("ERREUR: Aucun pixel valide pour la comparaison")
        return None
    
    cm = confusion_matrix(y_true, y_pred)
    accuracy = accuracy_score(y_true, y_pred)
    kappa = cohen_kappa_score(y_true, y_pred)
    report = classification_report(y_true, y_pred, output_dict=True)
    
    print(f"Comparaison - Précision globale: {accuracy:.2f}")
    print(f"Comparaison - Coefficient Kappa: {kappa:.2f}")
    
    # Normaliser la matrice de confusion pour obtenir des pourcentages par ligne
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    
    # Générer un graphique de la matrice de confusion en valeurs absolues
    plt.figure(figsize=(10, 8))
    class_names = [config["class_names"].get(cls, f"Classe {cls}") for cls in np.unique(y_true)]
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, 
                yticklabels=class_names)
    plt.title(f'Matrice de Confusion (Comparaison) - Valeurs absolues\nPrécision: {accuracy:.2f}, Kappa: {kappa:.2f}')
    plt.ylabel('Classe référence')
    plt.xlabel('Classe prédite')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "matrice_confusion_comparaison.png"), dpi=300)
    plt.close()
    
    # Générer un graphique de la matrice de confusion en pourcentages
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_percent, annot=True, fmt='.1f', cmap='Blues', 
                xticklabels=class_names, 
                yticklabels=class_names)
    plt.title(f'Matrice de Confusion (Comparaison) - Pourcentages (%)\nPrécision: {accuracy:.2f}, Kappa: {kappa:.2f}')
    plt.ylabel('Classe référence')
    plt.xlabel('Classe prédite')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "matrice_confusion_comparaison_pourcent.png"), dpi=300)
    plt.close()
    
    print("Comparaison terminée et résultats sauvegardés")
    
    return {
        'accuracy': accuracy,
        'kappa': kappa,
        'confusion_matrix': cm,
        'confusion_matrix_percent': cm_percent,
        'report': report
    }

def calculate_class_statistics(classification, classes_info, config):
    """
    Calcule et génère des statistiques détaillées sur la classification.
    
    Args:
        classification (numpy.ndarray): Résultat de la classification
        classes_info (dict): Informations sur les classes
        config (dict): Configuration
        
    Returns:
        pandas.DataFrame: DataFrame contenant les statistiques de classe
    """
    print("\nCalcul des statistiques de classification...")
    
    output_dir = config["output_dir"]
    
    # Statistiques de distribution des pixels par classe
    unique_classes, counts = np.unique(classification, return_counts=True)
    
    # Filtrer pour exclure les pixels non classifiés (valeur 0)
    class_mask = unique_classes > 0
    unique_classes = unique_classes[class_mask]
    counts = counts[class_mask]
    
    # Calculer les pourcentages
    total_pixels = np.sum(counts)
    percentages = (counts / total_pixels) * 100
    
    # Créer un DataFrame pour les statistiques de classe
    stats_df = pd.DataFrame({
        'Classe': unique_classes,
        'Nom': [config["class_names"].get(cls, f"Classe {cls}") for cls in unique_classes],
        'Nombre de pixels': counts,
        'Pourcentage (%)': percentages,
        'Échantillons utilisés': [classes_info.get(cls, {}).get('samples', 0) for cls in unique_classes]
    })
    
    # Créer un tableau visuel des statistiques
    fig, ax = plt.figure(figsize=(10, 5), dpi=150), plt.subplot(111)
    ax.axis('tight')
    ax.axis('off')
    
    # Formatter les valeurs numériques correctement
    formatted_values = []
    for row in stats_df.values:
        formatted_row = []
        for i, val in enumerate(row):
            # Vérifier le type de colonne pour appliquer le formatage approprié
            if i in [0]:  # Colonnes pour classes (entiers)
                formatted_row.append(str(val))
            elif i in [1]:  # Colonnes pour noms (texte)
                formatted_row.append(val)
            elif i in [2, 4]:  # Colonnes pour nombres entiers (pixels, échantillons)
                formatted_row.append(f"{val:,}")
            elif i in [3]:  # Colonnes pour pourcentages
                formatted_row.append(f"{val:.2f}")
        formatted_values.append(formatted_row)
    
    # Vérifier si des classes valides ont été trouvées
    if not formatted_values:
        print("AVERTISSEMENT: Aucune classe valide n'a été trouvée dans le résultat de classification.")
        # Retourner un DataFrame vide
        return pd.DataFrame(columns=['Classe', 'Nom', 'Nombre de pixels', 'Pourcentage (%)', 'Échantillons utilisés'])
    
    table = ax.table(
        cellText=formatted_values,
        colLabels=stats_df.columns,
        cellLoc='center',
        loc='center',
        colWidths=[0.15, 0.25, 0.2, 0.2, 0.2]
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)
    
    # Ajuster l'apparence
    for (i, j), cell in table.get_celld().items():
        if i == 0:  # Header
            cell.set_text_props(weight='bold', color='white')
            cell.set_facecolor('#4472C4')
        elif i % 2 == 1:  # Odd rows
            cell.set_facecolor('#D9E1F2')
        else:  # Even rows
            cell.set_facecolor('#E9EDF4')
    
    plt.title('Statistiques de Classification par Classe', fontsize=14, pad=20)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "statistiques_classes.png"), bbox_inches='tight')
    plt.close()
    
    # Générer un graphique de distribution des classes
    plt.figure(figsize=(12, 6))
    
    # Graphique en barres avec pourcentages
    bars = plt.bar(
        [config["class_names"].get(cls, f"Classe {cls}") for cls in unique_classes],
        percentages,
        color=[config["class_colors"].get(cls, f"C{cls%10}") for cls in unique_classes]
    )
    
    plt.title('Distribution des classes dans l\'image classifiée', fontsize=14)
    plt.ylabel('Pourcentage (%)', fontsize=12)
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Ajouter les valeurs sur les barres
    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width()/2.,
            height + 0.5,
            f'{height:.1f}%',
            ha='center', va='bottom',
            fontsize=10
        )
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "distribution_classes.png"), dpi=300)
    plt.close()
    
    print("Statistiques de classification calculées et sauvegardées")
    return stats_df
