"""
VISUALIZE MODULE
===============
Ce module contient les fonctions pour visualiser les données, les résultats de classification
et les statistiques associées.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap, BoundaryNorm
import seaborn as sns
import pandas as pd

def visualize_spectral_signatures(classes_info, config):
    """
    Génère des visualisations des signatures spectrales des classes.
    
    Args:
        classes_info (dict): Informations sur les classes
        config (dict): Configuration contenant les paramètres de visualisation
    """
    print("\nGénération des visualisations des signatures spectrales...")
    
    output_dir = config["output_dir"]
    
    # Graphique des signatures spectrales
    plt.figure(figsize=(12, 8))
    
    for class_id, info in classes_info.items():
        class_name = config["class_names"].get(class_id, f"Classe {class_id}")
        color = config["class_colors"].get(class_id, f"C{class_id%10}")
        
        plt.errorbar(
            config["selected_bands"], 
            info['mean'], 
            yerr=info['std'], 
            fmt='o-', 
            label=class_name,
            color=color,
            capsize=5,
            linewidth=2,
            markersize=8
        )
    
    plt.title('Signatures Spectrales des Classes', fontsize=16)
    plt.xlabel('Numéro de bande', fontsize=14)
    plt.ylabel('Réflectance', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "signatures_spectrales.png"), dpi=300)
    plt.close()
    
    # Tableau des statistiques spectrales
    spectral_stats = {}
    for cls_id, info in classes_info.items():
        spectral_stats[cls_id] = {
            'class_name': config["class_names"].get(cls_id, f"Classe {cls_id}"),
            'mean': info['mean'],
            'std': info['std']
        }
    
    # Créer un tableau pour toutes les bandes
    headers = ['Classe']
    for band in config["selected_bands"]:
        headers.extend([f'Moyenne B{band}', f'Écart-type B{band}'])
    
    table_data = []
    for cls_id in sorted(spectral_stats.keys()):
        row = [spectral_stats[cls_id]['class_name']]
        for i, _ in enumerate(config["selected_bands"]):
            row.extend([
                f"{spectral_stats[cls_id]['mean'][i]:.2f}",
                f"{spectral_stats[cls_id]['std'][i]:.2f}"
            ])
        table_data.append(row)
    
    # Créer le tableau avec statistiques spectrales
    fig, ax = plt.figure(figsize=(12, len(table_data)*0.5 + 2), dpi=150), plt.subplot(111)
    ax.axis('tight')
    ax.axis('off')
    table = ax.table(
        cellText=table_data,
        colLabels=headers,
        cellLoc='center',
        loc='center'
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
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
    
    plt.title('Statistiques Spectrales par Classe et par Bande', fontsize=14, pad=20)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "statistiques_spectrales.png"), bbox_inches='tight')
    plt.close()

def create_scatterplots(classes_info, config):
    """
    Génère des scatterplots pour visualiser la séparation des classes dans l'espace spectral.
    Tous les graphiques sont placés dans un même layout pour faciliter la comparaison.
    
    Args:
        classes_info (dict): Informations sur les classes
        config (dict): Configuration contenant les paramètres de visualisation
        
    Returns:
        bool: True si réussi, False sinon
    """
    print("\nGénération des scatterplots pour l'analyse de la séparation des classes...")
    
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    from matplotlib.lines import Line2D
    import numpy as np
    import os
    
    # Utiliser le chemin spécifié pour les résultats
    output_dir = config["output_dir"]
    selected_bands = config["selected_bands"]
    
    # Créer un sous-répertoire pour les scatterplots
    scatterplot_dir = os.path.join(output_dir, "scatterplots")
    os.makedirs(scatterplot_dir, exist_ok=True)
    
    # Vérifier qu'il y a au moins 2 bandes sélectionnées
    if len(selected_bands) < 2:
        print("  AVERTISSEMENT: Pas assez de bandes pour créer des scatterplots")
        return False
    
    # Créer manuellement des paires d'indices de bandes à visualiser
    band_pairs = []
    
    # Limiter à un nombre raisonnable de paires
    if len(selected_bands) >= 2: band_pairs.append((0, 1))
    if len(selected_bands) >= 3: band_pairs.append((0, 2))
    if len(selected_bands) >= 4: band_pairs.append((1, 3))
    if len(selected_bands) >= 5: band_pairs.append((2, 4))
    if len(selected_bands) >= 6: band_pairs.append((3, 5))
    if len(selected_bands) >= 7: band_pairs.append((4, 6))
    if len(selected_bands) >= 8: band_pairs.append((0, 7))
    
    # Ajouter quelques paires supplémentaires si nécessaire
    if len(selected_bands) >= 6: band_pairs.append((1, 5))
    if len(selected_bands) >= 7: band_pairs.append((2, 6))
    
    print(f"  {len(band_pairs)} combinaisons de bandes à visualiser")
    
    # Calculer le nombre de lignes et colonnes pour le layout
    n_plots = len(band_pairs)
    n_cols = min(3, n_plots)  # Maximum 3 colonnes
    n_rows = (n_plots + n_cols - 1) // n_cols  # Arrondi supérieur
    
    # Créer une figure avec des sous-graphiques
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 5*n_rows))
    
    # S'assurer que axes est toujours un tableau 2D même si n_rows ou n_cols = 1
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)
    
    # Préparer une légende commune
    legend_elements = []
    
    # Pour chaque classe, créer un élément de légende
    for class_id, info in classes_info.items():
        class_name = config["class_names"].get(class_id, f"Classe {class_id}")
        color = config["class_colors"].get(class_id, f"C{class_id%10}")
        
        # Ajouter un élément à la légende commune
        legend_elements.append(
            Line2D([0], [0], marker='o', color='w', markerfacecolor=color, 
                   markersize=10, label=class_name)
        )
    
    # Pour chaque paire de bandes, créer un scatterplot dans son propre sous-graphique
    for i, band_pair in enumerate(band_pairs):
        row_idx = i // n_cols
        col_idx = i % n_cols
        ax = axes[row_idx, col_idx]
        
        # Extraire les indices de bandes
        band_idx1, band_idx2 = band_pair
        
        # Obtenir les numéros réels des bandes
        band1 = selected_bands[band_idx1]
        band2 = selected_bands[band_idx2]
        
        # Pour chaque classe, tracer les échantillons
        for class_id, info in classes_info.items():
            color = config["class_colors"].get(class_id, f"C{class_id%10}")
            
            # Extraire les données pour les deux bandes sélectionnées
            data = info['training_data']
            x = data[:, band_idx1]
            y = data[:, band_idx2]
            
            # Tracer les échantillons
            ax.scatter(x, y, c=color, alpha=0.5, edgecolors='none')
            
            # Calculer et tracer l'ellipse de confiance (intervalle de confiance à 95%)
            if len(data) > 2:  # Au moins 3 points pour calculer l'ellipse
                from scipy.stats import chi2
                
                # Extraire la moyenne et la covariance pour ces deux bandes
                mean = info['mean'][[band_idx1, band_idx2]]
                cov = np.array([[info['cov'][band_idx1, band_idx1], info['cov'][band_idx1, band_idx2]],
                               [info['cov'][band_idx2, band_idx1], info['cov'][band_idx2, band_idx2]]])
                
                # Calculer les valeurs propres et vecteurs propres
                eigenvals, eigenvecs = np.linalg.eigh(cov)
                
                # Calculer l'angle de l'ellipse
                angle = np.degrees(np.arctan2(eigenvecs[1, 0], eigenvecs[0, 0]))
                
                # Calculer la largeur et la hauteur de l'ellipse (intervalle de confiance à 95%)
                chi2_val = chi2.ppf(0.95, 2)  # 2 degrés de liberté pour un ellipse 2D
                width = 2 * np.sqrt(chi2_val * eigenvals[0])
                height = 2 * np.sqrt(chi2_val * eigenvals[1])
                
                # Créer et ajouter l'ellipse
                ellipse = patches.Ellipse(xy=(mean[0], mean[1]), 
                                         width=width, 
                                         height=height, 
                                         angle=angle, 
                                         fill=False, 
                                         edgecolor=color, 
                                         linewidth=2)
                ax.add_patch(ellipse)
        
        # Définir les labels et le titre pour ce sous-graphique
        ax.set_xlabel(f'Bande {band1}', fontsize=12)
        ax.set_ylabel(f'Bande {band2}', fontsize=12)
        ax.set_title(f'Bandes {band1} vs {band2}', fontsize=14)
        ax.grid(True, linestyle='--', alpha=0.7)
    
    # Masquer les axes non utilisés
    for i in range(len(band_pairs), n_rows * n_cols):
        row_idx = i // n_cols
        col_idx = i % n_cols
        axes[row_idx, col_idx].axis('off')
    
    # Ajouter une légende commune au bas de la figure
    fig.legend(handles=legend_elements, 
              loc='lower center', 
              bbox_to_anchor=(0.5, 0.02),
              ncol=min(6, len(legend_elements)),
              fontsize=12,
              title="Classes",
              title_fontsize=14,
              frameon=True)
    
    # Ajuster le layout et ajouter un titre global
    plt.suptitle('Analyse de la séparation des classes par paires de bandes', fontsize=16, y=0.98)
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])  # Laisser de l'espace pour la légende
    
    # Sauvegarder le scatterplot combiné
    output_path = os.path.join(scatterplot_dir, 'scatterplots_combinés.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  Scatterplots combinés sauvegardés dans: {output_path}")
    
    # Créer également une matrice de scatterplots avec seaborn si peu de bandes
    if len(selected_bands) <= 5:
        try:
            import seaborn as sns
            import pandas as pd
            
            # Créer un DataFrame avec toutes les données d'entraînement
            all_data = []
            all_classes = []
            
            for class_id, info in classes_info.items():
                data = info['training_data']
                class_name = config["class_names"].get(class_id, f"Classe {class_id}")
                
                for sample in data:
                    sample_dict = {f'Bande {selected_bands[i]}': sample[i] for i in range(len(selected_bands))}
                    sample_dict['Classe'] = class_name
                    all_data.append(sample_dict)
                    all_classes.append(class_id)
            
            df = pd.DataFrame(all_data)
            
            # Définir la palette de couleurs
            palette = {config["class_names"].get(cls, f"Classe {cls}"): config["class_colors"].get(cls, f"C{cls%10}") 
                      for cls in set(all_classes)}
            
            # Créer la matrice de scatterplots
            g = sns.pairplot(
                df, 
                hue='Classe', 
                palette=palette,
                plot_kws={'alpha': 0.5, 's': 15, 'edgecolor': 'none'},
                diag_kind='kde',
                corner=True  # Pour réduire la redondance
            )
            
            g.fig.suptitle('Matrice de Scatterplots pour Toutes les Bandes', fontsize=16, y=1.02)
            
            # Sauvegarder la matrice
            output_path = os.path.join(scatterplot_dir, 'scatterplot_matrix.png')
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"  Matrice de scatterplots sauvegardée dans: {output_path}")
        except Exception as e:
            print(f"  Erreur lors de la création de la matrice de scatterplots: {e}")
    
    return True

def create_pca_scatterplots(classes_info, config, band_weights=None):
    """
    Génère des scatterplots avec analyse en composantes principales (PCA)
    pour visualiser la séparation des classes dans un espace réduit.
    
    Args:
        classes_info (dict): Informations sur les classes
        config (dict): Configuration contenant les paramètres de visualisation
        band_weights (numpy.ndarray, optional): Pondérations des bandes à appliquer avant PCA
        
    Returns:
        bool: True si réussi, False sinon
    """
    print("\nGénération des scatterplots combinés avec PCA...")
    
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    import matplotlib.gridspec as gridspec
    from matplotlib.lines import Line2D
    import numpy as np
    import os
    
    try:
        from sklearn.decomposition import PCA
    except ImportError:
        print("  ERREUR: scikit-learn n'est pas installé. Impossible de créer les scatterplots PCA.")
        return False
    
    # Utiliser le chemin spécifié pour les résultats
    output_dir = config["output_dir"]
    scatterplot_dir = os.path.join(output_dir, "scatterplots")
    os.makedirs(scatterplot_dir, exist_ok=True)
    
    try:
        # Collecter toutes les données d'entraînement
        all_data = []
        all_labels = []
        
        for class_id, info in classes_info.items():
            all_data.append(info['training_data'])
            all_labels.extend([class_id] * len(info['training_data']))
        
        # Concaténer toutes les données
        all_data = np.vstack(all_data)
        all_labels = np.array(all_labels)
        
        # Appliquer les pondérations si fournies
        if band_weights is not None:
            print("  Application des pondérations aux données avant PCA...")
            
            # Vérifier la compatibilité des dimensions
            if all_data.shape[1] != len(band_weights):
                print(f"  AVERTISSEMENT: Dimensions incompatibles - données ({all_data.shape[1]} bandes) vs poids ({len(band_weights)} valeurs)")
                
                # Ajuster les poids si nécessaire
                if len(band_weights) > all_data.shape[1]:
                    print("  Troncation des poids excédentaires...")
                    band_weights = band_weights[:all_data.shape[1]]
                else:
                    print("  Extension des poids avec des valeurs = 1...")
                    extended_weights = np.ones(all_data.shape[1])
                    extended_weights[:len(band_weights)] = band_weights
                    band_weights = extended_weights
            
            # Afficher les poids utilisés
            print("  Poids appliqués pour la PCA:")
            for i, weight in enumerate(band_weights):
                print(f"    Bande {i+1}: {weight:.2f}")
            
            # Appliquer les pondérations
            weighted_data = np.zeros_like(all_data)
            for i in range(all_data.shape[1]):
                weighted_data[:, i] = all_data[:, i] * band_weights[i]
                
            # Utiliser les données pondérées pour la PCA
            all_data = weighted_data
        
        # Appliquer PCA pour réduire à 3 dimensions maximum
        n_components = min(3, all_data.shape[1])
        pca = PCA(n_components=n_components)
        pca_result = pca.fit_transform(all_data)
        
        # Afficher la variance expliquée
        explained_variance = pca.explained_variance_ratio_
        print(f"  Variance expliquée par les composantes principales: {explained_variance * 100}")
        
        # Créer la figure avec GridSpec pour une mise en page flexible
        fig = plt.figure(figsize=(18, 16))
        
        # Définir la mise en page: 2x2 avec fusion des cellules du bas
        gs = gridspec.GridSpec(2, 2, height_ratios=[1.2, 1])
        
        # 1. Scatterplot 2D (PC1 vs PC2) en haut à gauche
        ax1 = fig.add_subplot(gs[0, 0])
        
        # Préparer des éléments de légende commune
        legend_elements = []
        
        # Tracer les points pour chaque classe
        for class_id in np.unique(all_labels):
            class_mask = all_labels == class_id
            class_name = config["class_names"].get(class_id, f"Classe {class_id}")
            color = config["class_colors"].get(class_id, f"C{class_id%10}")
            
            ax1.scatter(
                pca_result[class_mask, 0], 
                pca_result[class_mask, 1], 
                c=color, 
                alpha=0.6,
                edgecolors='none'
            )
            
            # Ajouter un élément à la légende commune
            legend_elements.append(
                Line2D([0], [0], marker='o', color='w', markerfacecolor=color, 
                       markersize=10, label=class_name)
            )
            
            # Ajouter des ellipses de confiance pour chaque classe
            class_data = pca_result[class_mask, :2]
            
            if len(class_data) > 2:
                from scipy.stats import chi2
                
                # Calculer la moyenne et la covariance
                mean = np.mean(class_data, axis=0)
                cov = np.cov(class_data, rowvar=False)
                
                # Calculer les valeurs propres et vecteurs propres
                eigenvals, eigenvecs = np.linalg.eigh(cov)
                
                # Calculer l'angle de l'ellipse
                angle = np.degrees(np.arctan2(eigenvecs[1, 0], eigenvecs[0, 0]))
                
                # Calculer la largeur et la hauteur de l'ellipse (intervalle de confiance à 95%)
                chi2_val = chi2.ppf(0.95, 2)
                width = 2 * np.sqrt(chi2_val * eigenvals[0])
                height = 2 * np.sqrt(chi2_val * eigenvals[1])
                
                # Créer et ajouter l'ellipse
                ellipse = patches.Ellipse(xy=(mean[0], mean[1]), 
                                         width=width, 
                                         height=height, 
                                         angle=angle, 
                                         fill=False, 
                                         edgecolor=color, 
                                         linewidth=2)
                ax1.add_patch(ellipse)
        
        ax1.set_xlabel(f'Composante Principale 1 ({explained_variance[0]*100:.1f}%)', fontsize=14)
        ax1.set_ylabel(f'Composante Principale 2 ({explained_variance[1]*100:.1f}%)', fontsize=14)
        ax1.set_title('Projection 2D des classes (PC1 vs PC2)', fontsize=16)
        ax1.grid(True, linestyle='--', alpha=0.7)
        
        # 2. Graphique 3D en haut à droite (si nous avons 3 composantes)
        if pca_result.shape[1] >= 3:
            from mpl_toolkits.mplot3d import Axes3D
            
            ax2 = fig.add_subplot(gs[0, 1], projection='3d')
            
            for class_id in np.unique(all_labels):
                class_mask = all_labels == class_id
                class_name = config["class_names"].get(class_id, f"Classe {class_id}")
                color = config["class_colors"].get(class_id, f"C{class_id%10}")
                
                ax2.scatter(
                    pca_result[class_mask, 0], 
                    pca_result[class_mask, 1], 
                    pca_result[class_mask, 2],
                    c=color, 
                    alpha=0.6,
                    s=30
                )
            
            ax2.set_xlabel(f'PC1 ({explained_variance[0]*100:.1f}%)', fontsize=14)
            ax2.set_ylabel(f'PC2 ({explained_variance[1]*100:.1f}%)', fontsize=14)
            ax2.set_zlabel(f'PC3 ({explained_variance[2]*100:.1f}%)', fontsize=14)
            ax2.set_title('Projection 3D des classes', fontsize=16)
            
            # Ajuster la vue 3D pour une meilleure visualisation
            ax2.view_init(30, 45)
        
        # 3. Matrice des coefficients en bas (fusionnant les deux cellules)
        ax3 = fig.add_subplot(gs[1, :])
        
        components = pca.components_
        columns = [f'Bande {b}' for b in config["selected_bands"]]
        
        import seaborn as sns
        
        sns.heatmap(
            components, 
            annot=True, 
            cmap='coolwarm', 
            xticklabels=columns,
            yticklabels=[f'PC{i+1} ({var*100:.1f}%)' for i, var in enumerate(explained_variance)],
            ax=ax3
        )
        
        ax3.set_title('Coefficients des Composantes Principales', fontsize=16)
        
        # Ajouter la légende commune en bas de la figure
        fig.legend(handles=legend_elements, 
                  loc='lower center', 
                  bbox_to_anchor=(0.5, 0.02),
                  ncol=min(6, len(legend_elements)),
                  fontsize=12,
                  title="Classes",
                  title_fontsize=14,
                  frameon=True)
        
        # Ajouter un titre global
        plt.suptitle('Analyse en Composantes Principales (PCA) des Classes', fontsize=20, y=0.98)
        
        # Ajuster le layout
        plt.tight_layout(rect=[0, 0.05, 1, 0.95])  # Laisser de l'espace pour le titre et la légende
        
        # Sauvegarder la figure combinée
        output_path = os.path.join(scatterplot_dir, 'scatterplot_pca_combiné.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  Scatterplot PCA combiné sauvegardé dans: {output_path}")
        return True
    
    except Exception as e:
        print(f"  ERREUR lors de la création des scatterplots PCA: {e}")
        import traceback
        traceback.print_exc()
        return False

def generate_classification_map(classification, config):
    """
    Génère une carte de classification avec légende.
    
    Args:
        classification (numpy.ndarray): Résultat de la classification
        config (dict): Configuration contenant les paramètres de visualisation
        
    Returns:
        bool: True si réussi, False sinon
    """
    print("\nGénération de la carte de classification...")
    
    output_dir = config["output_dir"]
    map_output = os.path.join(output_dir, "carte_classification.png")
    
    # Supprimer le fichier existant s'il existe
    if os.path.exists(map_output):
        os.remove(map_output)
    
    # Obtenir les classes uniques dans le résultat
    unique_classes = np.unique(classification)
    unique_classes = unique_classes[unique_classes > 0]  # Ignorer la classe 0 (non classé)
    
    if len(unique_classes) > 0:
        # Préparer les couleurs pour la colormap
        cmap_colors = []
        for cls in range(max(unique_classes) + 1):
            if cls == 0:  # Classe non classifiée (noir)
                cmap_colors.append('#000000')
            else:
                cmap_colors.append(config["class_colors"].get(cls, f"C{cls%10}"))
        
        # Créer la colormap personnalisée
        cmap = ListedColormap(cmap_colors)
        
        # Créer une figure
        plt.figure(figsize=(12, 8), dpi=300)
        
        # Masquer les zones non classifiées (classe 0)
        masked_result = np.ma.masked_where(classification == 0, classification)
        
        # Définir les limites pour la colormap
        bounds = np.arange(0, max(unique_classes) + 2) - 0.5
        norm = BoundaryNorm(bounds, cmap.N)
        
        # Afficher la classification
        plt.imshow(masked_result, cmap=cmap, norm=norm)
        plt.title('Carte de Classification du Sol', fontsize=16, fontweight='bold')
        
        # Ajouter la légende
        legend_elements = []
        for cls in sorted(unique_classes):
            color = config["class_colors"].get(cls, f"C{cls%10}")
            name = config["class_names"].get(cls, f"Classe {cls}")
            legend_elements.append(mpatches.Patch(facecolor=color, edgecolor='black', label=name))
        
        # Positionner la légende à droite de la carte
        plt.legend(handles=legend_elements, 
                  loc='center left', 
                  bbox_to_anchor=(1, 0.5),
                  title="CLASSES", 
                  fontsize=12,
                  title_fontsize=14,
                  frameon=True)
        
        plt.tight_layout()
        plt.savefig(map_output, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Carte de classification sauvegardée: {map_output}")
        return True
    else:
        print("Aucune classe trouvée dans la classification")
        return False

def visualize_uncertainty(uncertainty_map, config):
    """
    Génère une carte d'incertitude/entropie.
    
    Args:
        uncertainty_map (numpy.ndarray): Carte d'incertitude (entropie)
        config (dict): Configuration contenant les paramètres de visualisation
        
    Returns:
        bool: True si réussi, False sinon
    """
    print("\nGénération de la carte d'incertitude...")
    
    output_dir = config["output_dir"]
    uncertainty_output = os.path.join(output_dir, "carte_incertitude.png")
    
    # Créer une figure
    plt.figure(figsize=(10, 8), dpi=300)
    
    # Utiliser une colormap appropriée pour l'incertitude (ex: inferno, plasma)
    cmap = plt.cm.plasma
    
    # Afficher l'incertitude
    im = plt.imshow(uncertainty_map, cmap=cmap, vmin=0, vmax=1)
    
    # Ajouter une barre de couleur
    cbar = plt.colorbar(im, shrink=0.8)
    cbar.set_label('Incertitude (entropie normalisée)', fontsize=12)
    
    plt.title('Carte d\'Incertitude de Classification', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(uncertainty_output, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Carte d'incertitude sauvegardée: {uncertainty_output}")
    return True
