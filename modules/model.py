"""
MODEL MODULE
===========
Ce module contient les implémentations des différents modèles de classification,
notamment la classification par maximum de vraisemblance.
"""

import numpy as np
from sklearn.mixture import GaussianMixture

def perform_classification(raster_data, classes_info, config):
    """
    Réalise la classification par maximum de vraisemblance en utilisant scikit-learn.
    Retourne la classification et les probabilités par classe.
    
    Args:
        raster_data (numpy.ndarray): Données raster en entrée (bandes, hauteur, largeur)
        classes_info (dict): Informations sur les classes
        config (dict): Configuration du processus de classification
        
    Returns:
        tuple: (classification, probabilités) - La classification (2D) et les probabilités par classe
    """
    print("\nClassification en cours...")
    
    # Préparer les données raster pour la classification
    bands, height, width = raster_data.shape
    raster_data_2d = raster_data.reshape(bands, -1).T
    
    # Initialiser les tableaux pour le résultat et les probabilités
    result = np.zeros(raster_data_2d.shape[0], dtype=np.uint8)
    max_prob = np.full(raster_data_2d.shape[0], -np.inf)
    probabilities = {}  # Stocker les probabilités pour chaque classe
    
    # Créer un masque des pixels valides
    valid_mask = ~np.isnan(raster_data_2d).any(axis=1) & ~np.all(raster_data_2d == 0, axis=1)
    valid_indices = np.where(valid_mask)[0]
    valid_data = raster_data_2d[valid_indices]
    
    print(f"Classification de {len(valid_indices)} pixels valides...")
    
    # Pour chaque classe, entraîner un modèle GaussianMixture et calculer les probabilités
    for class_id, class_info in classes_info.items():
        try:
            # Initialiser un modèle de mixture gaussienne (scikit-learn)
            gm = GaussianMixture(
                n_components=1,
                covariance_type='full',
                means_init=[class_info['mean']],
                random_state=42
            )
            
            # Entraîner le modèle avec les données d'entraînement
            gm.fit(class_info['training_data'])
            
            # Calculer les log-probabilités pour les pixels valides
            log_probs = gm.score_samples(valid_data)
            
            # Stocker les probabilités pour cette classe
            if class_id not in probabilities:
                probabilities[class_id] = np.full(raster_data_2d.shape[0], -np.inf)
            
            probabilities[class_id][valid_indices] = log_probs
            
            # Mettre à jour la classification si c'est la meilleure log-probabilité
            better_mask = log_probs > max_prob[valid_indices]
            max_prob[valid_indices[better_mask]] = log_probs[better_mask]
            result[valid_indices[better_mask]] = class_id
            
            print(f"  Classe {class_id} traitée")
        except Exception as e:
            print(f"  Erreur pour la classe {class_id}: {e}")
    
    # Reformer le résultat à la forme originale du raster
    classification = result.reshape(height, width)
    print("Classification terminée!")
    
    return classification, probabilities

def perform_classification_weighted(raster_data, classes_info, config, band_weights=None):
    """
    Réalise la classification par maximum de vraisemblance avec des bandes pondérées.
    Implémentation complètement revue pour assurer que les poids ont un impact réel.
    
    Args:
        raster_data (numpy.ndarray): Données raster en entrée
        classes_info (dict): Informations sur les classes
        config (dict): Configuration de la classification
        band_weights (numpy.ndarray, optional): Pondérations personnalisées pour chaque bande
    """
    print("\nNouvelle approche de classification avec pondération des bandes...")
    
    # Préparer les dimensions des bandes
    bands, height, width = raster_data.shape
    
    # Définir les poids des bandes
    if band_weights is None:
        band_weights = np.ones(bands)
    
    # Vérification des dimensions
    if len(band_weights) != bands:
        print(f"ATTENTION: Les dimensions des poids ({len(band_weights)}) ne correspondent pas aux bandes ({bands})")
        # Ajuster les poids si nécessaire
        if len(band_weights) > bands:
            band_weights = band_weights[:bands]
        else:
            extended_weights = np.ones(bands)
            extended_weights[:len(band_weights)] = band_weights
            band_weights = extended_weights
    
    # Affichage des poids
    print("Poids appliqués aux bandes:")
    for i, weight in enumerate(band_weights):
        print(f"  Bande {i+1}: {weight:.2f}")
    
    # Préparer les données raster pour la classification
    raster_data_2d = raster_data.reshape(bands, -1).T
    
    # Initialiser les tableaux pour le résultat et les probabilités
    result = np.zeros(raster_data_2d.shape[0], dtype=np.uint8)
    max_prob = np.full(raster_data_2d.shape[0], -np.inf)
    probabilities = {}  # Stocker les probabilités pour chaque classe
    
    # Créer un masque des pixels valides
    valid_mask = ~np.isnan(raster_data_2d).any(axis=1) & ~np.all(raster_data_2d == 0, axis=1)
    valid_indices = np.where(valid_mask)[0]
    valid_data = raster_data_2d[valid_indices]
    
    print(f"Classification de {len(valid_indices)} pixels valides avec pondération...")
    
    # Matrice diagonale de poids pour transformer la matrice de covariance
    # Cette approche modifie directement l'importance de chaque bande dans le calcul de vraisemblance
    W = np.diag(band_weights)
    
    # Pour chaque classe, calculer les probabilités pondérées
    for class_id, class_info in classes_info.items():
        print(f"  Traitement de la classe {class_id} avec pondération...")
        
        try:
            # Extraire les statistiques de classe
            mean = class_info['mean']
            cov = class_info['cov']
            
            # APPROCHE 1: Pondérer la matrice de covariance directement
            # Cov_weighted = W^(1/2) * Cov * W^(1/2)
            # Cette transformation modifie l'importance des variables dans la distribution
            W_sqrt = np.sqrt(W)
            weighted_cov = W_sqrt @ cov @ W_sqrt
            
            # Ajouter une petite régularisation pour éviter les problèmes numériques
            weighted_cov += np.eye(bands) * 1e-5
            
            # DEBUGGING - Vérifier l'impact sur la covariance
            print(f"    Impact sur la variance pour classe {class_id}:")
            for i in range(bands):
                print(f"      Bande {i+1}: Original={cov[i,i]:.4f}, Pondéré={weighted_cov[i,i]:.4f}, Facteur={weighted_cov[i,i]/cov[i,i]:.2f}x")
            
            # Calculer le déterminant et l'inverse pour le calcul de la log-vraisemblance
            try:
                sign, logdet = np.linalg.slogdet(weighted_cov)
                if sign <= 0:
                    print(f"    ATTENTION: Déterminant négatif ou nul pour classe {class_id}, ajout de régularisation...")
                    weighted_cov += np.eye(bands) * 0.01  # Ajouter plus de régularisation
                    sign, logdet = np.linalg.slogdet(weighted_cov)
                
                inv_cov = np.linalg.inv(weighted_cov)
            except np.linalg.LinAlgError:
                print(f"    ERREUR: Matrice de covariance singulière pour classe {class_id}, utilisation de pseudo-inverse...")
                inv_cov = np.linalg.pinv(weighted_cov)
                sign, logdet = np.linalg.slogdet(weighted_cov + np.eye(bands) * 0.01)
            
            # Calculer manuellement les log-probabilités pour chaque pixel valide
            log_probs = np.zeros(len(valid_indices))
            
            for i, pixel in enumerate(valid_data):
                # Distance de Mahalanobis pondérée: (x-μ)' Σ^-1 (x-μ)
                diff = pixel - mean
                mahalanobis2 = diff @ inv_cov @ diff
                
                # Log de la densité de probabilité d'une distribution normale multivariée
                log_probs[i] = -0.5 * (bands * np.log(2 * np.pi) + logdet + mahalanobis2)
            
            # Stocker les probabilités pour cette classe
            if class_id not in probabilities:
                probabilities[class_id] = np.full(raster_data_2d.shape[0], -np.inf)
            
            probabilities[class_id][valid_indices] = log_probs
            
            # Mettre à jour la classification si c'est la meilleure log-probabilité
            better_mask = log_probs > max_prob[valid_indices]
            max_prob[valid_indices[better_mask]] = log_probs[better_mask]
            result[valid_indices[better_mask]] = class_id
            
            print(f"    Classe {class_id} traitée avec pondération")
            
        except Exception as e:
            print(f"    ERREUR pour la classe {class_id}: {e}")
            import traceback
            print(traceback.format_exc())
    
    # Reformer le résultat à la forme originale du raster
    classification = result.reshape(height, width)
    print("Classification avec pondération terminée!")
    
    return classification, probabilities

def calculate_uncertainty(probabilities, height, width):
    """
    Calcule une carte d'incertitude/entropie basée sur les probabilités de classification.
    Retourne une carte d'entropie normalisée.
    
    Args:
        probabilities (dict): Dictionnaire des probabilités par classe
        height (int): Hauteur de l'image
        width (int): Largeur de l'image
        
    Returns:
        numpy.ndarray: Carte d'entropie normalisée
    """
    print("\nCalcul de la carte d'incertitude...")
    
    # Vérifier si le dictionnaire de probabilités est vide
    if not probabilities:
        print("AVERTISSEMENT: Aucune probabilité à analyser")
        return np.zeros((height, width))
    
    try:
        # Convertir les log-probabilités en probabilités
        prob_arrays = []
        for class_id in probabilities:
            probs = np.exp(probabilities[class_id])
            prob_arrays.append(probs)
        
        prob_arrays = np.array(prob_arrays)
        
        # Vérifier la forme du tableau
        if prob_arrays.ndim < 2:
            print("AVERTISSEMENT: Format de probabilités incorrect, retourne matrice zéro")
            return np.zeros((height, width))
        
        # Calculer la somme des probabilités
        sum_probs = np.sum(prob_arrays, axis=0)
        
        # Gérer le cas où sum_probs est un scalaire
        if np.isscalar(sum_probs):
            print("AVERTISSEMENT: La somme des probabilités est un scalaire")
            return np.zeros((height, width))
        
        # Éviter la division par zéro
        sum_probs = np.where(sum_probs == 0, 1, sum_probs)
        
        # Normaliser les probabilités
        norm_probs = prob_arrays / sum_probs
        
        # Calculer l'entropie (mesure d'incertitude)
        epsilon = 1e-10  # Pour éviter log(0)
        entropy = np.zeros(norm_probs.shape[1])
        for i in range(norm_probs.shape[0]):
            p = norm_probs[i]
            entropy -= p * np.log2(p + epsilon)
        
        # Normaliser l'entropie entre 0 et 1
        entropy = entropy / np.log2(len(probabilities))
        entropy_map = entropy.reshape(height, width)
        
        print("Calcul de la carte d'incertitude terminé")
        return entropy_map
        
    except Exception as e:
        print(f"ERREUR dans le calcul de l'incertitude: {e}")
        import traceback
        traceback.print_exc()
        return np.zeros((height, width))
