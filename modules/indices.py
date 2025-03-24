"""
INDICES SPECTRAUX AVANCÉS
=========================
Ce module calcule des indices spectraux avancés à partir des bandes Sentinel-2.
"""

import numpy as np

def calculate_ndvi(nir, red):
    """
    Calcule l'indice NDVI (Normalized Difference Vegetation Index).
    NDVI = (NIR - Rouge) / (NIR + Rouge)
    
    Args:
        nir: Bande proche infrarouge (B8)
        red: Bande rouge (B4)
    
    Returns:
        Array numpy de l'indice NDVI
    """
    # Éviter la division par zéro
    denominator = nir + red
    mask = denominator != 0
    
    ndvi = np.zeros_like(nir, dtype=np.float32)
    ndvi[mask] = (nir[mask] - red[mask]) / denominator[mask]
    
    return ndvi

def calculate_ndre(nir, rededge):
    """
    Calcule l'indice NDRE (Normalized Difference Red Edge).
    NDRE = (NIR - RedEdge) / (NIR + RedEdge)
    
    Cet indice est particulièrement utile pour distinguer les différents
    types de végétation et leur état de santé.
    
    Args:
        nir: Bande proche infrarouge (B8)
        rededge: Bande red edge (B5, B6 ou B7)
    
    Returns:
        Array numpy de l'indice NDRE
    """
    # Éviter la division par zéro
    denominator = nir + rededge
    mask = denominator != 0
    
    ndre = np.zeros_like(nir, dtype=np.float32)
    ndre[mask] = (nir[mask] - rededge[mask]) / denominator[mask]
    
    return ndre

def calculate_ndwi(nir, green):
    """
    Calcule l'indice NDWI (Normalized Difference Water Index).
    NDWI = (Vert - NIR) / (Vert + NIR)
    
    Cet indice est utile pour détecter les plans d'eau et l'humidité.
    
    Args:
        nir: Bande proche infrarouge (B8)
        green: Bande verte (B3)
    
    Returns:
        Array numpy de l'indice NDWI
    """
    # Éviter la division par zéro
    denominator = green + nir
    mask = denominator != 0
    
    ndwi = np.zeros_like(nir, dtype=np.float32)
    ndwi[mask] = (green[mask] - nir[mask]) / denominator[mask]
    
    return ndwi

def calculate_mndwi(green, swir):
    """
    Calcule l'indice MNDWI (Modified Normalized Difference Water Index).
    MNDWI = (Vert - SWIR) / (Vert + SWIR)
    
    Cet indice est plus efficace que le NDWI standard pour distinguer
    l'eau des surfaces urbaines.
    
    Args:
        green: Bande verte (B3)
        swir: Bande SWIR (B11 ou B12)
    
    Returns:
        Array numpy de l'indice MNDWI
    """
    # Éviter la division par zéro
    denominator = green + swir
    mask = denominator != 0
    
    mndwi = np.zeros_like(green, dtype=np.float32)
    mndwi[mask] = (green[mask] - swir[mask]) / denominator[mask]
    
    return mndwi

def calculate_savi(nir, red, L=0.5):
    """
    Calcule l'indice SAVI (Soil Adjusted Vegetation Index).
    SAVI = ((NIR - Rouge) / (NIR + Rouge + L)) * (1 + L)
    
    Cet indice réduit l'influence du sol dans l'indice de végétation.
    
    Args:
        nir: Bande proche infrarouge (B8)
        red: Bande rouge (B4)
        L: Facteur d'ajustement du sol (0 pour haute densité végétale, 1 pour faible densité)
    
    Returns:
        Array numpy de l'indice SAVI
    """
    # Éviter la division par zéro
    denominator = nir + red + L
    mask = denominator != 0
    
    savi = np.zeros_like(nir, dtype=np.float32)
    savi[mask] = ((nir[mask] - red[mask]) / denominator[mask]) * (1 + L)
    
    return savi

def calculate_evi(nir, red, blue, G=2.5, C1=6.0, C2=7.5, L=1.0):
    """
    Calcule l'indice EVI (Enhanced Vegetation Index).
    EVI = G * ((NIR - Rouge) / (NIR + C1 * Rouge - C2 * Bleu + L))
    
    Cet indice est optimisé pour les zones de forte biomasse et réduit
    l'influence de l'atmosphère.
    
    Args:
        nir: Bande proche infrarouge (B8)
        red: Bande rouge (B4)
        blue: Bande bleue (B2)
        G, C1, C2, L: Coefficients d'ajustement
    
    Returns:
        Array numpy de l'indice EVI
    """
    # Éviter la division par zéro
    denominator = nir + C1 * red - C2 * blue + L
    mask = denominator != 0
    
    evi = np.zeros_like(nir, dtype=np.float32)
    evi[mask] = G * ((nir[mask] - red[mask]) / denominator[mask])
    
    # Limiter les valeurs entre -1 et 1
    evi = np.clip(evi, -1, 1)
    
    return evi

def calculate_bsi(blue, red, nir, swir):
    """
    Calcule l'indice BSI (Bare Soil Index).
    BSI = ((SWIR + Rouge) - (NIR + Bleu)) / ((SWIR + Rouge) + (NIR + Bleu))
    
    Cet indice est utile pour distinguer les sols nus.
    
    Args:
        blue: Bande bleue (B2)
        red: Bande rouge (B4)
        nir: Bande proche infrarouge (B8)
        swir: Bande SWIR (B11)
    
    Returns:
        Array numpy de l'indice BSI
    """
    # Éviter la division par zéro
    numerator = (swir + red) - (nir + blue)
    denominator = (swir + red) + (nir + blue)
    mask = denominator != 0
    
    bsi = np.zeros_like(blue, dtype=np.float32)
    bsi[mask] = numerator[mask] / denominator[mask]
    
    return bsi

def calculate_ndmi(nir, swir):
    """
    Calcule l'indice NDMI (Normalized Difference Moisture Index).
    NDMI = (NIR - SWIR) / (NIR + SWIR)
    
    Cet indice est utile pour évaluer l'humidité de la végétation.
    
    Args:
        nir: Bande proche infrarouge (B8)
        swir: Bande SWIR (B11)
    
    Returns:
        Array numpy de l'indice NDMI
    """
    # Éviter la division par zéro
    denominator = nir + swir
    mask = denominator != 0
    
    ndmi = np.zeros_like(nir, dtype=np.float32)
    ndmi[mask] = (nir[mask] - swir[mask]) / denominator[mask]
    
    return ndmi

def calculate_all_indices(bands):
    """
    Calcule tous les indices spectraux à partir des bandes Sentinel-2.
    
    Args:
        bands: Dictionnaire contenant les bandes Sentinel-2
              (clés: 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B11', 'B12')
    
    Returns:
        Dictionnaire contenant tous les indices calculés
    """
    indices = {}
    
    # Vérifier que toutes les bandes nécessaires sont présentes
    required_bands = ['B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B11']
    for band in required_bands:
        if band not in bands:
            raise ValueError(f"La bande {band} est requise pour calculer les indices")
    
    # Calculer les indices de végétation
    indices['NDVI'] = calculate_ndvi(bands['B8'], bands['B4'])
    indices['NDRE1'] = calculate_ndre(bands['B8'], bands['B5'])  # RedEdge05
    indices['NDRE2'] = calculate_ndre(bands['B8'], bands['B6'])  # RedEdge06
    indices['NDRE3'] = calculate_ndre(bands['B8'], bands['B7'])  # RedEdge07
    
    # Calculer les indices d'eau et d'humidité
    indices['NDWI'] = calculate_ndwi(bands['B8'], bands['B3'])
    indices['MNDWI'] = calculate_mndwi(bands['B3'], bands['B11'])
    indices['NDMI'] = calculate_ndmi(bands['B8'], bands['B11'])
    
    # Calculer les indices de sol et de végétation améliorés
    indices['SAVI'] = calculate_savi(bands['B8'], bands['B4'])
    indices['EVI'] = calculate_evi(bands['B8'], bands['B4'], bands['B2'])
    indices['BSI'] = calculate_bsi(bands['B2'], bands['B4'], bands['B8'], bands['B11'])
    
    return indices
