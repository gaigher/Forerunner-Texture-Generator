"""
Effet glow (lueur) pour Texture Halo.
"""
from PIL import Image, ImageFilter
import numpy as np
from typing import Optional
from ui.logger import ui_log


def apply_glow(
    img: Image.Image,
    radius: int = 18,
    threshold: int = 200,
    intensity: float = 1.4,
    allow_conversion: bool = False
) -> Image.Image:
    """
    Applique un effet de lueur (glow) à une image.
    
    Extrait les zones lumineuses, les floute, puis les recompose avec l'image originale.
    
    Paramètres :
        img: Image d'entrée
        radius: Rayon du flou gaussien (en pixels)
        threshold: Seuil de luminosité (0-255) pour extraire les zones lumineuses
        intensity: Intensité de la lueur (multiplicateur)
        allow_conversion: Si True, permet la conversion de mode si nécessaire
        
    Renvoie :
        Image avec effet glow appliqué (mode RGBA)
    """
    # Convertir en RGBA si nécessaire
    if img.mode != 'RGBA':
        if allow_conversion:
            img = img.convert('RGBA')
        else:
            from ui.logger import ConversionNotAllowed
            raise ConversionNotAllowed(
                f"Conversion de mode {img.mode} vers RGBA requise. "
                "Utilisez allow_conversion=True pour autoriser."
            )
    
    # Extraire les zones lumineuses
    lum = img.convert('L')
    mask_arr = np.asarray(lum, dtype='float32')
    # Créer un masque pour les zones >= threshold
    bright_mask = (mask_arr >= threshold).astype('float32')
    
    # Extraire les pixels lumineux
    img_arr = np.asarray(img, dtype='float32')
    bright_arr = img_arr * bright_mask[:, :, np.newaxis]
    bright_img = Image.fromarray(bright_arr.clip(0, 255).astype('uint8'), 'RGBA')
    
    # Appliquer le flou gaussien
    if radius > 0:
        blurred = bright_img.filter(ImageFilter.GaussianBlur(radius=radius))
    else:
        blurred = bright_img
    
    # Multiplier par l'intensité
    if intensity != 1.0:
        blurred_arr = np.asarray(blurred, dtype='float32')
        blurred_arr = blurred_arr * intensity
        blurred = Image.fromarray(blurred_arr.clip(0, 255).astype('uint8'), 'RGBA')
    
    # Composer avec l'image originale (mélange additif)
    result_arr = np.asarray(img, dtype='float32')
    blurred_arr = np.asarray(blurred, dtype='float32')
    
    # Composition alpha avec addition pour l'effet glow
    alpha_blur = blurred_arr[:, :, 3:4] / 255.0
    result_arr = result_arr + blurred_arr * alpha_blur * 0.5
    result_arr = result_arr.clip(0, 255).astype('uint8')
    
    return Image.fromarray(result_arr, 'RGBA')
