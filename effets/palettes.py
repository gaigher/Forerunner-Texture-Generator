"""
Utilitaires de palettes pour Texture Halo.
"""
from PIL import Image
import numpy as np
from typing import Union, List, Tuple


def apply_palette(
    image: Image.Image,
    palette: Union[List[Tuple[int, int, int]], Image.Image]
) -> Image.Image:
    """
    Applique une palette de couleurs à une image.

    Paramètres :
        image: Image à traiter
        palette: Palette sous forme de liste de tuples RGB ou Image PIL

    Renvoie :
        Image avec palette appliquée
    """
    # Convertir la palette en liste de tuples RGB si c'est une image
    if isinstance(palette, Image.Image):
        pal_arr = np.asarray(palette.convert('RGB'))
        if pal_arr.ndim == 2:
            # Palette 1D (ligne de pixels)
            palette_list = [tuple(pal_arr[i]) for i in range(len(pal_arr))]
        elif pal_arr.ndim == 3:
            # Palette 2D, extraire une ligne (par exemple la première)
            if pal_arr.shape[0] == 1:
                palette_list = [tuple(pal_arr[0, i]) for i in range(pal_arr.shape[1])]
            else:
                # Prendre la ligne du milieu
                mid_row = pal_arr.shape[0] // 2
                palette_list = [tuple(pal_arr[mid_row, i]) for i in range(pal_arr.shape[1])]
        else:
            raise ValueError("Format de palette image non supporté")
    else:
        palette_list = palette

    if len(palette_list) == 0:
        return image.convert('RGB')

    # Convertir l'image en niveaux de gris
    img = image.convert('L')
    arr = np.asarray(img, dtype='float32')

    # Mapper les valeurs 0..255 aux indices de la palette
    n = len(palette_list)
    idx = (arr * (n - 1) / 255.0).round().astype(int).clip(0, n - 1)

    # Créer l'image de sortie
    pal_array = np.array(palette_list, dtype='uint8')
    out = pal_array[idx]

    result = Image.fromarray(out, 'RGB')

    # Préserver l'alpha si présent
    if image.mode == 'RGBA':
        alpha = image.split()[3]
        result = result.convert('RGBA')
        result.putalpha(alpha)

    return result
