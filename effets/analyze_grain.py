"""
Préparation du grain pour les étapes de blanchiment.
"""
import numpy as np


def remplir_etages_dessous_reference_affleurement(
    rgb: np.ndarray,
    idx: np.ndarray,
    k: int,
) -> np.ndarray:
    """
    Uniformise les pixels ``idx < k`` avec la couleur RVB du premier pixel du palier ``k``
    (même règle que la teinte « affleurement » utilisée en ``blanch_04``), sans modifier
    les pixels ``idx >= k``. Entrée / sortie : ``uint8``, forme (H, W, 3).
    """
    if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("rgb doit être un tableau (H, W, 3) uint8")
    if idx.shape[:2] != rgb.shape[:2]:
        raise ValueError("idx et rgb doivent avoir les mêmes dimensions H×W")
    below = idx < k
    on_k = idx == k
    out = rgb.copy()
    if not np.any(on_k):
        return out
    first = int(np.flatnonzero(on_k.ravel())[0])
    y0, x0 = np.unravel_index(first, idx.shape)
    fill_rgb = out[y0, x0].copy()
    out[below] = fill_rgb
    return out


