"""
Effet Spread Grain pour Texture Halo.
Implémentation du filtre "Éparpiller" de GIMP : réarrangement aléatoire de pixels.
"""
from PIL import Image, ImageFilter
import numpy as np
from typing import Optional, Any
from ui.logger import ui_log

# Si la boîte du masque couvre plus que ce ratio de l'image, on garde le chemin
# plein cadre (évite un recadrage quasi total sans gain).
_SUPPORT_MASK_BBOX_MAX_COVER = 0.92


def _apply_spread_grain_fast_support_mask_bbox(
    img_data: np.ndarray,
    mask: np.ndarray,
    spread_x: float,
    spread_y: float,
    seed: Optional[int],
    h: int,
    w: int,
):
    """
    Spread grain avec masque : n'évalue map_coordinates que sur la boîte du masque
    (+ marge), en échantillonnant depuis l'image entière. Réduit fortement le coût
    quand le masque ne couvre qu'une partie de la texture (ex. un seul étage).
    Retourne un tableau float32 (H,W,4) ou None pour utiliser le chemin plein cadre.
    """
    active = mask > 1e-6
    ys, xs = np.where(active)
    margin = int(np.ceil(max(float(spread_x), float(spread_y)))) + 2
    y0 = max(0, int(ys.min()) - margin)
    y1 = min(h, int(ys.max()) + margin + 1)
    x0 = max(0, int(xs.min()) - margin)
    x1 = min(w, int(xs.max()) + margin + 1)
    hc, wc = y1 - y0, x1 - x0

    if hc * wc >= _SUPPORT_MASK_BBOX_MAX_COVER * h * w:
        return None  # signal : utiliser le chemin plein cadre ci-dessous

    rng = np.random.RandomState(seed)
    mask_c = mask[y0:y1, x0:x1]
    y_loc, x_loc = np.meshgrid(np.arange(hc), np.arange(wc), indexing="ij")
    dx = rng.uniform(-spread_x, spread_x, size=(hc, wc)) * mask_c
    dy = rng.uniform(-spread_y, spread_y, size=(hc, wc)) * mask_c
    new_y = np.clip(y0 + y_loc + dy, 0, h - 1)
    new_x = np.clip(x0 + x_loc + dx, 0, w - 1)

    try:
        from scipy.ndimage import map_coordinates
    except ImportError:
        return None

    out_c = np.zeros((hc, wc, 4), dtype=np.float32)
    for ch in range(4):
        out_c[:, :, ch] = map_coordinates(
            img_data[:, :, ch],
            [new_y, new_x],
            order=1,
            mode="constant",
            cval=0.0,
            prefilter=False,
        )
    mask_3d = mask_c[:, :, np.newaxis]
    src_c = img_data[y0:y1, x0:x1]
    blended = out_c * mask_3d + src_c * (1.0 - mask_3d)

    result = img_data.copy()
    result[y0:y1, x0:x1] = blended
    return result


def apply_spread_grain(
    img: Image.Image,
    spread_x: float = 2.0,
    spread_y: float = 2.0,
    seed: Optional[int] = None,
    allow_conversion: bool = False,
    edge_mask: bool = False,
    zone_width: float = 2.0,
    support_mask: Optional[Any] = None,
) -> Image.Image:
    """
    Applique un effet de grain par dispersion de pixels (style GIMP "Éparpiller").
    
    Réarrange aléatoirement les pixels dans un rayon donné, créant un effet
    de grain naturel sans introduire de nouvelles couleurs.
    
    Paramètres :
        img: Image d'entrée
        spread_x: Distance maximale de déplacement horizontal (pixels)
        spread_y: Distance maximale de déplacement vertical (pixels)
        seed: Graine aléatoire pour la reproductibilité
        allow_conversion: Si True, permet la conversion de mode si nécessaire
        edge_mask: Si True, applique uniquement sur les zones d'arêtes (mode hybride)
        zone_width: Largeur de la zone d'application autour des arêtes (si edge_mask=True)
        support_mask: Masque float (H, W) dans [0, 1] ; si fourni, prioritaire sur edge_mask.
        
    Renvoie :
        Image avec effet d'éparpillage du grain appliqué (mode RGBA)
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
    
    h, w = img.size[1], img.size[0]
    img_data = np.array(img, dtype='float32')
    if not img_data.flags['WRITEABLE']:
        img_data = img_data.copy()
    
    # Créer le masque d'application
    if support_mask is not None:
        sm = np.asarray(support_mask, dtype=np.float32)
        if sm.shape != (h, w):
            raise ValueError(
                f"support_mask must have shape (H, W)=({h}, {w}), got {sm.shape}"
            )
        mask = np.clip(sm, 0.0, 1.0)
    elif edge_mask:
        # Détecter les arêtes (zones de transition entre couleurs)
        gray = img.convert('L')
        edges = gray.filter(ImageFilter.FIND_EDGES)
        edges_data = np.array(edges, dtype='float32')
        if not edges_data.flags['WRITEABLE']:
            edges_data = edges_data.copy()
        
        # Normaliser les arêtes (0-1)
        if edges_data.max() > 0:
            edges_normalized = edges_data / 255.0
        else:
            edges_normalized = np.zeros((h, w), dtype='float32')
        
        # Créer un masque binaire des zones d'arêtes fortes
        edge_threshold = 0.3
        mask = (edges_normalized > edge_threshold).astype('float32')
        
        # Dilater le masque pour élargir la zone
        if zone_width > 1.0:
            kernel_size = int(zone_width * 2) + 1
            if kernel_size % 2 == 0:
                kernel_size += 1
            
            # Essayer d'utiliser scipy pour la dilatation
            try:
                from scipy.ndimage import maximum_filter
                mask = maximum_filter(mask, size=kernel_size)
            except ImportError:
                # Repli : dilatation manuelle
                try:
                    import cv2
                    kernel = np.ones((kernel_size, kernel_size), np.uint8)
                    mask = cv2.dilate(mask.astype('uint8'), kernel, iterations=1).astype('float32')
                except ImportError:
                    # Dernier repli : dilatation simple
                    half_k = kernel_size // 2
                    dilated_mask = np.zeros((h, w), dtype='float32')
                    padded = np.pad(mask, half_k, mode='edge')
                    
                    for y in range(h):
                        for x in range(w):
                            y_start = y
                            y_end = y + kernel_size
                            x_start = x
                            x_end = x + kernel_size
                            dilated_mask[y, x] = float(np.max(padded[y_start:y_end, x_start:x_end]))
                    mask = dilated_mask
    else:
        # Masque complet : appliquer partout
        mask = np.ones((h, w), dtype='float32')
    
    # Générer les déplacements aléatoires pour chaque pixel
    rng = np.random.RandomState(seed if seed is not None else None)
    
    # Créer les coordonnées de destination pour chaque pixel
    y_coords, x_coords = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
    
    # Générer les déplacements aléatoires
    dx = rng.uniform(-spread_x, spread_x, size=(h, w))
    dy = rng.uniform(-spread_y, spread_y, size=(h, w))
    
    # Appliquer le masque : si le pixel n'est pas dans le masque, pas de déplacement
    dx = dx * mask
    dy = dy * mask
    
    # Calculer les nouvelles coordonnées
    new_x = (x_coords + dx).astype('float32')
    new_y = (y_coords + dy).astype('float32')
    
    # Borner les coordonnées aux limites de l'image
    new_x = np.clip(new_x, 0, w - 1)
    new_y = np.clip(new_y, 0, h - 1)
    
    # Créer l'image de sortie
    result = np.zeros((h, w, 4), dtype='uint8')
    
    # Pour chaque pixel de sortie, lire depuis la position déplacée
    # Utiliser l'interpolation bilinéaire pour des résultats plus lisses
    for y in range(h):
        for x in range(w):
            if mask[y, x] > 0:  # Seulement si dans le masque
                src_x = new_x[y, x]
                src_y = new_y[y, x]
                
                # Interpolation bilinéaire
                x0 = int(np.floor(src_x))
                y0 = int(np.floor(src_y))
                x1 = min(int(np.ceil(src_x)), w - 1)
                y1 = min(int(np.ceil(src_y)), h - 1)
                
                # Coefficients d'interpolation
                fx = src_x - x0
                fy = src_y - y0
                
                # Lire les 4 pixels voisins
                p00 = img_data[y0, x0]
                p01 = img_data[y1, x0] if y1 < h else p00
                p10 = img_data[y0, x1] if x1 < w else p00
                p11 = img_data[y1, x1] if (y1 < h and x1 < w) else p00
                
                # Interpolation bilinéaire
                result[y, x] = (
                    p00 * (1 - fx) * (1 - fy) +
                    p10 * fx * (1 - fy) +
                    p01 * (1 - fx) * fy +
                    p11 * fx * fy
                ).astype('uint8')
            else:
                # Conserver le pixel original si hors masque
                result[y, x] = img_data[y, x].astype('uint8')
    
    return Image.fromarray(result, 'RGBA')


def apply_spread_grain_fast(
    img: Image.Image,
    spread_x: float = 2.0,
    spread_y: float = 2.0,
    seed: Optional[int] = None,
    allow_conversion: bool = False,
    edge_mask: bool = False,
    zone_width: float = 2.0,
    support_mask: Optional[Any] = None,
) -> Image.Image:
    """
    Éparpillage du grain avec interpolation SciPy (``apply_spread_grain`` pour l’implémentation de référence).
    
    Avec ``support_mask`` (grain par étage), n'évalue l'interpolation que sur la
    boîte englobante du masque (+ marge), ce qui accélère fortement les grandes
    textures lorsque le palier n'occupe qu'une partie de l'image.
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
    
    h, w = img.size[1], img.size[0]
    img_data = np.array(img, dtype='float32')
    if not img_data.flags['WRITEABLE']:
        img_data = img_data.copy()
    
    # Créer le masque d'application
    if support_mask is not None:
        sm = np.asarray(support_mask, dtype=np.float32)
        if sm.shape != (h, w):
            raise ValueError(
                f"support_mask must have shape (H, W)=({h}, {w}), got {sm.shape}"
            )
        mask = np.clip(sm, 0.0, 1.0)
        if not np.any(mask > 1e-6):
            return img
        bbox_result = _apply_spread_grain_fast_support_mask_bbox(
            img_data, mask, spread_x, spread_y, seed, h, w
        )
        if bbox_result is not None:
            return Image.fromarray(bbox_result.clip(0, 255).astype("uint8"), "RGBA")
    elif edge_mask:
        gray = img.convert('L')
        edges = gray.filter(ImageFilter.FIND_EDGES)
        edges_data = np.array(edges, dtype='float32')
        if not edges_data.flags['WRITEABLE']:
            edges_data = edges_data.copy()
        
        if edges_data.max() > 0:
            edges_normalized = edges_data / 255.0
        else:
            edges_normalized = np.zeros((h, w), dtype='float32')
        
        edge_threshold = 0.3
        mask = (edges_normalized > edge_threshold).astype('float32')
        
        if zone_width > 1.0:
            kernel_size = int(zone_width * 2) + 1
            if kernel_size % 2 == 0:
                kernel_size += 1
            
            try:
                from scipy.ndimage import maximum_filter
                mask = maximum_filter(mask, size=kernel_size)
            except ImportError:
                ui_log("scipy non disponible : repli sur apply_spread_grain", 'warning')
                return apply_spread_grain(
                    img, spread_x, spread_y, seed, allow_conversion,
                    edge_mask, zone_width, support_mask,
                )
    else:
        mask = np.ones((h, w), dtype='float32')
    
    # Générer les déplacements aléatoires
    rng = np.random.RandomState(seed if seed is not None else None)
    y_coords, x_coords = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
    
    dx = rng.uniform(-spread_x, spread_x, size=(h, w)) * mask
    dy = rng.uniform(-spread_y, spread_y, size=(h, w)) * mask
    
    new_x = np.clip(x_coords + dx, 0, w - 1)
    new_y = np.clip(y_coords + dy, 0, h - 1)
    
    # Utiliser scipy pour l'interpolation rapide
    try:
        from scipy.ndimage import map_coordinates
        
        result = np.zeros((h, w, 4), dtype='float32')
        for channel in range(4):
            result[:, :, channel] = map_coordinates(
                img_data[:, :, channel],
                [new_y, new_x],
                order=1,  # Interpolation bilinéaire
                mode='constant',
                cval=0.0,
                prefilter=False
            )
        
        # Appliquer le masque : pixels hors masque conservent leur valeur originale
        mask_3d = mask[:, :, np.newaxis]
        result = result * mask_3d + img_data * (1 - mask_3d)
        
        return Image.fromarray(result.clip(0, 255).astype('uint8'), 'RGBA')
    except ImportError:
        ui_log("scipy non disponible : repli sur apply_spread_grain", 'warning')
        return apply_spread_grain(
            img, spread_x, spread_y, seed, allow_conversion,
            edge_mask, zone_width, support_mask,
        )

