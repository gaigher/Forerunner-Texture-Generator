"""
Utilitaires pour le traitement d'images Texture Halo.
"""
from PIL import Image, ImageFilter, ImageChops, ImageEnhance
import numpy as np
from scipy import ndimage
from typing import List, Optional, Sequence, Tuple
from ui.logger import ui_log
from ui.settings import (
    DEFAULT_MASQUE_BLANC_SEUIL,
    DEFAULT_FUSION_MODE,
    DEFAULT_FUSION_FORCE,
    DEFAULT_DEBLUR_UNSHARP_RADIUS,
    DEFAULT_DEBLUR_UNSHARP_PERCENT,
    DEFAULT_DEBLUR_UNSHARP_THRESHOLD,
    DEFAULT_DEBLUR_UNSHARP_PASSES,
    DEFAULT_AA_UNSHARP_RADIUS,
    DEFAULT_AA_UNSHARP_PERCENT,
    DEFAULT_AA_UNSHARP_THRESHOLD,
    DEFAULT_AA_UNSHARP_PASSES,
)
from .forerunner_floors import FORERUNNER_FLOOR_GRAYS


def charger_image(chemin: str) -> Image.Image:
    """
    Charge une image depuis un chemin de fichier.
    
    Paramètres :
        chemin: Chemin vers le fichier image
        
    Renvoie :
        Image PIL chargée
    """
    try:
        return Image.open(chemin)
    except Exception as e:
        from ui.logger import ProcessingError
        raise ProcessingError(f"Impossible de charger l'image {chemin}: {e}")


# Modes d'adaptation de la texture métallique (fichier TM) au format de l'image de base / TB_q.
# anisotropic : comportement historique — redimensionnement indépendant largeur/hauteur (LANCZOS).
TM_FIT_ANISOTROPIC = "anisotropic"
TM_FIT_UNIFORM_COVER = "uniform_cover"
TM_FIT_TILE = "tile"
TM_FIT_TILE_APPROX_STRETCH = "tile_approx_stretch"


def prepare_metal_texture_rgba(tm: Image.Image, tw: int, th: int, mode: str = TM_FIT_ANISOTROPIC) -> Image.Image:
    """
    Adapte une texture métallique RGBA au canevas (tw×th).

    - **anisotropic** : étirement libre sur tout le cadre (LANCZOS), comme les versions précédentes.
    - **uniform_cover** : échelle uniforme pour couvrir le cadre, recadrage centré (tilable préservé localement).
    - **tile** : mosaïque par répétition sans déformation des tuiles ; rognage si le cadre n'est pas un multiple.
    - **tile_approx_stretch** : nombre entier de tuiles au plus proche dans chaque direction, puis étirement global
      jusqu'au cadre exact (tilable « presque » à l'échelle du fichier).
    """
    tm = tm.convert("RGBA")
    mw, mh = tm.size
    if mw <= 0 or mh <= 0 or tw <= 0 or th <= 0:
        return tm.resize((max(1, tw), max(1, th)), Image.Resampling.LANCZOS)
    if mw == tw and mh == th:
        return tm

    if mode == TM_FIT_UNIFORM_COVER:
        scale = max(tw / float(mw), th / float(mh))
        nw = max(1, int(round(mw * scale)))
        nh = max(1, int(round(mh * scale)))
        up = tm.resize((nw, nh), Image.Resampling.LANCZOS)
        left = max(0, (nw - tw) // 2)
        top = max(0, (nh - th) // 2)
        return up.crop((left, top, left + tw, top + th))

    if mode == TM_FIT_TILE:
        out = Image.new("RGBA", (tw, th))
        for y in range(0, th, mh):
            for x in range(0, tw, mw):
                out.paste(tm, (x, y))
        return out

    if mode == TM_FIT_TILE_APPROX_STRETCH:
        n_w = max(1, int(round(tw / float(mw))))
        n_h = max(1, int(round(th / float(mh))))
        cw, ch = n_w * mw, n_h * mh
        tiled = Image.new("RGBA", (cw, ch))
        for j in range(n_h):
            for i in range(n_w):
                tiled.paste(tm, (i * mw, j * mh))
        return tiled.resize((tw, th), Image.Resampling.LANCZOS)

    # TM_FIT_ANISOTROPIC ou valeur inconnue
    return tm.resize((tw, th), Image.Resampling.LANCZOS)


def detecter_flou(img: Image.Image) -> float:
    """
    Score de netteté composite sur la luminance (plus élevé = plus net).

    - **Variance globale du Laplacien** : sensible au flou qui lisse toute l'image
      (gaussien, bougé léger, manque de focus).
    - **|Laplacien| moyen sur les forts gradients** : conserve la sensibilité sur les
      **aplats + lignes de construction** (peu de pixels « contours » mais très nets) ;
      un flou étale les transitions donc ce terme s'effondre même si les aplats restent plats.

    Un dessin entièrement uniforme renvoie un score neutre (pas d'alerte « flou »).
    """
    L = np.asarray(img.convert("L"), dtype=np.float32)
    gx = ndimage.sobel(L, axis=1)
    gy = ndimage.sobel(L, axis=0)
    G = np.hypot(gx, gy)
    lap = ndimage.laplace(L)
    g_max = float(G.max())
    if g_max < 1e-6:
        return 100.0

    global_lap_var = float(np.var(lap))
    thr_g = float(np.percentile(G, 94))
    if thr_g < 1.0:
        thr_g = max(g_max * 0.06, 1.0)
    strong = G >= thr_g
    if not np.any(strong):
        strong = G >= max(g_max * 0.25, 1.0)
    if not np.any(strong):
        return max(global_lap_var, 30.0)

    edge_focus = float(np.mean(np.abs(lap[strong])))
    # Pondération : les aplats dominants rendent la variance globale faible ;
    # les contours portent alors le poids pour éviter les faux « très net » sur flou réel.
    w_var = 0.38
    return w_var * global_lap_var + edge_focus


def renforcer_netete_dessin_flou(
    img: Image.Image,
    radius: float = DEFAULT_DEBLUR_UNSHARP_RADIUS,
    percent: int = DEFAULT_DEBLUR_UNSHARP_PERCENT,
    threshold: int = DEFAULT_DEBLUR_UNSHARP_THRESHOLD,
    passes: int = DEFAULT_DEBLUR_UNSHARP_PASSES,
) -> Image.Image:
    """
    Renforce les transitions entre aplats (masque flou / UnsharpMask), sans toucher à l’alpha.

    Ciblée après détection de flou : resserre les zones mi-tons entre deux niveaux avant
    le recollage Forerunner.
    """
    rgba = img.convert("RGBA")
    r, g, b, a = rgba.split()

    def um(chan: Image.Image) -> Image.Image:
        out = chan
        for _ in range(max(1, int(passes))):
            out = out.filter(
                ImageFilter.UnsharpMask(
                    radius=float(radius),
                    percent=int(percent),
                    threshold=int(threshold),
                )
            )
        return out

    return Image.merge("RGBA", (um(r), um(g), um(b), a))


def trace_melange_antialias(
    img: Image.Image,
    ng_lo: float = 0.06,
    ng_hi: float = 0.45,
    gmax_floor: float = 8.0,
    frac_min: float = 0.021,
) -> bool:
    """
    Heuristique : beaucoup de pixels en « gradient modéré » (ni plat ni arête très dure),
    typique d’un tracé **avec** antialiasing (frange grise étroite), par opposition à des
    aplats à contours durs.
    """
    L = _luminance_2d_rgba(img)
    gx = ndimage.sobel(L, axis=1)
    gy = ndimage.sobel(L, axis=0)
    G = np.hypot(gx, gy)
    gmax = float(G.max())
    if gmax < gmax_floor:
        return False
    ng = G / gmax
    soft = (ng > ng_lo) & (ng < ng_hi)
    return float(np.mean(soft)) >= frac_min


def renforcer_netete_antialias_trace(
    img: Image.Image,
    radius: float = DEFAULT_AA_UNSHARP_RADIUS,
    percent: int = DEFAULT_AA_UNSHARP_PERCENT,
    threshold: int = DEFAULT_AA_UNSHARP_THRESHOLD,
    passes: int = DEFAULT_AA_UNSHARP_PASSES,
) -> Image.Image:
    """
    Ramène le **lissage local** des arêtes (antialiasing type Paint Shop : tracé adouci) :

    - Rayon **court** et seuil **0** sur le masque flou : agit sur 1–2 px autour des ruptures.
    - À enchaîner après le passage « flou global » ou seul si seules des transitions douces
      sont détectées.

    Ne modifie pas l’alpha.
    """
    rgba = img.convert("RGBA")
    r, g, b, a = rgba.split()

    def um_aa(chan: Image.Image) -> Image.Image:
        out = chan
        for _ in range(max(1, int(passes))):
            out = out.filter(
                ImageFilter.UnsharpMask(
                    radius=float(radius),
                    percent=int(percent),
                    threshold=int(threshold),
                )
            )
        return out

    return Image.merge("RGBA", (um_aa(r), um_aa(g), um_aa(b), a))


def renforcer_contours(img: Image.Image, methode: str = 'pillow') -> Image.Image:
    """
    Renforce les contours d'une image.
    
    Paramètres :
        img: Image PIL
        methode: Méthode à utiliser ('pillow' ou 'opencv' si disponible)
        
    Renvoie :
        Image avec contours renforcés
    """
    if methode == 'opencv':
        try:
            import cv2
            arr = np.asarray(img.convert('RGB'))
            # Unsharp mask avec OpenCV
            gaussian = cv2.GaussianBlur(arr, (0, 0), 2.0)
            unsharp = cv2.addWeighted(arr, 1.5, gaussian, -0.5, 0)
            return Image.fromarray(unsharp).convert(img.mode)
        except ImportError:
            ui_log("OpenCV non disponible, utilisation de la méthode Pillow", 'warning')
            methode = 'pillow'
    
    # Méthode Pillow par défaut
    if img.mode != 'RGB':
        img = img.convert('RGB')
    # Unsharp mask avec Pillow
    enhancer = ImageEnhance.Sharpness(img)
    sharpened = enhancer.enhance(2.0)
    return sharpened.convert(img.mode)


def appliquer_masque_blanc_vers_alpha(
    base: Image.Image,
    masque: Image.Image,
    seuil: int = DEFAULT_MASQUE_BLANC_SEUIL
) -> Image.Image:
    """
    Applique un masque où les pixels blancs du masque deviennent transparents, révélant l'image de base.
    Processus (équivalent GIMP "Couleur vers alpha" avec blanc) :
    1. Image de base A
    2. Créer un calque avec B (le masque)
    3. Appliquer "blanc vers alpha" sur B : calcule la distance de couleur RGB entre chaque pixel et le blanc
       - Transparency threshold = 0,000 : seule la couleur exacte devient totalement transparente
       - Opacity threshold = 1,000 : les couleurs très différentes restent totalement opaques
       - Transition graduelle entre les deux
    4. Fusionner B vers le bas sur A : là où B est transparent, on voit A ; là où B est opaque, on voit B
    5. Convertir en RGB (les zones transparentes affichent le calque du dessous)
    
    Paramètres :
        base: Image de base A
        masque: Image masque B (peut être en couleur ou niveaux de gris)
        seuil: Paramètre non utilisé (ignoré)
        
    Renvoie :
        Image en RGB (sans alpha), résultat de la fusion du masque sur la base
    """
    base = base.convert('RGBA')
    masque_img = masque.convert('RGB')
    
    # Couleur cible : blanc (255, 255, 255)
    target_color = np.array([255.0, 255.0, 255.0])
    
    # Convertir le masque en array RGB
    masque_arr = np.array(masque_img, dtype='float32')
    
    # Calculer la distance de couleur entre chaque pixel et le blanc
    # Distance euclidienne dans l'espace RGB, normalisée sur [0, 1]
    # distance = sqrt((R-255)² + (G-255)² + (B-255)²) / sqrt(3*255²)
    diff = masque_arr - target_color
    distance = np.sqrt(np.sum(diff ** 2, axis=2)) / np.sqrt(3 * 255.0 ** 2)
    
    # Avec Transparency threshold = 0,000 et Opacity threshold = 1,000 :
    # - distance = 0 (blanc pur) → alpha = 0 (transparent)
    # - distance = 1 (couleur très différente) → alpha = 255 (opaque)
    # - Transition linéaire entre les deux
    # Formule : alpha = distance * 255
    alpha_arr = (distance * 255.0).clip(0, 255).astype('uint8')
    
    # Appliquer le masque alpha sur B
    masque_rgba = np.zeros((masque_arr.shape[0], masque_arr.shape[1], 4), dtype='uint8')
    masque_rgba[:, :, :3] = masque_arr.astype('uint8')
    masque_rgba[:, :, 3] = alpha_arr
    masque_transparent = Image.fromarray(masque_rgba, 'RGBA')
    
    # Fusionner B (avec transparence) sur A : là où B est transparent, on voit A ; là où B est opaque, on voit B
    result_rgba = base.copy()
    result_rgba.paste(masque_transparent, mask=masque_transparent.split()[3])
    
    # Convertir en RGB : les zones transparentes affichent le calque du dessous (qui est la base)
    result_rgb = result_rgba.convert('RGB')
    
    return result_rgb


def appliquer_couleur_vers_alpha_direct(
    img: Image.Image,
    couleur_cible: str = 'blanc',
    seuil: int = DEFAULT_MASQUE_BLANC_SEUIL
) -> Image.Image:
    """
    Applique "Couleur vers alpha" directement sur une image (comme GIMP avec Mode > Remplacer).
    Comportement :
    - Blanc vers alpha : Alpha = image inversée (en niveaux de gris), RGB = noir
    - Noir vers alpha : Alpha = image (en niveaux de gris), RGB = blanc
    
    Paramètres :
        img: Image à traiter (RGB ou RGBA)
        couleur_cible: 'blanc' ou 'noir'
        seuil: Paramètre non utilisé (ignoré)
        
    Renvoie :
        Image en RGBA avec canal alpha
    """
    img_rgb = img.convert('RGB')
    img_gray = img_rgb.convert('L')  # Convertir en niveaux de gris pour l'alpha
    
    if couleur_cible == 'blanc':
        # Blanc vers alpha : Alpha = image inversée, RGB = noir (0, 0, 0)
        alpha_arr = np.array(Image.eval(img_gray, lambda x: 255 - x), dtype='uint8')
        rgb_arr = np.zeros((img_gray.size[1], img_gray.size[0], 3), dtype='uint8')
    elif couleur_cible == 'noir':
        # Noir vers alpha : Alpha = image, RGB = blanc (255, 255, 255)
        alpha_arr = np.array(img_gray, dtype='uint8')
        rgb_arr = np.full((img_gray.size[1], img_gray.size[0], 3), 255, dtype='uint8')
    else:
        raise ValueError(f"Couleur cible inconnue: {couleur_cible}")
    
    # Créer l'image RGBA
    img_rgba = np.zeros((img_gray.size[1], img_gray.size[0], 4), dtype='uint8')
    img_rgba[:, :, :3] = rgb_arr
    img_rgba[:, :, 3] = alpha_arr
    
    return Image.fromarray(img_rgba, 'RGBA')


def _luminance_flat_masked(img: Image.Image) -> np.ndarray:
    """Luminance 0..255 pour pixels non transparents (tout l'image si opaque)."""
    rgba = img.convert('RGBA')
    arr = np.asarray(rgba, dtype=np.uint8)
    a = arr[:, :, 3].astype(np.float32) / 255.0
    L = (
        0.299 * arr[:, :, 0].astype(np.float32)
        + 0.587 * arr[:, :, 1].astype(np.float32)
        + 0.114 * arr[:, :, 2].astype(np.float32)
    )
    mask = a >= 0.02
    if not np.any(mask):
        mask = np.ones(L.shape[:2], dtype=bool)
    return np.clip(L[mask], 0, 255).astype(np.uint8).ravel()


def infer_forerunner_floor_indices_from_image(
    img: Image.Image,
    merge_peak_distance: int = 22,
    min_peak_ratio: float = 0.001,
) -> List[int]:
    """
    Estime les indices de paliers Forerunner réellement présents (peu d'étages possibles),
    via pics du histogramme de luminance — évite d'inventer un gris entre deux aplats.

    Les transitions floues / anti-crénelage sont regroupées (merge_peak_distance en niveaux L).
    """
    Lv = _luminance_flat_masked(img)
    return _infer_forerunner_indices_from_luminance(
        Lv, merge_peak_distance=merge_peak_distance, min_peak_ratio=min_peak_ratio
    )


def _infer_forerunner_indices_histogram_peaks(
    Lv: np.ndarray,
    merge_peak_distance: int = 22,
    min_peak_ratio: float = 0.001,
) -> List[int]:
    """
    Infère les indices de paliers par pics sur l’histogramme de luminance.
    Les pics en L=0 / L=255 et le lissage 5×1 peuvent biaiser la détection ; le chemin principal
    utilise le vote par palette lorsqu’il est disponible.
    """
    hist, _ = np.histogram(Lv, bins=256, range=(0, 256))
    n = int(Lv.size)
    hmax = float(hist.max()) if hist.size else 0.0
    min_peak_height = max(
        24,
        int(min_peak_ratio * n),
        min(int(0.012 * hmax), int(0.004 * n)),
    )
    peak_bins: List[int] = []
    for i in range(0, 256):
        if hist[i] < min_peak_height:
            continue
        h_left = float(hist[i - 1]) if i > 0 else -1.0
        h_right = float(hist[i + 1]) if i < 255 else -1.0
        hi = float(hist[i])
        if i == 0:
            if hi >= h_right:
                peak_bins.append(i)
        elif i == 255:
            if hi >= h_left:
                peak_bins.append(i)
        elif hi >= h_left and hi >= h_right:
            peak_bins.append(i)
    levels = np.array(FORERUNNER_FLOOR_GRAYS, dtype=np.int16)
    if not peak_bins:
        mean_L = float(np.mean(Lv))
        j = int(np.abs(levels - mean_L).argmin())
        return [int(j)]
    peak_bins.sort()
    merged_centers: List[int] = []
    i = 0
    while i < len(peak_bins):
        j = i
        acc_w = 0.0
        acc_c = 0.0
        while j < len(peak_bins) and peak_bins[j] - peak_bins[i] <= merge_peak_distance:
            w = float(hist[peak_bins[j]])
            acc_w += w
            acc_c += w * float(peak_bins[j])
            j += 1
        merged_centers.append(int(round(acc_c / max(acc_w, 1e-9))))
        i = j
    found_idx = []
    for c in merged_centers:
        j = int(np.abs(levels - c).argmin())
        found_idx.append(j)
    return sorted(set(found_idx))


def _infer_forerunner_indices_from_luminance(
    Lv: np.ndarray,
    merge_peak_distance: int = 22,
    min_peak_ratio: float = 0.001,
) -> List[int]:
    """
    Paliers présents : d'abord **vote par plus proche gris Forerunner** (robuste pour
    gros aplats noir/blanc même flous + halos), puis analyse par histogramme si le vote est insuffisant.
    """
    if Lv.size == 0:
        return [0]
    levels = np.array(FORERUNNER_FLOOR_GRAYS, dtype=np.int16)
    L16 = np.clip(Lv.astype(np.int32), 0, 255)
    d = np.abs(L16[:, None] - levels[None, :])
    nearest = d.argmin(axis=1).astype(np.int32)
    counts = np.bincount(nearest, minlength=len(levels)).astype(np.float64)
    n = int(Lv.size)
    maxc = float(counts.max())
    if maxc <= 0:
        return [0]

    abs_min = max(24.0, 0.015 * n)
    rel_dom = 0.03 * maxc
    thresh_count = max(abs_min, rel_dom)
    voted = [i for i in range(len(levels)) if counts[i] >= thresh_count]
    if voted:
        return sorted(voted)

    return _infer_forerunner_indices_histogram_peaks(
        Lv, merge_peak_distance=merge_peak_distance, min_peak_ratio=min_peak_ratio
    )


def _luminance_2d_rgba(img: Image.Image) -> np.ndarray:
    rgba = img.convert("RGBA")
    arr = np.asarray(rgba, dtype=np.uint8)
    return (
        0.299 * arr[:, :, 0].astype(np.float32)
        + 0.587 * arr[:, :, 1].astype(np.float32)
        + 0.114 * arr[:, :, 2].astype(np.float32)
    )


def refine_forerunner_floor_indices_autoclean_once(
    img: Image.Image,
    cands: List[int],
    *,
    thin_ratio_threshold: float = 0.7,
    max_thin_bbox_side: int = 2,
    gradient_sandwich_threshold: float = 0.55,
) -> List[int]:
    """
    Retire des candidats d'étages :
    - « traits parasites » : si ≥ thin_ratio_threshold des pixels d'un étage appartiennent
      à des composantes connexes de boîte englobante min(l,h) ≤ max_thin_bbox_side ;
    - « palier dans un dégradé » : étage strictement entre deux autres candidats dont la majorité
      des pixels touchent à la fois un voisin plus sombre et un plus clair (zone de transition).
    """
    cands = sorted({int(i) for i in cands})
    if len(cands) <= 1:
        return cands
    L = _luminance_2d_rgba(img)
    h, w = L.shape
    levels = np.array([FORERUNNER_FLOOR_GRAYS[i] for i in cands], dtype=np.float32)
    pos = np.abs(L[:, :, np.newaxis] - levels[np.newaxis, np.newaxis, :]).argmin(axis=2)
    idx_map = np.array(cands, dtype=np.int16)
    floormap = idx_map[pos]
    fp_pos = np.pad(pos, 1, mode="edge")

    struct8 = np.ones((3, 3), dtype=bool)
    to_remove: set[int] = set()

    total_px = h * w
    for k in cands:
        mask = floormap == k
        total = int(mask.sum())
        if total == 0:
            to_remove.add(k)
            continue

        lows = [x for x in cands if x < k]
        highs = [x for x in cands if x > k]
        if lows and highs and total / total_px < 0.028:
            to_remove.add(k)
            continue

        labeled, nlab = ndimage.label(mask, structure=struct8)
        thin_px = 0
        for lid in range(1, nlab + 1):
            comp = labeled == lid
            ys, xs = np.where(comp)
            bh = int(ys.max() - ys.min() + 1)
            bw = int(xs.max() - xs.min() + 1)
            if min(bh, bw) <= max_thin_bbox_side:
                thin_px += int(comp.sum())
        if thin_px / total >= thin_ratio_threshold:
            to_remove.add(k)
            continue

        if lows and highs:
            low_m = np.zeros(len(cands), dtype=bool)
            high_m = np.zeros(len(cands), dtype=bool)
            for i, c in enumerate(cands):
                low_m[i] = c in lows
                high_m[i] = c in highs
            is_low_neigh = np.zeros((h, w), dtype=bool)
            is_high_neigh = np.zeros((h, w), dtype=bool)
            for du in (-1, 0, 1):
                for dv in (-1, 0, 1):
                    if du == 0 and dv == 0:
                        continue
                    nb = fp_pos[1 + du : 1 + du + h, 1 + dv : 1 + dv + w]
                    is_low_neigh |= low_m[nb]
                    is_high_neigh |= high_m[nb]
            sandwich = mask & is_low_neigh & is_high_neigh
            if int(sandwich.sum()) / total >= gradient_sandwich_threshold:
                to_remove.add(k)

    out = [x for x in cands if x not in to_remove]
    if not out:
        areas = [(k, int((floormap == k).sum())) for k in cands]
        areas.sort(key=lambda t: -t[1])
        out = [areas[0][0]]
    return sorted(out)


def _mostly_linear_gradient_luminance(L: np.ndarray, r2_min: float = 0.94) -> bool:
    """
    Détecte un dégradé quasi linéaire (L expliqué à > r2_min par x seul ou y seul).
    Les aplats en bandes (même 2–3 niveaux) ont un R² plus bas ; on ne les confond pas.
    """
    h, w = L.shape
    if h < 8 or w < 8:
        return False
    lv = L.ravel().astype(np.float64)
    xv = np.tile(np.arange(w, dtype=np.float64), h)
    yv = np.repeat(np.arange(h, dtype=np.float64), w)

    def r2_line(coord: np.ndarray) -> float:
        c = np.polyfit(coord, lv, 1)
        pred = np.polyval(c, coord)
        ss = float(((lv - pred) ** 2).sum())
        st = float(((lv - lv.mean()) ** 2).sum())
        return 1.0 - ss / st if st > 1e-9 else 0.0

    return max(r2_line(xv), r2_line(yv)) >= r2_min


def infer_snap_indices_for_image(img: Image.Image) -> List[int]:
    """
    Paliers Forerunner pour snap sous-ensemble : histogramme puis retrait itératif
    des étages parasites (flou / dégradés / traits fins).
    """
    L0 = _luminance_2d_rgba(img)
    base = infer_forerunner_floor_indices_from_image(img)
    if not base:
        return [0]
    if _mostly_linear_gradient_luminance(L0) and len(base) >= 2:
        levels = np.array(FORERUNNER_FLOOR_GRAYS, dtype=np.float32)
        i_lo = int(np.abs(levels - float(L0.min())).argmin())
        i_hi = int(np.abs(levels - float(L0.max())).argmin())
        return sorted({i_lo, i_hi})
    cands = list(base)
    for _ in range(len(FORERUNNER_FLOOR_GRAYS) + 2):
        nxt = refine_forerunner_floor_indices_autoclean_once(img, cands)
        if nxt == cands:
            break
        cands = nxt
    return cands


def detect_used_forerunner_floor_indices(
    img: Image.Image,
    tolerance: int = 28,
) -> List[int]:
    """
    Indices d'étages Forerunner utilisés dans l'image (interface, limitation des blocs).

    Inférence + nettoyage auto des étages parasites ; retombe sur les valeurs uniques
    si l'histogramme ne donne rien.
    """
    inferred = infer_snap_indices_for_image(img)
    if inferred:
        return inferred
    L = np.asarray(img.convert('L'))
    unq = np.unique(L)
    levels = np.array(FORERUNNER_FLOOR_GRAYS, dtype=np.int16)
    found = set()
    for v in unq:
        d = np.abs(levels - int(v))
        j = int(d.argmin())
        if int(d[j]) <= tolerance:
            found.add(j)
    return sorted(found)


def forerunner_floor_indices_present_in_image(img: Image.Image) -> List[int]:
    """
    Paliers Forerunner réellement présents dans l'image : chaque gris de la palette
    qui apparaît au moins une fois (après snap, ce sont des aplats exacts).

    Un étage « manquant » du dessin n'a aucun pixel à ce niveau : il est exclu,
    ce qui permet de cibler l'ombre sur le palier inférieur encore utilisé.
    """
    L = np.unique(np.asarray(img.convert("L"), dtype=np.uint8))
    out: List[int] = []
    for i, g in enumerate(FORERUNNER_FLOOR_GRAYS):
        if np.any(L == int(g)):
            out.append(i)
    return out


def repair_phantom_intermediate_floors_after_snap(
    snapped: Image.Image,
    L_orig: np.ndarray,
    *,
    max_component_area: Optional[int] = None,
) -> Image.Image:
    """
    Atténue les « faux » paliers entre deux étages non voisins (ex. trait fin d’étage 1
    entre un aplats 0 et 2) : dégradés, contours renforcés ou snap « plus proche » peuvent
    créer des pixels au gris intermédiaire alors que le dessin ne comporte pas cet étage.

    On repère les petites composantes connexes d’un indice *m* avec des voisins de paliers
    *ni* et *nj* tels que ni < m < nj et nj − ni ≥ 2 ; chaque pixel est réassigné au gris
    *ni* ou *nj* selon la luminance d’origine (avant snap) la plus proche.
    """
    arr = np.asarray(snapped.convert("RGBA"), dtype=np.uint8)
    h, w = arr.shape[:2]
    if L_orig.shape != (h, w):
        raise ValueError("L_orig doit avoir la même taille que l’image snapée")
    levels = np.array(FORERUNNER_FLOOR_GRAYS, dtype=np.int16)
    nlev = len(levels)
    if max_component_area is None:
        max_component_area = max(96, min(4096, (h * w) // 3000))
    struct8 = np.ones((3, 3), dtype=bool)
    out = arr.copy()
    L_guide = L_orig.astype(np.float64, copy=False)

    for m in range(1, nlev - 1):
        L_snapped = out[:, :, 0].astype(np.int16)
        idx_map = np.abs(L_snapped[:, :, np.newaxis] - levels[np.newaxis, np.newaxis, :]).argmin(
            axis=2
        ).astype(np.int16)
        mask_m = idx_map == m
        if not np.any(mask_m):
            continue
        labeled, ncomp = ndimage.label(mask_m, structure=struct8)
        for comp_id in range(1, ncomp + 1):
            comp = labeled == comp_id
            if int(comp.sum()) > max_component_area:
                continue
            dil = ndimage.binary_dilation(comp, structure=struct8)
            border = dil & ~comp
            if not np.any(border):
                continue
            neigh_idx = np.unique(idx_map[border])
            neigh_idx = neigh_idx[neigh_idx != m]
            if neigh_idx.size < 2:
                continue
            ni = int(neigh_idx.min())
            nj = int(neigh_idx.max())
            if nj - ni < 2:
                continue
            if not (ni < m < nj):
                continue
            gray_i = float(levels[ni])
            gray_j = float(levels[nj])
            ys, xs = np.where(comp)
            lo = L_guide[ys, xs]
            pick_i = np.abs(lo - gray_i) <= np.abs(lo - gray_j)
            gi = int(levels[ni])
            gj = int(levels[nj])
            gv = np.where(pick_i, gi, gj).astype(np.uint8)
            out[ys, xs, 0] = gv
            out[ys, xs, 1] = gv
            out[ys, xs, 2] = gv

    return Image.fromarray(out, "RGBA")


def snap_luminance_to_forerunner_floors(img: Image.Image) -> Image.Image:
    """
    Recolle chaque pixel sur le gris d'étage Forerunner le plus proche (canaux RGB identiques).
    Préserve le canal alpha si présent.
    """
    rgba = img.convert('RGBA')
    arr = np.asarray(rgba, dtype=np.uint8)
    L = (
        0.299 * arr[:, :, 0].astype(np.float32)
        + 0.587 * arr[:, :, 1].astype(np.float32)
        + 0.114 * arr[:, :, 2].astype(np.float32)
    )
    levels = np.array(FORERUNNER_FLOOR_GRAYS, dtype=np.float32)
    idx = np.abs(L[:, :, np.newaxis] - levels[np.newaxis, np.newaxis, :]).argmin(axis=2)
    new_L = levels[idx].astype(np.uint8)
    out = arr.copy()
    out[:, :, 0] = new_L
    out[:, :, 1] = new_L
    out[:, :, 2] = new_L
    snapped = Image.fromarray(out, 'RGBA')
    return repair_phantom_intermediate_floors_after_snap(snapped, L)


def snap_luminance_to_forerunner_floors_subset(
    img: Image.Image,
    floor_indices: Sequence[int],
) -> Image.Image:
    """
    Comme snap_luminance_to_forerunner_floors, mais chaque pixel est projeté uniquement
    sur les gris des étages listés (ex. seulement 0, 102 et 255 si indices {0,2,5}).
    """
    idxs = sorted({int(i) for i in floor_indices if 0 <= int(i) < len(FORERUNNER_FLOOR_GRAYS)})
    if not idxs:
        return snap_luminance_to_forerunner_floors(img)
    rgba = img.convert('RGBA')
    arr = np.asarray(rgba, dtype=np.uint8)
    L = (
        0.299 * arr[:, :, 0].astype(np.float32)
        + 0.587 * arr[:, :, 1].astype(np.float32)
        + 0.114 * arr[:, :, 2].astype(np.float32)
    )
    allowed = np.array([FORERUNNER_FLOOR_GRAYS[i] for i in idxs], dtype=np.float32)
    mid = np.abs(L[:, :, np.newaxis] - allowed[np.newaxis, np.newaxis, :]).argmin(axis=2)
    new_L = allowed[mid].astype(np.uint8)
    out = arr.copy()
    out[:, :, 0] = new_L
    out[:, :, 1] = new_L
    out[:, :, 2] = new_L
    snapped = Image.fromarray(out, 'RGBA')
    return repair_phantom_intermediate_floors_after_snap(snapped, L)


def visualiser_masque_gris_etage(L_tb: np.ndarray, gray_cible: int) -> Image.Image:
    """
    Image RVB : blanc où L_tb == gray_cible (zone conservée par le masque dur), noir ailleurs.
    Utile pour déboguer les masques ombre / blanchiment par étage.
    """
    m = (L_tb == np.int16(gray_cible)).astype(np.uint8) * 255
    return Image.fromarray(m, 'L').convert('RGB')


def fusionner_grains(
    base_img: Image.Image,
    grain_img: Image.Image,
    mode: str = DEFAULT_FUSION_MODE,
    force: float = DEFAULT_FUSION_FORCE
) -> Image.Image:
    """
    Fusionne une image de grains avec une image de base.
    
    Utilise le mode "Fusion de grain" de GIMP par défaut (formule: E = I + M - 128).
    Référence: https://docs.gimp.org/2.6/fr/gimp-concepts-layer-modes.html
    
    Paramètres :
        base_img: Image de base (calque inférieur I)
        grain_img: Image de grains/texture (calque supérieur M)
        mode: Mode de fusion ('grain_merge', 'multiply', 'overlay', 'soft_light', etc.)
        force: Opacité de fusion (0.0 à 1.0). 1.0 = 100% opacité, 0.5 = 50% opacité
        
    Renvoie :
        Image fusionnée
    """
    base = base_img.convert('RGBA')
    grain = grain_img.convert('RGBA')
    
    # Redimensionner le grain si nécessaire
    if grain.size != base.size:
        grain = grain.resize(base.size, Image.Resampling.LANCZOS)
    
    if mode == 'grain_merge':
        # Mode "Fusion de grain" de GIMP
        # Formule: E = I + M - 128
        # où I = calque inférieur (base), M = calque supérieur (grain)
        base_arr = np.asarray(base.convert('RGB'), dtype='float32')
        grain_arr = np.asarray(grain.convert('RGB'), dtype='float32')
        grain_alpha = np.asarray(grain.split()[3], dtype='float32') / 255.0  # Alpha normalisé [0, 1]
        
        # Appliquer la formule: E = I + M - 128
        result_arr = base_arr + grain_arr - 128.0
        # Borner entre 0 et 255
        result_arr = result_arr.clip(0, 255)
        
        # Utiliser l'alpha du grain comme masque pour contrôler où appliquer la fusion
        # Là où alpha = 0 (transparent) → garder la base
        # Là où alpha = 1 (opaque) → appliquer la fusion
        alpha_mask = np.expand_dims(grain_alpha, axis=2)
        masked_result = base_arr * (1.0 - alpha_mask) + result_arr * alpha_mask
        
        # Appliquer l'opacité (force) : mélange entre base et résultat masqué
        # Opacité 50% = 50% résultat masqué + 50% base
        if force < 1.0:
            final_result = base_arr * (1.0 - force) + masked_result * force
        else:
            final_result = masked_result
        
        result = Image.fromarray(final_result.clip(0, 255).astype('uint8'), 'RGB').convert('RGBA')
    elif mode == 'multiply':
        # Multiplication simple
        result = ImageChops.multiply(base.convert('RGB'), grain.convert('RGB'))
        result = result.convert('RGBA')
        # Appliquer la force
        if force < 1.0:
            base_arr = np.asarray(base, dtype='float32')
            result_arr = np.asarray(result, dtype='float32')
            blended = base_arr * (1.0 - force) + result_arr * force
            result = Image.fromarray(blended.clip(0, 255).astype('uint8'), 'RGBA')
        else:
            # S'assurer que result est une copie modifiable
            result_arr = np.asarray(result, dtype='uint8')
            if not result_arr.flags['WRITEABLE']:
                result_arr = result_arr.copy()
            result = Image.fromarray(result_arr, 'RGBA')
    elif mode == 'overlay':
        # Overlay blend
        base_arr = np.asarray(base.convert('RGB'), dtype='float32') / 255.0
        grain_arr = np.asarray(grain.convert('RGB'), dtype='float32') / 255.0
        mask = base_arr < 0.5
        result_arr = np.where(
            mask,
            2 * base_arr * grain_arr,
            1 - 2 * (1 - base_arr) * (1 - grain_arr)
        )
        result = Image.fromarray((result_arr * 255).clip(0, 255).astype('uint8'), 'RGB')
        result = result.convert('RGBA')
        # Appliquer la force
        if force < 1.0:
            base_arr = np.asarray(base, dtype='float32')
            result_arr = np.asarray(result, dtype='float32')
            blended = base_arr * (1.0 - force) + result_arr * force
            result = Image.fromarray(blended.clip(0, 255).astype('uint8'), 'RGBA')
    else:
        # Mode par défaut: grain_merge
        base_arr = np.asarray(base.convert('RGB'), dtype='float32')
        grain_arr = np.asarray(grain.convert('RGB'), dtype='float32')
        result_arr = base_arr + grain_arr - 128.0
        result_arr = result_arr.clip(0, 255).astype('uint8')
        result = Image.fromarray(result_arr, 'RGB').convert('RGBA')
        if force < 1.0:
            base_rgb = np.asarray(base.convert('RGB'), dtype='float32')
            result_rgb = np.asarray(result.convert('RGB'), dtype='float32')
            blended = base_rgb * (1.0 - force) + result_rgb * force
            result = Image.fromarray(blended.clip(0, 255).astype('uint8'), 'RGB').convert('RGBA')
    
    # Préserver l'alpha de la base
    base_alpha = np.asarray(base.split()[3])
    result_arr = np.asarray(result, dtype='uint8')
    # Créer une copie modifiable si nécessaire
    if not result_arr.flags['WRITEABLE']:
        result_arr = result_arr.copy()
    result_arr[:, :, 3] = base_alpha
    return Image.fromarray(result_arr, 'RGBA')


def composite_grain_contributions_from_base(
    base_img: Image.Image,
    layers: List[Tuple[Image.Image, float]],
    mode: str = DEFAULT_FUSION_MODE,
) -> Image.Image:
    """
    Recompose plusieurs calques grain (ombres ou blanchiments, déjà en RVB) depuis une même base.

    Au lieu d'enchaîner ``fusion(base, c1)`` puis ``fusion(résultat, c2)`` (qui accumule
    les défauts aux joints entre étages), on somme en RVB les écarts
    ``fusionner_grains(base, calque) - base`` pour chaque contribution, puis on clamp.

    Les ombres sont composées depuis la base métal ; le blanchiment depuis l’image déjà fusionnée avec ces ombres.
    """
    if not layers:
        return base_img.convert('RGBA')
    base_rgb = np.asarray(base_img.convert('RGB'), dtype=np.float32)
    acc = np.zeros_like(base_rgb)
    base_rgba = base_img.convert('RGBA')
    for grain_img, force in layers:
        if force <= 0:
            continue
        merged = fusionner_grains(base_rgba, grain_img, mode=mode, force=force)
        m = np.asarray(merged.convert('RGB'), dtype=np.float32)
        acc += m - base_rgb
    out = np.clip(base_rgb + acc, 0, 255).astype(np.uint8)
    result = Image.fromarray(out, 'RGB').convert('RGBA')
    if base_img.mode == 'RGBA':
        result.putalpha(base_img.split()[3])
    return result


def quantifier(
    img: Image.Image,
    niveaux: int,
    palette: Optional[Image.Image] = None
) -> Image.Image:
    """
    Quantifie une image en N niveaux de couleurs.
    
    Paramètres :
        img: Image à quantifier
        niveaux: Nombre de niveaux de quantification
        palette: Palette optionnelle (Image PIL) pour quantification avec palette
        
    Renvoie :
        Image quantifiée
    """
    if niveaux <= 1:
        return img
    
    if palette is not None:
        # Quantification avec palette personnalisée
        from effets.palettes import apply_palette
        # Convertir la palette en liste de tuples RGB
        pal_arr = np.asarray(palette.convert('RGB'))
        if pal_arr.ndim == 2:
            palette_list = [tuple(pal_arr[i]) for i in range(len(pal_arr))]
        else:
            # Palette 2D, prendre la première ligne
            palette_list = [tuple(pal_arr[0, i]) for i in range(pal_arr.shape[1])]
        return apply_palette(img, palette_list)
    
    # Quantification uniforme par canal
    img_rgb = img.convert('RGB')
    arr = (np.asarray(img_rgb).astype('float32') / 255.0)
    # Quantification uniforme
    arr_q = (arr * (niveaux - 1)).round() / (niveaux - 1)
    arr_q = (arr_q * 255.0).clip(0, 255).astype('uint8')
    result = Image.fromarray(arr_q, 'RGB')
    
    # Préserver le mode original si possible
    if img.mode == 'RGBA':
        alpha = img.split()[3]
        result = result.convert('RGBA')
        result.putalpha(alpha)
    elif img.mode != 'RGB':
        result = result.convert(img.mode)
    
    return result


def exporter(
    img: Image.Image,
    chemin: str,
    preserve_input_format: bool = True,
    original_format: Optional[str] = None
) -> str:
    """
    Exporte une image en préservant le format d'entrée par défaut.
    
    Paramètres :
        img: Image à exporter
        chemin: Chemin de destination
        preserve_input_format: Si True, préserve le format d'origine
        original_format: Format original (ex: 'PNG', 'JPEG')
        
    Renvoie :
        Chemin du fichier exporté
    """
    # Déterminer le format de sortie
    if preserve_input_format and original_format:
        format_out = original_format.upper()
    else:
        # Déduire du chemin
        ext = chemin.lower().split('.')[-1] if '.' in chemin else 'png'
        format_map = {
            'png': 'PNG',
            'jpg': 'JPEG',
            'jpeg': 'JPEG',
            'tiff': 'TIFF',
            'tif': 'TIFF',
            'bmp': 'BMP'
        }
        format_out = format_map.get(ext, 'PNG')
    
    # Convertir si nécessaire
    if format_out == 'JPEG' and img.mode in ('RGBA', 'LA', 'P'):
        # JPEG ne supporte pas la transparence
        img = img.convert('RGB')
    elif format_out in ('PNG', 'TIFF') and img.mode not in ('RGBA', 'RGB', 'L', 'LA'):
        img = img.convert('RGBA')
    
    try:
        img.save(chemin, format=format_out)
        return chemin
    except Exception as e:
        from ui.logger import ProcessingError
        raise ProcessingError(f"Impossible d'exporter l'image vers {chemin}: {e}")
