"""
Pipeline de masques Forerunner (copies N/B et dérivés pour ombres / blanchiments).

Ancrage : toutes les constructions ci-dessous prennent l’**image de base alignée sur les paliers Forerunner**
comme champ de pixels ; les exclusions de paliers et le remap des indices suivent **opts**
via ``apply_idx_remap_for_pipeline(..., opts)`` lorsque l’appelant fournit ``idx`` ainsi dérivé.
Les paramètres de glow / seuils par étage relèvent des dicts passés depuis ``opts['floors'][...]``
(côté ``traitement``).

Chaîne **ombre** (étage porteur → palier récepteur en dessous) :
  masque N/B source → glow éventuel sur l’image à paliers → inversion RVB → pondération douce sur le palier récepteur.

Chaîne **blanchiment** (palier courant) :
  masque N/B → glow ; si grain blanchiment actif → préremplissage des étages sous le palier
  (``analyze_grain.remplir_etages_dessous_reference_affleurement``, cliché ``blanch_02b_*``) ;
  puis grain « plein cadre » (spread) ; collage par palier (aplats sous le palier / noir au-dessus / dégradé sur le palier) ;
  pondération douce sur ce palier ; grain affleurement éventuel en calque séparé ; composite final ;
  puis noir→α à la fusion comme pour le calque blanchiment standard du moteur.
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np
from PIL import Image
from scipy import ndimage

from .forerunner_floors import FORERUNNER_FLOOR_GRAYS

from .glow import apply_glow
from .spread_grain import apply_spread_grain_fast

from . import analyze_grain as _analyze_grain_prep

# Adoucissement des bords de masque palier (évite contours durs après glow / grain).
_SOFT_FLOOR_MASK_SIGMA = 1.2


def _soft_weight_for_floor_pixels(
    idx: np.ndarray,
    floor_idx: int,
    sigma: float = _SOFT_FLOOR_MASK_SIGMA,
) -> np.ndarray:
    """
    Poids 0–1 sur le palier ``floor_idx`` : le flou gaussien du masque dur est annulé
    hors palier (pas de halo sur les autres étages), mais les pixels du bord du palier
    gardent un mélange progressif au lieu d'un masque binaire.
    """
    m = (idx == floor_idx).astype(np.float32)
    if not np.any(m):
        return m
    a = ndimage.gaussian_filter(m, sigma=sigma, mode="nearest")
    a = np.where(m > 0.5, a, 0.0)
    return np.clip(a, 0.0, 1.0)


def stable_spread_grain_seed_for_floor(
    grain_seed_base: int,
    k: int,
    spread_x: float,
    spread_y: float,
    tag: str,
) -> int:
    """
    Graine déterministe par étage et par type de grain (blanchiment / affleurement).
    Stabilise le motif sur les étages non modifiés lorsqu’un seul palier change.
    """
    h = (int(grain_seed_base) + k * 1000033) & 0xFFFFFFFF
    h = (h + int(spread_x * 1000) * 9176) & 0xFFFFFFFF
    h = (h + int(spread_y * 1000) * 31337) & 0xFFFFFFFF
    h = (h + (0xB100 if tag == "blanch" else 0xAFF1)) & 0xFFFFFFFF
    return int(h)


def luminance_to_floor_index_map(L: np.ndarray, grays: Tuple[int, ...] = FORERUNNER_FLOOR_GRAYS) -> np.ndarray:
    """Pour chaque pixel, indice d'étage 0..n-1 (plus proche gris palette)."""
    levels = np.array(grays, dtype=np.int16)
    L16 = L.astype(np.int16)
    d = np.abs(L16[..., None] - levels[None, None, :])
    return d.argmin(axis=2).astype(np.int16)


def _floor_processed_in_surface_pipeline(opts: dict, d: int) -> bool:
    """True si l'étage d est parcouru dans la boucle surfaces (activé et dans process_floor_indices si défini)."""
    floors = opts.get('floors') or []
    if d < 0 or d >= len(floors):
        return True
    # Ancres pour apply_idx_remap : le fond (0) et le blanc (5) ne sont jamais « exclus » ici,
    # sinon si process_floor_indices omet 0, tout idx==0 serait fusionné vers l'étage 1.
    if d in (0, 5):
        return True
    po = opts.get('process_floor_indices')
    if po is not None and d not in po:
        return False
    return bool(floors[d].get('enabled', True))


def apply_idx_remap_for_pipeline(
    idx: np.ndarray,
    opts: dict,
    n_floors: int = 6,
) -> np.ndarray:
    """
    Réassigne chaque indice d'étage exclu du pipeline vers un palier **traité** voisin.

    Priorité au palier **inférieur** (ex. étage 4 désactivé → fusion vers 3) : ainsi les
    comparaisons ``idx < k`` / ``idx >= k`` du blanchiment restent cohérentes (une fusion 4→5
    faisait passer les pixels du gris 4 de « sous k=5 » à « palier 5 », ce qui annulait le
    blanchiment de l'étage 5).

    Utilisé pour adoucir les frontières masques quand un étage intermédiaire ne contribue pas.
    """
    processed = [_floor_processed_in_surface_pipeline(opts, d) for d in range(n_floors)]
    remap = np.zeros(n_floors, dtype=np.int16)

    def _target_for_disabled(d: int) -> int:
        for j in range(d - 1, -1, -1):
            if processed[j]:
                return j
        for j in range(d + 1, n_floors):
            if processed[j]:
                return j
        return d

    for d in range(n_floors):
        if processed[d]:
            remap[d] = d
        else:
            remap[d] = _target_for_disabled(d)

    ii = np.asarray(idx, dtype=np.intp)
    return remap[ii].astype(np.int16)


def clamp_shadow_receiver_floor(k_carrier: int, k_receiver: int, n_floors: int) -> int:
    """Récepteur valide : strictement sous le porteur, [0, n_floors-1]."""
    lo = 0
    hi = min(k_carrier - 1, n_floors - 1)
    if hi < 0:
        return 0
    return max(lo, min(hi, k_receiver))


def resolve_shadow_receiver_index(
    k_carrier: int,
    n_floors: int,
    used_floor_indices: Optional[Sequence[int]] = None,
) -> int:
    """
    Palier récepteur sous l'étage porteur *k_carrier* : plus grand indice **présent
    dans le dessin** et strictement inférieur à *k_carrier* ; si aucun, *k−1* (clampé).
    """
    if used_floor_indices:
        below = [int(i) for i in used_floor_indices if int(i) < k_carrier]
        r = max(below) if below else k_carrier - 1
    else:
        r = k_carrier - 1
    return clamp_shadow_receiver_floor(k_carrier, r, n_floors)


def manual_inok_rgb_from_tb_q(
    TB_q: Image.Image,
    k_carrier: int,
    k_receiver: int,
    glow_in: Dict[str, Any],
    allow_conversion: bool,
    glow_on_tb: bool,
    idx: Optional[np.ndarray] = None,
) -> Tuple[Image.Image, Dict[str, Image.Image]]:
    """
    Chaîne ombre manuelle (avant blanc→α) : dégradé d’ombre sur le palier récepteur,
    avec pondération douce aux bords (pas de masque dur sur le récepteur après inversion).

    Entrée pixel : **image à paliers** ; **glow_in** / **glow_on_tb** viennent typiquement d’``opts['floors'][...]``.
    Si ``idx`` est fourni, il doit refléter le remap pipeline (``apply_idx_remap_for_pipeline(..., opts)``).

    Retourne (image_rgb, dict étapes optionnelles pour debug).
    """
    L_tb = np.asarray(TB_q.convert('L'), dtype=np.int16)
    h, w = L_tb.shape
    if idx is None:
        idx = luminance_to_floor_index_map(L_tb)
    n = len(FORERUNNER_FLOOR_GRAYS)
    r = clamp_shadow_receiver_floor(k_carrier, k_receiver, n)
    steps: Dict[str, Image.Image] = {}

    # Masque N/B source (blanc si étage >= porteur)
    bw = np.where(idx >= k_carrier, 255, 0).astype(np.uint8)
    im32 = Image.fromarray(bw, mode="L").convert("RGB").convert("RGBA")
    steps["ombre_01_masque_NB"] = im32.convert("RGB")

    # Glow éventuel sur l’image à paliers (si glow_on_tb)
    if glow_on_tb:
        im_gl = apply_glow(
            im32,
            radius=int(glow_in.get("radius", 25)),
            threshold=int(glow_in.get("threshold", 127)),
            intensity=float(glow_in.get("intensity", 2.5)),
            allow_conversion=allow_conversion,
        )
    else:
        im_gl = im32
    steps["ombre_02_apres_glow"] = im_gl.convert("RGB")

    # Inversion RVB
    arr = np.asarray(im_gl.convert("RGB"), dtype=np.uint8)
    inv = 255 - arr
    steps["ombre_03_inversion"] = Image.fromarray(inv, "RGB")

    # Contribution ombre sur le palier récepteur (poids doux aux bords,
    # pas de troncature dure avant fusion — évite des contours parasites aux joints).
    alpha = _soft_weight_for_floor_pixels(idx, r)
    inv_f = inv.astype(np.float32)
    out = inv_f * alpha[..., None] + 255.0 * (1.0 - alpha[..., None])
    out = np.clip(out, 0, 255).astype(np.uint8)
    out_img = Image.fromarray(out, "RGB")
    steps["ombre_04_palier_recepteur_adouci"] = out_img
    return out_img, steps


def _grain_affleurement_rgb(
    idx: np.ndarray,
    k: int,
    grain_affleurement_opts: Dict[str, Any],
    allow_conversion: bool,
) -> Tuple[np.ndarray, Dict[str, Image.Image]]:
    """
    Calque affleurement seul (indépendant du glow / blanchiment) :
    masque test (zones sous le palier) → spread → ne garder que le palier courant (noir ailleurs).
    Retourne une image RGB uint8 (h,w,3), noire hors du palier traité.
    """
    steps: Dict[str, Image.Image] = {}
    plane = np.where(idx < k, 255, 0).astype(np.uint8)
    pre = np.stack([plane, plane, plane], axis=-1)
    steps["affleur_01_masque_test"] = Image.fromarray(pre, mode="RGB")

    im_pre = Image.fromarray(pre, mode="RGB").convert("RGBA")
    im_gr = apply_spread_grain_fast(
        im_pre.copy(),
        spread_x=float(grain_affleurement_opts.get("spread_x", 2.0)),
        spread_y=float(grain_affleurement_opts.get("spread_y", 2.0)),
        seed=grain_affleurement_opts.get("seed"),
        allow_conversion=allow_conversion,
        edge_mask=False,
    )
    gr = np.asarray(im_gr.convert("RGB"), dtype=np.uint8)
    steps["affleur_02_apres_spread"] = Image.fromarray(gr, mode="RGB")

    wk = _soft_weight_for_floor_pixels(idx, k)
    post = (gr.astype(np.float32) * wk[..., None]).astype(np.uint8)
    steps["affleur_03_isole_palier"] = Image.fromarray(post, mode="RGB")
    return post, steps


def manual_exok_rgb_from_tb_q(
    TB_q: Image.Image,
    k: int,
    glow_ex: Dict[str, Any],
    grain_blanchiment_opts: Dict[str, Any],
    grain_affleurement_opts: Dict[str, Any],
    allow_conversion: bool,
    glow_ex_active: bool,
    idx: Optional[np.ndarray] = None,
) -> Tuple[Image.Image, Dict[str, Image.Image]]:
    """
    Blanchiment : masque N/B → glow ; si grain actif, préremplissage sous-palier (``analyze_grain``) → grain plein cadre
    → collage par palier → adoucissement des bords sur le palier courant.

    Grain affleurement : procédure séparée (`_grain_affleurement_rgb`), puis fusion avec le blanchiment adouci.

    **Image alignée sur les paliers** en entrée ; glow / grains pilotés par les dicts passés depuis ``opts['floors'][...]`` (côté appelant).
    **idx** : carte d'étage (``apply_idx_remap_for_pipeline(..., opts)`` si étages exclus du pipeline).
    """
    L_tb = np.asarray(TB_q.convert("L"), dtype=np.int16)
    h, w = L_tb.shape
    if idx is None:
        idx = luminance_to_floor_index_map(L_tb)
    steps: Dict[str, Image.Image] = {}

    # Masque N/B (blanc si étage < palier courant)
    bw = np.where(idx < k, 255, 0).astype(np.uint8)
    im41 = Image.fromarray(bw, mode="L").convert("RGB").convert("RGBA")
    steps["blanch_01_masque_NB"] = im41.convert("RGB")

    # Glow sur le masque
    if glow_ex_active:
        im42 = apply_glow(
            im41,
            radius=int(glow_ex.get("radius", 25)),
            threshold=int(glow_ex.get("threshold", 127)),
            intensity=float(glow_ex.get("intensity", 2.5)),
            allow_conversion=allow_conversion,
        )
    else:
        im42 = im41
    steps["blanch_02_apres_glow"] = im42.convert("RGB")

    im = im42
    if grain_blanchiment_opts.get("enabled", False):
        arr_prep = _analyze_grain_prep.remplir_etages_dessous_reference_affleurement(
            np.asarray(im42.convert("RGB"), dtype=np.uint8),
            idx,
            k,
        )
        steps["blanch_02b_prepare_sous_palier_avant_grain"] = Image.fromarray(
            arr_prep, mode="RGB"
        )
        im = Image.fromarray(arr_prep, mode="RGB").convert("RGBA")
        # Grain sur tout le calque : ne pas tronquer au palier k avant dispersion (sinon arêtes dures).
        im = apply_spread_grain_fast(
            im.copy(),
            spread_x=float(grain_blanchiment_opts.get("spread_x", 2.0)),
            spread_y=float(grain_blanchiment_opts.get("spread_y", 2.0)),
            seed=grain_blanchiment_opts.get("seed"),
            allow_conversion=allow_conversion,
            edge_mask=False,
            support_mask=None,
        )
    steps["blanch_03_grain_plein_cadre"] = im.convert("RGB")

    # Collage par palier : sous le palier → teinte du 1er pixel affleurement sur l’image grainée ; au-dessus → noir ; sur le palier → image grainée
    arr42 = np.asarray(im.convert("RGB"), dtype=np.uint8)
    below = idx < k
    above = idx > k
    on_k = idx == k
    if np.any(on_k):
        first = int(np.flatnonzero(on_k.ravel())[0])
        y0, x0 = np.unravel_index(first, idx.shape)
        fill_rgb = arr42[y0, x0]
    else:
        fill_rgb = np.zeros(3, dtype=np.uint8)
    isolated = arr42.copy()
    isolated[below] = fill_rgb
    isolated[above] = 0
    im42 = Image.fromarray(isolated.astype(np.uint8), "RGB").convert("RGBA")
    steps["blanch_04_normalisation_paliers"] = im42.convert("RGB")

    # Pondération douce sur le palier courant (évite contour net au joint d'étages)
    arr = np.asarray(im42.convert("RGB"), dtype=np.uint8)
    wk = _soft_weight_for_floor_pixels(idx, k)
    masked = (arr.astype(np.float32) * wk[..., None]).astype(np.uint8)
    out_blanch = np.asarray(
        Image.fromarray(masked.astype(np.uint8), "RGB"),
        dtype=np.uint8,
    )
    steps["blanch_05_adoucissement_bords"] = Image.fromarray(out_blanch, mode="RGB")

    if grain_affleurement_opts.get("enabled", False):
        aff_post, aff_steps = _grain_affleurement_rgb(
            idx, k, grain_affleurement_opts, allow_conversion
        )
        for name, sim in aff_steps.items():
            steps[name] = sim
        # Combine blanchiment + affleurement : max RVB, pondéré par opacité (0 = blanchiment seul, 1 = max plein)
        o = float(grain_affleurement_opts.get("opacity", 1.0))
        o = max(0.0, min(1.0, o))
        bl = out_blanch.astype(np.float32)
        mx = np.maximum(bl, aff_post.astype(np.float32))
        out_rgb = np.clip((1.0 - o) * bl + o * mx, 0, 255).round().astype(np.uint8)
        out_img = Image.fromarray(out_rgb, mode="RGB")
    else:
        out_img = Image.fromarray(out_blanch, mode="RGB")

    steps["blanch_06_composite_final"] = out_img.copy()
    return out_img, steps
