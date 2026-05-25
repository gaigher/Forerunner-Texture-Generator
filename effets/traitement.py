"""
Pipeline de traitement principal pour Texture Halo.

Les étapes surfaces / masques s’enchaînent **à partir de l’image de base** du dessin (quantifiée puis alignée sur les paliers
Forerunner) et du dictionnaire **opts** (réglages utilisateur, TM, fusion, périmètre, graines).
"""
from PIL import Image
import hashlib
import numpy as np
from typing import Dict, Optional, Callable, Tuple, Any, List
from pathlib import Path

from .utilitaires import (
    charger_image,
    prepare_metal_texture_rgba,
    TM_FIT_ANISOTROPIC,
    detecter_flou,
    renforcer_contours,
    renforcer_netete_dessin_flou,
    renforcer_netete_antialias_trace,
    trace_melange_antialias,
    quantifier,
    appliquer_masque_blanc_vers_alpha,
    appliquer_couleur_vers_alpha_direct,
    fusionner_grains,
    composite_grain_contributions_from_base,
    exporter,
    snap_luminance_to_forerunner_floors,
    snap_luminance_to_forerunner_floors_subset,
    infer_snap_indices_for_image,
    forerunner_floor_indices_present_in_image,
    visualiser_masque_gris_etage,
)
from .glow import apply_glow
from .spread_grain import apply_spread_grain_fast
from ui.logger import ui_log
from ui.settings import (
    DEFAULT_SEUIL_FLOU,
    DEFAULT_AUTO_SHARPEN_BLUR,
    DEFAULT_DEBLUR_UNSHARP_RADIUS,
    DEFAULT_DEBLUR_UNSHARP_PERCENT,
    DEFAULT_DEBLUR_UNSHARP_THRESHOLD,
    DEFAULT_DEBLUR_UNSHARP_PASSES,
    DEFAULT_AUTO_ANTIALIAS_PASS,
    DEFAULT_AA_UNSHARP_RADIUS,
    DEFAULT_AA_UNSHARP_PERCENT,
    DEFAULT_AA_UNSHARP_THRESHOLD,
    DEFAULT_AA_UNSHARP_PASSES,
    DEFAULT_CONTOUR_METHODE,
    DEFAULT_SNAP_USE_FULL_PALETTE,
    DEFAULT_FORCE_FORERUNNER_SNAP,
    DEFAULT_SNAP_ONLY_SKIP_CONTOUR,
    DEFAULT_FUSION_FORCE_INM,
    DEFAULT_FUSION_FORCE_INEXM,
    UI_PROGRESS_FUSION_BLANCHIMENTS,
    UI_PROGRESS_FUSION_OMBRES,
    UI_PROGRESS_CALQUE_BLANCHIMENT_FUSION_METAL,
    UI_PROGRESS_CALQUE_BLANCHIMENT_MASQUE,
    UI_PROGRESS_CALQUE_OMBRE_FUSION_METAL,
    UI_PROGRESS_CALQUE_OMBRE_MASQUE,
    UI_GRAPHE_TITRE_IMAGE_BASE_AVANT_PALIERS,
    UI_GRAPHE_TITRE_IMAGE_BASE_PALIERS,
    UI_GRAPHE_TITRE_IMAGE_BASE_PREP,
    DEFAULT_GLOW_EX_RADIUS,
    DEFAULT_GLOW_EX_THRESHOLD,
    DEFAULT_GLOW_EX_INTENSITY,
    DEFAULT_GLOW_IN_RADIUS,
    DEFAULT_GLOW_IN_THRESHOLD,
    DEFAULT_GLOW_IN_INTENSITY
)
from .forerunner_floors import FORERUNNER_FLOOR_GRAYS
from .masques_manuels import (
    apply_idx_remap_for_pipeline,
    manual_exok_rgb_from_tb_q,
    manual_inok_rgb_from_tb_q,
    luminance_to_floor_index_map,
    resolve_shadow_receiver_index,
    stable_spread_grain_seed_for_floor,
)


def _appeler_progress(progress_cb: Optional[Callable], percent: int, message: str) -> None:
    """Appelle le rappel de progression s'il est défini."""
    if progress_cb:
        try:
            progress_cb(percent, message)
        except Exception as e:
            ui_log(f"Erreur dans progress_cb: {e}", 'warning')


def surface_pipeline_global_key(opts: dict) -> tuple:
    """
    Empreinte des options communes au bloc surfaces (TM, fusion, sous-ensemble d'étages, grain seed).
    """
    tm = opts.get('tm')
    if tm:
        try:
            tm_t = str(Path(tm).resolve())
        except (OSError, ValueError):
            tm_t = str(tm)
    else:
        tm_t = None
    pi = opts.get('process_floor_indices')
    pi_t = tuple(sorted(int(i) for i in pi)) if pi is not None else None
    return (
        tm_t,
        str(opts.get('tm_fit_mode', TM_FIT_ANISOTROPIC)),
        bool(opts.get('allow_conversion', False)),
        str(opts.get('fusion_mode', 'grain_merge')),
        int(opts.get('masque_blanc_seuil', 250)),
        int(opts.get('masque_noir_seuil', 5)),
        int(opts.get('_grain_seed_base', 0)),
        pi_t,
        'surfaces_recomp_sum_delta',
    )


def _floors_enabled_mask(opts: dict) -> Tuple[bool, ...]:
    """True pour chaque étage 1..5 si « Traiter cet étage » est coché (invalide le cache calques si changement)."""
    floors: List[dict] = opts.get('floors') or []
    return tuple(
        bool(floors[i].get('enabled', True)) if i < len(floors) else True
        for i in range(1, 6)
    )


def _active_shadow_receiver_candidates(
    k_carrier: int,
    floors: List[dict],
    process_set: Optional[set],
) -> List[int]:
    """
    Indices de paliers autorisés comme récepteur pour un porteur k :
    strictement sous k, étage 0 toujours admis, étages >=1 uniquement si actifs
    (enabled=True et dans process_floor_indices si défini).
    """
    out: List[int] = []
    for i in range(max(0, int(k_carrier))):
        if i == 0:
            out.append(i)
            continue
        if process_set is not None and i not in process_set:
            continue
        if i < len(floors) and not floors[i].get('enabled', True):
            continue
        out.append(i)
    return out


def floor_surface_fingerprint(opts: dict, k: int) -> tuple:
    """
    Empreinte des paramètres d'un étage k (1..5) influençant _process_per_floor_surfaces.
    Étages hors process_floor_indices ou désactivés : empreinte stable (ignorés pour la sortie).

    Inclut le masque d'activation des étages 1..5 dans l'empreinte courante : sinon les calques ombre/blanchiment
    mis en cache restent réutilisés alors que la composition (piles ombre/blanchiment) change.
    """
    floors: List[dict] = opts.get('floors') or []
    if k < 0 or k >= len(floors):
        return ('__missing__', k)
    po = opts.get('process_floor_indices')
    if po is not None and k not in po:
        return ('__skipped__', k)
    fd = floors[k]
    if not fd.get('enabled', True):
        return ('__disabled__', k)
    glow_ex = fd.get('glow_ex') or {}
    glow_in = fd.get('glow_in') or {}
    gb = dict(fd.get('grain_blanchiment') or fd.get('spread_grain') or {})
    ga = dict(fd.get('grain_affleurement') or {})
    gb.pop('seed', None)
    ga.pop('seed', None)
    glow_ex_on = fd.get('glow_ex_enabled')
    blanch = fd.get('blanchiment_enabled')
    return (
        'current',
        _floors_enabled_mask(opts),
        bool(fd.get('ombrage_enabled', True)),
        bool(fd.get('glow_on_tb', True)),
        float(fd.get('fusion_force_inm', DEFAULT_FUSION_FORCE_INM)),
        float(fd.get('fusion_force_inexm', DEFAULT_FUSION_FORCE_INEXM)),
        glow_ex_on,
        blanch,
        float(glow_ex.get('radius', DEFAULT_GLOW_EX_RADIUS)),
        float(glow_ex.get('threshold', DEFAULT_GLOW_EX_THRESHOLD)),
        float(glow_ex.get('intensity', DEFAULT_GLOW_EX_INTENSITY)),
        float(glow_in.get('radius', DEFAULT_GLOW_IN_RADIUS)),
        float(glow_in.get('threshold', DEFAULT_GLOW_IN_THRESHOLD)),
        float(glow_in.get('intensity', DEFAULT_GLOW_IN_INTENSITY)),
        bool(gb.get('enabled', False)),
        float(gb.get('spread_x', 2.0)),
        float(gb.get('spread_y', 2.0)),
        bool(ga.get('enabled', False)),
        float(ga.get('spread_x', 2.0)),
        float(ga.get('spread_y', 2.0)),
        float(ga.get('opacity', 1.0)),
    )


def first_changed_floor_index(
    fp_old: Optional[Tuple[Any, ...]],
    fp_new: Tuple[Any, ...],
) -> Optional[int]:
    """
    Premier indice d'étage k (1..5) dont l'empreinte change.
    fp_old None → 1 (recalcul complet des surfaces).
    None si identique (aucun changement de paramètres par étage).
    """
    if fp_old is None:
        return 1
    for i in range(min(len(fp_old), len(fp_new))):
        if fp_old[i] != fp_new[i]:
            return i + 1
    if len(fp_old) != len(fp_new):
        return 1
    return None


def floor_removal_requires_full_surface_rebuild(
    fp_old: Optional[Tuple[Any, ...]],
    fp_new: Tuple[Any, ...],
) -> bool:
    """
    True si un étage était traité (empreinte courante) et ne l'est plus (__disabled__ ou __skipped__).

    Dans ce cas la reprise incrémentale depuis l'étage k peut laisser ombre/blanchiment résiduels ;
    le pipeline doit repartir de k_start=1 avec des piles vides (recalcul complet des surfaces).
    """
    if fp_old is None:
        return False
    n = min(len(fp_old), len(fp_new))
    for i in range(n):
        o, ne = fp_old[i], fp_new[i]
        if not isinstance(ne, tuple) or len(ne) < 1:
            continue
        if ne[0] not in ('__disabled__', '__skipped__'):
            continue
        if isinstance(o, tuple) and len(o) >= 1 and o[0] == 'current':
            return True
    return False


def copy_floor_surface_state(state: Dict[str, Any]) -> Dict[str, Any]:
    """Copie profonde des images PIL d'un état intermédiaire surfaces."""
    out: Dict[str, Any] = {}
    for key in (
        'INM',
        'INEXM',
        'last_EX',
        'last_EX_gl',
        'last_EX_glgr',
        'last_EXOK',
        'last_IN_gl',
        'last_IN_n',
        'last_INOK',
    ):
        img = state.get(key)
        if img is not None:
            out[key] = img.copy()
        else:
            out[key] = None
    ss = state.get('shadow_stack')
    out['shadow_stack'] = [(im.copy(), float(f)) for im, f in ss] if ss else []
    bs = state.get('bleach_stack')
    out['bleach_stack'] = [(im.copy(), float(f)) for im, f in bs] if bs else []
    return out


def image_to_preview_rgb(img: Image.Image) -> Image.Image:
    """RGB pour affichage prévisualisation (fond blanc si RGBA)."""
    if img.mode == "RGBA":
        bg = Image.new("RGB", img.size, (255, 255, 255))
        bg.paste(img, mask=img.split()[3])
        return bg
    return img.convert("RGB")


def tb_q_layer_cache_signature(TB_q: Image.Image) -> Tuple[int, int, str]:
    """Empreinte légère de l’image à paliers (TB_q) pour invalider le cache de calques par étage."""
    w, h = TB_q.size
    sm = TB_q.convert('L').resize((64, 64), Image.Resampling.NEAREST)
    return (w, h, hashlib.md5(sm.tobytes()).hexdigest())


def _floor_layer_cache_usable(
    fd: dict,
    entry: Optional[Dict[str, Any]],
    fp_k: tuple,
    sig_tb: Tuple[int, int, str],
    glow_ex_on: bool,
    grain_on: bool,
) -> bool:
    """
    Indique si les calques ombre et blanchiment (prêts pour fusion métal) mis en cache suffisent pour éviter
    manual_inok_rgb / manual_exok_rgb (recomposition identique, fusion seule).
    """
    if entry is None or entry.get('fp') != fp_k or entry.get('sig') != sig_tb:
        return False
    fin_inm = float(fd.get('fusion_force_inm', DEFAULT_FUSION_FORCE_INM))
    fin_ex = float(fd.get('fusion_force_inexm', DEFAULT_FUSION_FORCE_INEXM))
    ombre_on = fd.get('ombrage_enabled', True)
    blanch_on = bool(glow_ex_on) or bool(grain_on)
    if ombre_on:
        if fin_inm > 0:
            if entry.get('INOKTR') is None:
                return False
        else:
            return False
    if blanch_on:
        if fin_ex > 0:
            if entry.get('EXOKTR') is None:
                return False
        else:
            return False
    return True


def _refresh_last_layers_for_floor(
    TB_q: Image.Image,
    opts: dict,
    k: int,
    floors: List[dict],
    L_tb: np.ndarray,
    idx_for_masks: np.ndarray,
    used_in_tb_q: List[int],
    n_floors: int,
    allow_conversion: bool,
    seuil_blanc: int,
    seuil_blanc_inm: int,
    seuil_noir: int,
    capture_on: bool,
    captured_masks: List[Tuple[str, Image.Image]],
    process_set: Optional[set] = None,
) -> Dict[str, Any]:
    """
    Recalcule uniquement les couches last_* pour l'étage k (après fusion depuis cache).
    Sans modifier les images intermédiaires après ombres ni après blanchiments.
    """
    fd = floors[k]
    glow_ex = fd.get('glow_ex', {})
    glow_in = fd.get('glow_in', {})
    grain_b = dict(fd.get('grain_blanchiment') or fd.get('spread_grain', {}))
    grain_a = dict(fd.get('grain_affleurement', {}))
    gbase = int(opts.get('_grain_seed_base', 0))
    if grain_b.get('enabled') and grain_b.get('seed') is None:
        grain_b['seed'] = stable_spread_grain_seed_for_floor(
            gbase,
            k,
            float(grain_b.get('spread_x', 2.0)),
            float(grain_b.get('spread_y', 2.0)),
            'blanch',
        )
    if grain_a.get('enabled') and grain_a.get('seed') is None:
        grain_a['seed'] = stable_spread_grain_seed_for_floor(
            gbase,
            k,
            float(grain_a.get('spread_x', 2.0)),
            float(grain_a.get('spread_y', 2.0)),
            'aff',
        )
    grain_on = bool(grain_b.get('enabled', False)) or bool(grain_a.get('enabled', False))
    glow_ex_on = fd.get('glow_ex_enabled')
    if glow_ex_on is None:
        glow_ex_on = fd.get('blanchiment_enabled', True)

    last_EX = None
    last_EX_gl = None
    last_EX_glgr = None
    last_EXOK = None
    last_IN_gl = None
    last_IN_n = None
    last_INOK = None

    if fd.get('ombrage_enabled', True):
        receiver_candidates = set(
            _active_shadow_receiver_candidates(k, floors, process_set)
        )
        used_active = [i for i in used_in_tb_q if i in receiver_candidates]
        r_idx = resolve_shadow_receiver_index(k, n_floors, used_active)
        gray_r = FORERUNNER_FLOOR_GRAYS[r_idx]
        INOK_rgb, steps_in = manual_inok_rgb_from_tb_q(
            TB_q,
            k_carrier=k,
            k_receiver=r_idx,
            glow_in=glow_in,
            allow_conversion=allow_conversion,
            glow_on_tb=fd.get('glow_on_tb', True),
            idx=idx_for_masks,
        )
        if capture_on:
            for step_name, sim in steps_in.items():
                captured_masks.append((f"Étage {k} ombre — {step_name}", sim))
            captured_masks.append((
                f"Étage {k} — palier récepteur ombre L == {gray_r} (indice palier {r_idx})",
                visualiser_masque_gris_etage(L_tb, gray_r),
            ))
        EX = Image.eval(TB_q.convert('RGB'), lambda x: 255 - x).convert('RGBA')
        if TB_q.mode == 'RGBA':
            EX.putalpha(TB_q.split()[3])
        last_EX, last_EX_gl, last_EX_glgr = EX, EX.copy(), EX.copy()
        last_EXOK = appliquer_masque_blanc_vers_alpha(
            last_EX_glgr, TB_q.convert('L'), seuil=seuil_blanc
        )
        last_IN_gl = TB_q.copy()
        last_IN_n = appliquer_masque_blanc_vers_alpha(
            last_IN_gl, EX.convert('L'), seuil=seuil_blanc
        )
        last_INOK = INOK_rgb

    if glow_ex_on or grain_on:
        EXOK_rgb, steps_ex = manual_exok_rgb_from_tb_q(
            TB_q,
            k=k,
            glow_ex=glow_ex,
            grain_blanchiment_opts=grain_b,
            grain_affleurement_opts=grain_a,
            allow_conversion=allow_conversion,
            glow_ex_active=glow_ex_on,
            idx=idx_for_masks,
        )
        gray_k = FORERUNNER_FLOOR_GRAYS[k]
        if capture_on:
            for step_name, sim in steps_ex.items():
                captured_masks.append((f"Étage {k} blanchiment — {step_name}", sim))
            captured_masks.append((
                f"Étage {k} — palier blanchiment L == {gray_k}",
                visualiser_masque_gris_etage(L_tb, gray_k),
            ))
        spread_for_legacy_chain = {**grain_b, "enabled": False}
        EX, EX_gl, EX_glgr, EXOK, IN_gl, IN_n, INOK = _compute_exok_inok_pair(
            TB_q,
            glow_ex_active=glow_ex_on,
            glow_ex=glow_ex,
            spread_grain_opts=spread_for_legacy_chain,
            glow_on_tb=False,
            glow_in_active=False,
            glow_in=glow_in,
            allow_conversion=allow_conversion,
            seuil_blanc=seuil_blanc,
        )
        last_EX, last_EX_gl, last_EX_glgr = EX, EX_gl, EX_glgr
        last_IN_gl, last_IN_n, last_INOK = IN_gl, IN_n, INOK
        last_EXOK = EXOK_rgb

    return {
        'last_EX': last_EX,
        'last_EX_gl': last_EX_gl,
        'last_EX_glgr': last_EX_glgr,
        'last_EXOK': last_EXOK,
        'last_IN_gl': last_IN_gl,
        'last_IN_n': last_IN_n,
        'last_INOK': last_INOK,
    }


def _compute_exok_inok_pair(
    TB_q: Image.Image,
    glow_ex_active: bool,
    glow_ex: Dict[str, Any],
    spread_grain_opts: Dict[str, Any],
    glow_on_tb: bool,
    glow_in_active: bool,
    glow_in: Dict[str, Any],
    allow_conversion: bool,
    seuil_blanc: int,
) -> Tuple[Image.Image, Image.Image, Image.Image, Image.Image, Image.Image, Image.Image]:
    """Calcule couche inversée, glow/spread blanchiment, masque blanchiment, glow ombre, masques intermédiaires et calque ombre RVB."""
    EX = Image.eval(TB_q.convert('RGB'), lambda x: 255 - x)
    EX = EX.convert('RGBA')
    if TB_q.mode == 'RGBA':
        EX.putalpha(TB_q.split()[3])

    EX_gl = EX.copy()
    if glow_ex_active:
        EX_gl = apply_glow(
            EX_gl,
            radius=int(glow_ex.get('radius', DEFAULT_GLOW_EX_RADIUS)),
            threshold=int(glow_ex.get('threshold', DEFAULT_GLOW_EX_THRESHOLD)),
            intensity=float(glow_ex.get('intensity', DEFAULT_GLOW_EX_INTENSITY)),
            allow_conversion=allow_conversion,
        )
        EX_gl = EX_gl.copy()

    EX_glgr = EX_gl.copy()
    if spread_grain_opts.get('enabled', False):
        EX_glgr = apply_spread_grain_fast(
            EX_glgr.copy(),
            spread_x=float(spread_grain_opts.get('spread_x', 2.0)),
            spread_y=float(spread_grain_opts.get('spread_y', 2.0)),
            seed=spread_grain_opts.get('seed'),
            allow_conversion=allow_conversion,
            edge_mask=False,
        )

    EXOK = appliquer_masque_blanc_vers_alpha(EX_glgr, TB_q.convert('L'), seuil=seuil_blanc)

    IN_gl = TB_q.copy()
    if glow_on_tb and glow_in_active:
        IN_gl = apply_glow(
            IN_gl,
            radius=int(glow_in.get('radius', DEFAULT_GLOW_IN_RADIUS)),
            threshold=int(glow_in.get('threshold', DEFAULT_GLOW_IN_THRESHOLD)),
            intensity=float(glow_in.get('intensity', DEFAULT_GLOW_IN_INTENSITY)),
            allow_conversion=allow_conversion,
        )

    IN_n = appliquer_masque_blanc_vers_alpha(IN_gl, EX.convert('L'), seuil=seuil_blanc)
    INOK = Image.eval(IN_n.convert('RGB'), lambda x: 255 - x)

    return EX, EX_gl, EX_glgr, EXOK, IN_gl, IN_n, INOK


def _single_layer_exok_inok_from_tb_q(
    TB_q: Image.Image,
    opts: dict,
    progress_cb: Optional[Callable],
    seuil_blanc: int,
    allow_conversion: bool,
) -> Tuple[Image.Image, Image.Image, Image.Image, Image.Image, Image.Image, Image.Image, Image.Image, Image.Image]:
    """Calcule un seul jeu calques blanchiment + ombre à partir de l’image à paliers (mode sans liste ``floors``)."""
    glow_opts = opts.get('glow', {})
    glow_master = glow_opts.get('enabled', False)
    glow_ex_opts = opts.get('glow_ex', {})
    glow_in_opts = opts.get('glow_in', {})
    spread_grain_opts = opts.get('grain_blanchiment') or opts.get('spread_grain', {})

    EX, EX_gl, EX_glgr, EXOK, IN_gl, IN_n, INOK = _compute_exok_inok_pair(
        TB_q,
        glow_ex_active=glow_master,
        glow_ex={
            'radius': glow_ex_opts.get('radius', glow_opts.get('radius', DEFAULT_GLOW_EX_RADIUS)),
            'threshold': glow_ex_opts.get('threshold', glow_opts.get('threshold', DEFAULT_GLOW_EX_THRESHOLD)),
            'intensity': glow_ex_opts.get('intensity', glow_opts.get('intensity', DEFAULT_GLOW_EX_INTENSITY)),
        },
        spread_grain_opts=spread_grain_opts,
        glow_on_tb=opts.get('glow_on_tb', False),
        glow_in_active=glow_master,
        glow_in={
            'radius': glow_in_opts.get('radius', glow_opts.get('radius', DEFAULT_GLOW_IN_RADIUS)),
            'threshold': glow_in_opts.get('threshold', glow_opts.get('threshold', DEFAULT_GLOW_IN_THRESHOLD)),
            'intensity': glow_in_opts.get('intensity', glow_opts.get('intensity', DEFAULT_GLOW_IN_INTENSITY)),
        },
        allow_conversion=allow_conversion,
        seuil_blanc=seuil_blanc,
    )
    _appeler_progress(progress_cb, 60, UI_PROGRESS_CALQUE_BLANCHIMENT_MASQUE)
    _appeler_progress(progress_cb, 80, UI_PROGRESS_CALQUE_OMBRE_MASQUE)
    return EX, EX_gl, EX_glgr, EXOK, IN_gl, IN_n, INOK


def _merge_on_base_metal(
    base_rgba: Image.Image,
    opts: dict,
    progress_cb: Optional[Callable],
    seuil_blanc_inm: int,
    seuil_noir: int,
    fusion_mode: str,
    EXOK: Image.Image,
    INOK: Image.Image,
    fusion_force_inm: float,
    fusion_force_inexm: float,
) -> Tuple[Optional[Image.Image], Optional[Image.Image], Image.Image, Image.Image]:
    INOKTR = appliquer_couleur_vers_alpha_direct(INOK, couleur_cible='blanc', seuil=seuil_blanc_inm)
    _appeler_progress(progress_cb, 87, UI_PROGRESS_CALQUE_OMBRE_FUSION_METAL)
    INM = fusionner_grains(base_rgba.convert('RGB'), INOKTR, mode=fusion_mode, force=fusion_force_inm)
    _appeler_progress(progress_cb, 90, UI_PROGRESS_FUSION_OMBRES)
    EXOKTR = appliquer_couleur_vers_alpha_direct(EXOK, couleur_cible='noir', seuil=seuil_noir)
    _appeler_progress(progress_cb, 92, UI_PROGRESS_CALQUE_BLANCHIMENT_FUSION_METAL)
    INEXM = fusionner_grains(INM, EXOKTR, mode=fusion_mode, force=fusion_force_inexm)
    _appeler_progress(progress_cb, 95, UI_PROGRESS_FUSION_BLANCHIMENTS)
    return INOKTR, EXOKTR, INM, INEXM


def _prepare_tb_and_tb_q_presnap(
    pil_image: Image.Image,
    opts: dict,
    progress_cb: Optional[Callable],
) -> Tuple[Image.Image, Image.Image, Optional[List[int]]]:
    """
    Image de base corrigée (anti-flou / AA), copie quantifiée (contours + quantif), indices de snap sous-ensemble.
    N'applique pas encore le snap luminance Forerunner sur cette copie.
    """
    TB = pil_image.copy()
    if TB.mode != 'RGBA':
        TB = TB.convert('RGBA')

    seuil_flou = opts.get('seuil_flou', DEFAULT_SEUIL_FLOU)
    deblur_large_applique = False
    # Netteté déjà renforcée sur TB (unsharp) → ne pas réappliquer renforcer_contours sur TB_q (doublage d'arêtes).
    tb_netete_deja_renforcee = False
    if seuil_flou > 0:
        flou_avant = detecter_flou(TB)
        if flou_avant < seuil_flou:
            ui_log(
                "Image peu nette ou transitions trop douces (score de netteté faible : "
                f"{flou_avant:.1f} < {seuil_flou:.0f}) — nettoyage auto des étages parasites "
                "à la détection des paliers.",
                'warning',
            )
            if opts.get('auto_sharpen_blur', DEFAULT_AUTO_SHARPEN_BLUR):
                TB = renforcer_netete_dessin_flou(
                    TB,
                    radius=float(opts.get('deblur_unsharp_radius', DEFAULT_DEBLUR_UNSHARP_RADIUS)),
                    percent=int(opts.get('deblur_unsharp_percent', DEFAULT_DEBLUR_UNSHARP_PERCENT)),
                    threshold=int(opts.get('deblur_unsharp_threshold', DEFAULT_DEBLUR_UNSHARP_THRESHOLD)),
                    passes=int(opts.get('deblur_unsharp_passes', DEFAULT_DEBLUR_UNSHARP_PASSES)),
                )
                deblur_large_applique = True
                tb_netete_deja_renforcee = True
                flou_apres = detecter_flou(TB)
                ui_log(
                    "Anti-flou (masque flou) appliqué — "
                    f"indicateur de netteté {flou_avant:.1f} → {flou_apres:.1f}.",
                    'info',
                )

    antialias_actif = opts.get('auto_antialias_pass', DEFAULT_AUTO_ANTIALIAS_PASS)
    aa_suspect = antialias_actif and trace_melange_antialias(TB)
    # Seuil de référence pour décider si une « passe AA » seule est pertinente : les aplats à
    # paliers / géométrie à arêtes dures peuvent déclencher trace_melange_antialias à tort
    # (fraction de gradients modérés entre niveaux) ; un unsharp alors produit halos / double contour.
    seuil_ref = float(seuil_flou) if seuil_flou > 0 else float(DEFAULT_SEUIL_FLOU)
    flou_courant = detecter_flou(TB)
    if antialias_actif and (deblur_large_applique or aa_suspect):
        if not deblur_large_applique and aa_suspect and flou_courant >= seuil_ref:
            ui_log(
                "Passe arêtes fines (antialiasing) ignorée : "
                f"indicateur de netteté déjà élevé ({flou_courant:.1f} ≥ {seuil_ref:.0f}) — "
                "évite halos sur arêtes déjà nettes (ex. dégradés à paliers, motifs géométriques).",
                'info',
            )
        else:
            if aa_suspect and not deblur_large_applique:
                ui_log(
                    "Transitions douces détectées (ex. tracé avec antialiasing) : "
                    "resserrement des arêtes (masque flou fin) avant snap.",
                    'info',
                )
            TB = renforcer_netete_antialias_trace(
                TB,
                radius=float(opts.get('aa_unsharp_radius', DEFAULT_AA_UNSHARP_RADIUS)),
                percent=int(opts.get('aa_unsharp_percent', DEFAULT_AA_UNSHARP_PERCENT)),
                threshold=int(opts.get('aa_unsharp_threshold', DEFAULT_AA_UNSHARP_THRESHOLD)),
                passes=int(opts.get('aa_unsharp_passes', DEFAULT_AA_UNSHARP_PASSES)),
            )
            tb_netete_deja_renforcee = True
            if deblur_large_applique:
                ui_log(
                    "Passe arêtes fines (antialiasing / franges) appliquée ; puis contours et snap Forerunner.",
                    'info',
                )

    infer_snap_idx: Optional[List[int]] = None
    if opts.get('snap_forerunner_floors', True) and not opts.get(
        'snap_use_full_palette', DEFAULT_SNAP_USE_FULL_PALETTE
    ):
        infer_snap_idx = infer_snap_indices_for_image(TB)

    _appeler_progress(progress_cb, 10, "Image chargée")

    TB_q = TB.copy()
    contour_methode = opts.get('contour_methode', DEFAULT_CONTOUR_METHODE)
    # Par défaut désactivé : renforcer_contours sur TB_q seulement (pas sur TB) crée souvent
    # un « double contour » / halos avant snap ; le snap masque l’effet en étape 3.
    do_contour = bool(opts.get('do_contour', False))
    if (
        do_contour
        and opts.get('snap_only')
        and opts.get('snap_only_skip_contour', DEFAULT_SNAP_ONLY_SKIP_CONTOUR)
    ):
        do_contour = False
        _appeler_progress(
            progress_cb,
            20,
            "Contours non renforcés (mise au propre : garde arêtes nettes, évite frange 1 px)",
        )
    if do_contour:
        force_tb_q_contour = bool(opts.get('force_tb_q_contour_after_sharpen', False))
        if tb_netete_deja_renforcee and not force_tb_q_contour:
            ui_log(
                "Image quantifiée : renforcement « contours » ignoré — netteté déjà appliquée sur l’image de base "
                "(évite un double traitement d'arêtes). Option force_tb_q_contour_after_sharpen pour forcer.",
                "info",
            )
            _appeler_progress(
                progress_cb,
                20,
                "Contours (image quantifiée) : inchangés (évite doublage avec netteté sur l’image de base)",
            )
        else:
            TB_q = renforcer_contours(TB_q, methode=contour_methode)
            _appeler_progress(progress_cb, 20, "Contours renforcés")

    niveaux = opts.get('niveaux', 0)
    if niveaux > 1:
        palette = opts.get('palette')
        TB_q = quantifier(TB_q, niveaux, palette)
        _appeler_progress(progress_cb, 30, f"Quantification en {niveaux} niveaux")
    else:
        _appeler_progress(progress_cb, 30, "Pas de quantification")

    opts['_tb_netete_deja_renforcee'] = tb_netete_deja_renforcee
    return TB, TB_q, infer_snap_idx


def _forerunner_snap_should_run(opts: dict) -> bool:
    """
    Indique si le snap Forerunner doit s’exécuter : oui après corrections qui créent des gris
    intermédiaires, non si TB est déjà net et sans option de forçage.

    Le mode « mise au propre » (snap_only), dont la sortie éditeur de construction, impose
    toujours le snap lorsque snap_forerunner_floors est actif.
    """
    if not opts.get('snap_forerunner_floors', True):
        return False
    if opts.get('snap_only'):
        return True
    if opts.get('force_forerunner_snap', False):
        return True
    if '_tb_netete_deja_renforcee' not in opts:
        return True
    return bool(opts['_tb_netete_deja_renforcee'])


def _apply_forerunner_snap_to_tb_q(
    TB_q: Image.Image,
    TB: Image.Image,
    opts: dict,
    progress_cb: Optional[Callable],
) -> Image.Image:
    """Snap des luminances sur les paliers Forerunner (entrée : image quantifiée avant ce snap)."""
    if not opts.get('snap_forerunner_floors', True):
        return TB_q
    if not _forerunner_snap_should_run(opts):
        ui_log(
            "Snap Forerunner ignoré : texture déjà nette (aucune correction anti-flou / arêtes sur l’image de base). "
            "Option force_forerunner_snap pour quantifier quand même sur les paliers.",
            'info',
        )
        _appeler_progress(
            progress_cb,
            32,
            "Snap Forerunner ignoré (texture déjà nette)",
        )
        return TB_q
    use_full = opts.get('snap_use_full_palette', DEFAULT_SNAP_USE_FULL_PALETTE)
    if use_full:
        TB_q = snap_luminance_to_forerunner_floors(TB_q)
        _appeler_progress(progress_cb, 32, "Gris snapés sur les 6 paliers Forerunner")
    else:
        idx = opts.get('_infer_snap_indices')
        if not idx:
            idx = infer_snap_indices_for_image(TB)
        TB_q = snap_luminance_to_forerunner_floors_subset(TB_q, idx)
        _appeler_progress(
            progress_cb,
            32,
            f"Snap sur {len(idx)} palier(s) détecté(s) dans l'image",
        )
    return TB_q


def _process_per_floor_surfaces(
    TB_q: Image.Image,
    TB: Image.Image,
    opts: dict,
    progress_cb: Optional[Callable],
    *,
    k_start: int = 1,
    resume_state: Optional[Dict[str, Any]] = None,
    save_checkpoints: bool = False,
    visual_cb: Optional[Callable[[Image.Image, str], None]] = None,
) -> Dict[str, Any]:
    """Fusion par étage Forerunner (méthode manuelle : N/B, glow, récepteur d'ombre auto).

    Point d’entrée logique : **image de base alignée sur les paliers** (``_apply_forerunner_snap_to_tb_q``) et **opts**
    (floors, TM, fusion, périmètre, graines, debug, etc.). Toute la boucle par étage lit ``idx_for_masks``
    dérivé de cette image + ``apply_idx_remap_for_pipeline(..., opts)``.

    Les calques ombre et blanchiment (avec transparence pour la métal) ne sont plus fusionnés en chaîne sur le résultat précédent :
    ils sont recomposés depuis la base métal (ombres) puis depuis l’image déjà fusionnée avec ces ombres (blanchiments)
    en sommant les écarts RVB ``fusionner_grains(base, calque) - base`` — évite l'accumulation
    d'artefacts aux joints d'étages.

    k_start : premier indice d'étage à traiter (1..5). Si > 1, resume_state doit contenir
    l'état après l'étage k_start - 1 (copies d'images et piles shadow_stack / bleach_stack).
    save_checkpoints : si True, remplit ``_checkpoints`` avec une copie d'état après chaque k traité.
    """
    floors: List[dict] = opts['floors']
    if len(floors) != len(FORERUNNER_FLOOR_GRAYS):
        ui_log(f"opts['floors'] doit avoir {len(FORERUNNER_FLOOR_GRAYS)} entrées, reçu {len(floors)}", 'warning')

    allow_conversion = opts.get('allow_conversion', False)
    seuil_blanc = opts.get('masque_blanc_seuil', 250)
    seuil_blanc_inm = seuil_blanc
    seuil_noir = opts.get('masque_noir_seuil', 5)
    fusion_mode = opts.get('fusion_mode', 'grain_merge')

    L_tb = np.asarray(TB_q.convert('L'), dtype=np.int16)
    idx_for_masks = apply_idx_remap_for_pipeline(luminance_to_floor_index_map(L_tb), opts)
    used_in_tb_q = forerunner_floor_indices_present_in_image(TB_q)
    w, h = TB_q.size
    capture_on = bool(opts.get('captured_surface_masks', False))
    captured_masks: List[Tuple[str, Image.Image]] = []

    checkpoints: Optional[Dict[int, Dict[str, Any]]] = {} if save_checkpoints else None

    shadow_stack: List[Tuple[Image.Image, float]] = []
    bleach_stack: List[Tuple[Image.Image, float]] = []

    TM_path = opts.get('tm')
    tm_ok = bool(TM_path and Path(TM_path).exists())
    if tm_ok:
        TM = charger_image(TM_path).convert('RGBA')
        tw, th = TB_q.size
        if TM.size != (tw, th):
            TM = prepare_metal_texture_rgba(TM, tw, th, opts.get('tm_fit_mode', TM_FIT_ANISOTROPIC))
        base = TM
    else:
        base = Image.new('RGBA', (w, h), (255, 255, 255, 255))

    if k_start < 1:
        k_start = 1

    if resume_state is not None and k_start > 1:
        if 'shadow_stack' not in resume_state or 'bleach_stack' not in resume_state:
            ui_log(
                "Reprise incrémentale sans piles de calques : recalcul complet des étages.",
                "info",
            )
            resume_state = None
            k_start = 1

    def _snapshot() -> Dict[str, Any]:
        return copy_floor_surface_state(
            {
                'INM': INM,
                'INEXM': INEXM,
                'last_EX': last_EX,
                'last_EX_gl': last_EX_gl,
                'last_EX_glgr': last_EX_glgr,
                'last_EXOK': last_EXOK,
                'last_IN_gl': last_IN_gl,
                'last_IN_n': last_IN_n,
                'last_INOK': last_INOK,
                'shadow_stack': shadow_stack,
                'bleach_stack': bleach_stack,
            }
        )

    if resume_state is None:
        _appeler_progress(
            progress_cb,
            85,
            "Texture métallique chargée" if tm_ok else "Base blanche (pas de TM)",
        )
        INM = base
        INEXM = base
        last_EX = None
        last_EX_gl = None
        last_EX_glgr = None
        last_EXOK = None
        last_IN_gl = None
        last_IN_n = None
        last_INOK = None
        if save_checkpoints and checkpoints is not None:
            checkpoints[0] = _snapshot()
        if visual_cb is not None:
            visual_cb(image_to_preview_rgb(base), "Base — texture métallique")
    else:
        rs = copy_floor_surface_state(resume_state)
        shadow_stack = rs['shadow_stack']
        bleach_stack = rs['bleach_stack']
        INM = composite_grain_contributions_from_base(base, shadow_stack, fusion_mode)
        INEXM = composite_grain_contributions_from_base(INM, bleach_stack, fusion_mode)
        last_EX = rs['last_EX']
        last_EX_gl = rs['last_EX_gl']
        last_EX_glgr = rs['last_EX_glgr']
        last_EXOK = rs['last_EXOK']
        last_IN_gl = rs['last_IN_gl']
        last_IN_n = rs['last_IN_n']
        last_INOK = rs['last_INOK']

    n_floors = len(FORERUNNER_FLOOR_GRAYS)
    process_only = opts.get('process_floor_indices')
    process_set = set(process_only) if process_only is not None else None

    layer_cache: Optional[Dict[int, Any]] = opts.get('_floor_layer_cache')
    sig_tb: Optional[Tuple[int, int, str]] = (
        tb_q_layer_cache_signature(TB_q) if layer_cache is not None else None
    )
    last_processed_k: Optional[int] = None
    last_processed_was_cache = False

    for k in range(k_start, n_floors):
        if k >= len(floors):
            break
        if process_set is not None and k not in process_set:
            if layer_cache is not None:
                layer_cache.pop(k, None)
            if save_checkpoints and checkpoints is not None:
                checkpoints[k] = _snapshot()
            continue
        fd = floors[k]
        if not fd.get('enabled', True):
            if layer_cache is not None:
                layer_cache.pop(k, None)
            if save_checkpoints and checkpoints is not None:
                checkpoints[k] = _snapshot()
            continue

        pct_base = 40 + int(50 * (k - 1) / max(n_floors - 1, 1))
        glow_ex = fd.get('glow_ex', {})
        glow_in = fd.get('glow_in', {})

        grain_b = dict(fd.get('grain_blanchiment') or fd.get('spread_grain', {}))
        grain_a = dict(fd.get('grain_affleurement', {}))
        gbase = int(opts.get('_grain_seed_base', 0))
        if grain_b.get('enabled') and grain_b.get('seed') is None:
            grain_b['seed'] = stable_spread_grain_seed_for_floor(
                gbase,
                k,
                float(grain_b.get('spread_x', 2.0)),
                float(grain_b.get('spread_y', 2.0)),
                'blanch',
            )
        if grain_a.get('enabled') and grain_a.get('seed') is None:
            grain_a['seed'] = stable_spread_grain_seed_for_floor(
                gbase,
                k,
                float(grain_a.get('spread_x', 2.0)),
                float(grain_a.get('spread_y', 2.0)),
                'aff',
            )
        grain_on = bool(grain_b.get('enabled', False)) or bool(grain_a.get('enabled', False))
        glow_ex_on = fd.get('glow_ex_enabled')
        if glow_ex_on is None:
            glow_ex_on = fd.get('blanchiment_enabled', True)

        fin_inm = float(fd.get('fusion_force_inm', DEFAULT_FUSION_FORCE_INM))
        fin_ex = float(fd.get('fusion_force_inexm', DEFAULT_FUSION_FORCE_INEXM))

        fp_k = floor_surface_fingerprint(opts, k)
        entry = layer_cache.get(k) if layer_cache is not None else None
        use_layer = (
            layer_cache is not None
            and sig_tb is not None
            and _floor_layer_cache_usable(
                fd, entry, fp_k, sig_tb, glow_ex_on, grain_on
            )
        )

        if use_layer:
            assert entry is not None
            # Reprise incrémentale (k_start > 1) : pas de message « cache calques » pour chaque
            # étage au-dessus du premier recalculé — le statut reste sur l’étage en cours de traitement.
            _cache_msg = not (k_start > 1 and k > k_start)
            if fd.get('ombrage_enabled', True) and fin_inm > 0:
                if _cache_msg:
                    _appeler_progress(progress_cb, pct_base, f"Étage {k} : ombres (cache calques)")
                shadow_stack.append((entry['INOKTR'].copy(), fin_inm))
            if glow_ex_on or grain_on:
                if fin_ex > 0:
                    if _cache_msg:
                        _appeler_progress(
                            progress_cb,
                            min(94, pct_base + 8),
                            f"Étage {k} : blanchiment (cache calques)",
                        )
                    bleach_stack.append((entry['EXOKTR'].copy(), fin_ex))
            INM = composite_grain_contributions_from_base(base, shadow_stack, fusion_mode)
            INEXM = composite_grain_contributions_from_base(INM, bleach_stack, fusion_mode)
            last_processed_k = k
            last_processed_was_cache = True
        else:
            last_processed_was_cache = False
            INOKTR: Optional[Image.Image] = None
            EXOKTR: Optional[Image.Image] = None
            if fd.get('ombrage_enabled', True):
                _appeler_progress(progress_cb, pct_base, f"Étage {k} : ombres")

                receiver_candidates = set(
                    _active_shadow_receiver_candidates(k, floors, process_set)
                )
                used_active = [i for i in used_in_tb_q if i in receiver_candidates]
                r_idx = resolve_shadow_receiver_index(k, n_floors, used_active)
                gray_r = FORERUNNER_FLOOR_GRAYS[r_idx]
                INOK_rgb, steps_in = manual_inok_rgb_from_tb_q(
                    TB_q,
                    k_carrier=k,
                    k_receiver=r_idx,
                    glow_in=glow_in,
                    allow_conversion=allow_conversion,
                    glow_on_tb=fd.get('glow_on_tb', True),
                    idx=idx_for_masks,
                )
                if capture_on:
                    for step_name, sim in steps_in.items():
                        captured_masks.append((
                            f"Étage {k} ombre — {step_name}",
                            sim,
                        ))
                    captured_masks.append((
                        f"Étage {k} — palier récepteur ombre L == {gray_r} (indice palier {r_idx})",
                        visualiser_masque_gris_etage(L_tb, gray_r),
                    ))
                if fin_inm > 0:
                    INOKTR = appliquer_couleur_vers_alpha_direct(
                        INOK_rgb, couleur_cible='blanc', seuil=seuil_blanc_inm
                    )
                    shadow_stack.append((INOKTR, fin_inm))
                EX = Image.eval(TB_q.convert('RGB'), lambda x: 255 - x).convert('RGBA')
                if TB_q.mode == 'RGBA':
                    EX.putalpha(TB_q.split()[3])
                last_EX, last_EX_gl, last_EX_glgr = EX, EX.copy(), EX.copy()
                last_EXOK = appliquer_masque_blanc_vers_alpha(
                    last_EX_glgr, TB_q.convert('L'), seuil=seuil_blanc
                )
                last_IN_gl = TB_q.copy()
                last_IN_n = appliquer_masque_blanc_vers_alpha(
                    last_IN_gl, EX.convert('L'), seuil=seuil_blanc
                )
                last_INOK = INOK_rgb

            if glow_ex_on or grain_on:
                _appeler_progress(
                    progress_cb,
                    min(94, pct_base + 8),
                    f"Étage {k} : blanchiment (grain blanchiment / affleurement si activés)",
                )

                EXOK_rgb, steps_ex = manual_exok_rgb_from_tb_q(
                    TB_q,
                    k=k,
                    glow_ex=glow_ex,
                    grain_blanchiment_opts=grain_b,
                    grain_affleurement_opts=grain_a,
                    allow_conversion=allow_conversion,
                    glow_ex_active=glow_ex_on,
                    idx=idx_for_masks,
                )
                gray_k = FORERUNNER_FLOOR_GRAYS[k]
                if capture_on:
                    for step_name, sim in steps_ex.items():
                        captured_masks.append((
                            f"Étage {k} blanchiment — {step_name}",
                            sim,
                        ))
                    captured_masks.append((
                        f"Étage {k} — palier blanchiment L == {gray_k}",
                        visualiser_masque_gris_etage(L_tb, gray_k),
                    ))
                if fin_ex > 0:
                    EXOKTR = appliquer_couleur_vers_alpha_direct(
                        EXOK_rgb, couleur_cible='noir', seuil=seuil_noir
                    )
                    bleach_stack.append((EXOKTR, fin_ex))
                # Chaîne blanchiment « classique » pour last_* : sans re-appliquer le grain blanchiment ici,
                # il est déjà intégré dans le calque RVB manual_exok (glow, grain plein cadre, paliers, adouci, affleurement).
                spread_for_legacy_chain = {**grain_b, "enabled": False}
                EX, EX_gl, EX_glgr, EXOK, IN_gl, IN_n, INOK = _compute_exok_inok_pair(
                    TB_q,
                    glow_ex_active=glow_ex_on,
                    glow_ex=glow_ex,
                    spread_grain_opts=spread_for_legacy_chain,
                    glow_on_tb=False,
                    glow_in_active=False,
                    glow_in=glow_in,
                    allow_conversion=allow_conversion,
                    seuil_blanc=seuil_blanc,
                )
                last_EX, last_EX_gl, last_EX_glgr = EX, EX_gl, EX_glgr
                last_IN_gl, last_IN_n, last_INOK = IN_gl, IN_n, INOK
                last_EXOK = EXOK_rgb

            INM = composite_grain_contributions_from_base(base, shadow_stack, fusion_mode)
            INEXM = composite_grain_contributions_from_base(INM, bleach_stack, fusion_mode)

            if layer_cache is not None and sig_tb is not None:
                cin: Optional[Image.Image] = None
                cex: Optional[Image.Image] = None
                if fd.get('ombrage_enabled', True) and fin_inm > 0 and INOKTR is not None:
                    cin = INOKTR.copy()
                if (glow_ex_on or grain_on) and fin_ex > 0 and EXOKTR is not None:
                    cex = EXOKTR.copy()
                layer_cache[k] = {
                    'fp': fp_k,
                    'sig': sig_tb,
                    'INOKTR': cin,
                    'EXOKTR': cex,
                }

            last_processed_k = k

        if save_checkpoints and checkpoints is not None:
            checkpoints[k] = _snapshot()

        if visual_cb is not None:
            visual_cb(image_to_preview_rgb(INEXM), f"Surfaces — étage {k}")

    if (
        last_processed_was_cache
        and last_processed_k is not None
        and layer_cache is not None
    ):
        bundle = _refresh_last_layers_for_floor(
            TB_q,
            opts,
            last_processed_k,
            floors,
            L_tb,
            idx_for_masks,
            used_in_tb_q,
            n_floors,
            allow_conversion,
            seuil_blanc,
            seuil_blanc_inm,
            seuil_noir,
            capture_on,
            captured_masks,
            process_set,
        )
        last_EX = bundle['last_EX']
        last_EX_gl = bundle['last_EX_gl']
        last_EX_glgr = bundle['last_EX_glgr']
        last_EXOK = bundle['last_EXOK']
        last_IN_gl = bundle['last_IN_gl']
        last_IN_n = bundle['last_IN_n']
        last_INOK = bundle['last_INOK']

    final = INEXM.convert('RGB')
    _appeler_progress(progress_cb, 100, "Traitement terminé")

    if last_EX is None:
        last_EX = Image.eval(TB_q.convert('RGB'), lambda x: 255 - x).convert('RGBA')
        if TB_q.mode == 'RGBA':
            last_EX.putalpha(TB_q.split()[3])
        last_EX_gl = last_EX.copy()
        last_EX_glgr = last_EX_gl.copy()
        last_IN_gl = TB_q.copy()
        last_IN_n = appliquer_masque_blanc_vers_alpha(last_IN_gl, last_EX.convert('L'), seuil=seuil_blanc)
        last_INOK = Image.eval(last_IN_n.convert('RGB'), lambda x: 255 - x)
        last_EXOK = appliquer_masque_blanc_vers_alpha(last_EX_glgr, TB_q.convert('L'), seuil=seuil_blanc)

    out: Dict[str, Any] = {
        'EX': last_EX,
        'EX_gl': last_EX_gl,
        'EX_glgr': last_EX_glgr,
        'EXOK': last_EXOK,
        'IN_gl': last_IN_gl,
        'IN_n': last_IN_n,
        'INOK': last_INOK,
        'INM': INM,
        'INEXM': INEXM,
        'final': final,
        'captured_surface_masks': captured_masks,
    }
    if save_checkpoints and checkpoints is not None:
        out['_checkpoints'] = checkpoints
    return out


def process_image_stepwise(
    pil_image: Image.Image,
    opts: dict,
    progress_cb: Optional[Callable] = None
) -> Dict[str, Image.Image]:
    """
    Exécute le pipeline de traitement et retourne un dictionnaire d'images intermédiaires.

    Si opts['snap_only'] est True : snap sur les paliers détectés dans la source (ou 6 si
    snap_use_full_palette), sans contours si snap_only_skip_contour (défaut : évite une
    frange ~1 px sur arêtes déjà nettes), puis arrêt (pas d'ombre ni texture métal).

    Si opts contient une clé 'floors' (liste de 6 dicts), mode multi-étages Forerunner.
    Sinon, un seul jeu de calques blanchiment et ombre est calculé à partir de l’image à paliers.
    """
    _appeler_progress(progress_cb, 0, "Démarrage du traitement")

    original_format = pil_image.format

    TB, TB_q, infer_snap_idx = _prepare_tb_and_tb_q_presnap(pil_image, opts, progress_cb)

    if opts.get("snap_only"):
        TB_clean = TB_q
        if opts.get("snap_forerunner_floors", True) and _forerunner_snap_should_run(opts):
            if opts.get("snap_use_full_palette", DEFAULT_SNAP_USE_FULL_PALETTE):
                TB_clean = snap_luminance_to_forerunner_floors(TB_clean)
                _appeler_progress(progress_cb, 70, "Snap 6 paliers (aucune surface)")
            else:
                idx = infer_snap_idx or infer_snap_indices_for_image(TB)
                TB_clean = snap_luminance_to_forerunner_floors_subset(TB_clean, idx)
                _appeler_progress(
                    progress_cb,
                    70,
                    f"Snap {len(idx)} palier(s) détecté(s) (aucune surface)",
                )
        elif opts.get("snap_forerunner_floors", True):
            ui_log(
                "Snap Forerunner ignoré : texture déjà nette (aucune correction anti-flou / arêtes sur l’image de base). "
                "Option force_forerunner_snap pour quantifier quand même sur les paliers.",
                'info',
            )
            _appeler_progress(progress_cb, 70, "Snap Forerunner ignoré (texture déjà nette)")
        _appeler_progress(progress_cb, 100, "Terminé")
        final = TB_clean.convert("RGB")
        return {
            "TB": TB,
            "TB_q": TB_clean,
            "EX": TB_clean,
            "EX_gl": TB_clean,
            "EX_glgr": TB_clean,
            "EXOK": TB_clean,
            "IN_gl": TB_clean,
            "IN_n": TB_clean,
            "INOK": final,
            "final": final,
        }

    floors = opts.get('floors')
    allow_conversion = opts.get('allow_conversion', False)
    seuil_blanc = opts.get('masque_blanc_seuil', 250)

    if floors and isinstance(floors, list) and len(floors) > 0:
        opts_surfaces = dict(opts)
        if infer_snap_idx is not None:
            opts_surfaces['_infer_snap_indices'] = infer_snap_idx
        TB_q_snapped = _apply_forerunner_snap_to_tb_q(TB_q, TB, opts_surfaces, progress_cb)
        pf = _process_per_floor_surfaces(TB_q_snapped, TB, opts_surfaces, progress_cb)
        result: Dict[str, Image.Image] = {
            'TB': TB,
            'TB_q': TB_q,
            'EX': pf['EX'],
            'EX_gl': pf['EX_gl'],
            'EX_glgr': pf['EX_glgr'],
            'EXOK': pf['EXOK'],
            'IN_gl': pf['IN_gl'],
            'IN_n': pf['IN_n'],
            'INOK': pf['INOK'],
            'final': pf['final'],
        }
        result['INM'] = pf['INM']
        result['INEXM'] = pf['INEXM']
        dm = pf.get('captured_surface_masks') or []
        if dm:
            result['captured_surface_masks'] = dm
        return result

    EX, EX_gl, EX_glgr, EXOK, IN_gl, IN_n, INOK = _single_layer_exok_inok_from_tb_q(
        TB_q, opts, progress_cb, seuil_blanc, allow_conversion
    )

    TM_path = opts.get('tm')
    INOKTR = None
    EXOKTR = None
    INM = None
    INEXM = None
    final = None

    if TM_path and Path(TM_path).exists():
        TM = charger_image(TM_path).convert('RGBA')
        tw, th = TB.size
        if TM.size != (tw, th):
            TM = prepare_metal_texture_rgba(TM, tw, th, opts.get('tm_fit_mode', TM_FIT_ANISOTROPIC))

        fusion_mode = opts.get('fusion_mode', 'grain_merge')
        fusion_force_inm = opts.get('fusion_force_inm', DEFAULT_FUSION_FORCE_INM)
        fusion_force_inexm = opts.get('fusion_force_inexm', DEFAULT_FUSION_FORCE_INEXM)

        INOKTR, EXOKTR, INM, INEXM = _merge_on_base_metal(
            TM,
            opts,
            progress_cb,
            seuil_blanc,
            opts.get('masque_noir_seuil', 5),
            fusion_mode,
            EXOK,
            INOK,
            fusion_force_inm,
            fusion_force_inexm,
        )
        final = INEXM.convert('RGB')
    else:
        if opts.get('use_inok', False):
            final = INOK
        else:
            final = EXOK

    _appeler_progress(progress_cb, 100, "Traitement terminé")

    result = {
        'TB': TB,
        'TB_q': TB_q,
        'EX': EX,
        'EX_gl': EX_gl,
        'EX_glgr': EX_glgr,
        'EXOK': EXOK,
        'IN_gl': IN_gl,
        'IN_n': IN_n,
        'INOK': INOK,
    }

    if INOKTR is not None:
        result['INOKTR'] = INOKTR
    if EXOKTR is not None:
        result['EXOKTR'] = EXOKTR
    if INM is not None:
        result['INM'] = INM
    if INEXM is not None:
        result['INEXM'] = INEXM

    result['final'] = final
    return result


def forerunner_presnap_cache_key(opts: dict) -> tuple:
    """
    Empreinte des options qui influencent l’image de base et la copie quantifiée avant et pendant le snap Forerunner.
    À combiner avec un identifiant de génération de l’image source (ex. compteur au chargement).
    """
    pal = opts.get('palette')
    pal_t = str(pal) if pal is not None else None
    return (
        opts.get('seuil_flou', DEFAULT_SEUIL_FLOU),
        opts.get('auto_sharpen_blur', DEFAULT_AUTO_SHARPEN_BLUR),
        float(opts.get('deblur_unsharp_radius', DEFAULT_DEBLUR_UNSHARP_RADIUS)),
        int(opts.get('deblur_unsharp_percent', DEFAULT_DEBLUR_UNSHARP_PERCENT)),
        int(opts.get('deblur_unsharp_threshold', DEFAULT_DEBLUR_UNSHARP_THRESHOLD)),
        int(opts.get('deblur_unsharp_passes', DEFAULT_DEBLUR_UNSHARP_PASSES)),
        opts.get('auto_antialias_pass', DEFAULT_AUTO_ANTIALIAS_PASS),
        float(opts.get('aa_unsharp_radius', DEFAULT_AA_UNSHARP_RADIUS)),
        int(opts.get('aa_unsharp_percent', DEFAULT_AA_UNSHARP_PERCENT)),
        int(opts.get('aa_unsharp_threshold', DEFAULT_AA_UNSHARP_THRESHOLD)),
        int(opts.get('aa_unsharp_passes', DEFAULT_AA_UNSHARP_PASSES)),
        opts.get('snap_forerunner_floors', True),
        opts.get('snap_use_full_palette', DEFAULT_SNAP_USE_FULL_PALETTE),
        bool(opts.get('do_contour', False)),
        opts.get('contour_methode', DEFAULT_CONTOUR_METHODE),
        bool(opts.get('force_tb_q_contour_after_sharpen', False)),
        bool(opts.get('snap_only', False)),
        bool(opts.get('snap_only_skip_contour', DEFAULT_SNAP_ONLY_SKIP_CONTOUR)),
        int(opts.get('niveaux', 0)),
        pal_t,
        bool(opts.get('force_forerunner_snap', DEFAULT_FORCE_FORERUNNER_SNAP)),
    )


def build_forerunner_snap_cache_state(
    pil_image: Image.Image,
    opts: dict,
    progress_cb: Optional[Callable] = None,
    visual_cb: Optional[Callable[[Image.Image, str], None]] = None,
) -> Tuple[Image.Image, Image.Image, Image.Image, Optional[List[int]]]:
    """
    Calcule l’image de base préparée, la version quantifiée avant alignement paliers, puis après alignement Forerunner, indices sous-ensemble.
    Utilisé par la GUI pour mettre en cache la phase coûteuse avant les surfaces par étage.
    """
    TB, TB_q_presnap, infer_snap_idx = _prepare_tb_and_tb_q_presnap(pil_image, opts, progress_cb)
    if visual_cb is not None:
        visual_cb(image_to_preview_rgb(TB), UI_GRAPHE_TITRE_IMAGE_BASE_PREP)
        visual_cb(image_to_preview_rgb(TB_q_presnap), UI_GRAPHE_TITRE_IMAGE_BASE_AVANT_PALIERS)
    opts_s = dict(opts)
    if infer_snap_idx is not None:
        opts_s['_infer_snap_indices'] = infer_snap_idx
    TB_q_snapped = _apply_forerunner_snap_to_tb_q(TB_q_presnap, TB, opts_s, progress_cb)
    if visual_cb is not None:
        visual_cb(image_to_preview_rgb(TB_q_snapped), UI_GRAPHE_TITRE_IMAGE_BASE_PALIERS)
    return TB, TB_q_presnap, TB_q_snapped, infer_snap_idx


def process_forerunner_surfaces_only(
    TB: Image.Image,
    TB_q_presnap: Image.Image,
    TB_q_snapped: Image.Image,
    infer_snap_idx: Optional[List[int]],
    opts: dict,
    progress_cb: Optional[Callable] = None,
    *,
    k_start: int = 1,
    resume_state: Optional[Dict[str, Any]] = None,
    save_checkpoints: bool = False,
    visual_cb: Optional[Callable[[Image.Image, str], None]] = None,
) -> Dict[str, Any]:
    """
    Exécute uniquement _process_per_floor_surfaces à partir d’une image de base déjà alignée sur les paliers.
    Même résultat que process_image_stepwise en mode floors si le snap cache est cohérent.
    """
    opts_surfaces = dict(opts)
    if infer_snap_idx is not None:
        opts_surfaces['_infer_snap_indices'] = infer_snap_idx
    pf = _process_per_floor_surfaces(
        TB_q_snapped,
        TB,
        opts_surfaces,
        progress_cb,
        k_start=k_start,
        resume_state=resume_state,
        save_checkpoints=save_checkpoints,
        visual_cb=visual_cb,
    )
    result: Dict[str, Any] = {
        'TB': TB,
        'TB_q': TB_q_presnap,
        'TB_q_snapped': TB_q_snapped,
        'EX': pf['EX'],
        'EX_gl': pf['EX_gl'],
        'EX_glgr': pf['EX_glgr'],
        'EXOK': pf['EXOK'],
        'IN_gl': pf['IN_gl'],
        'IN_n': pf['IN_n'],
        'INOK': pf['INOK'],
        'final': pf['final'],
    }
    result['INM'] = pf['INM']
    result['INEXM'] = pf['INEXM']
    dm = pf.get('captured_surface_masks') or []
    if dm:
        result['captured_surface_masks'] = dm
    if pf.get('_checkpoints') is not None:
        result['_checkpoints'] = pf['_checkpoints']
    return result


def process_image(
    path: str,
    opts: dict,
    progress_cb: Optional[Callable] = None
) -> str:
    """
    Traite une image, la sauvegarde et retourne le chemin exporté.
    """
    img = charger_image(path)
    original_format = img.format

    output_path = opts.get('output_path')
    if not output_path:
        input_path = Path(path)
        output_path = str(input_path.parent / f"{input_path.stem}_processed{input_path.suffix}")

    results = process_image_stepwise(img, opts, progress_cb)
    final_img = results['final']

    preserve_format = opts.get('preserve_input_format', True)
    exported_path = exporter(
        final_img,
        output_path,
        preserve_input_format=preserve_format,
        original_format=original_format
    )

    return exported_path
