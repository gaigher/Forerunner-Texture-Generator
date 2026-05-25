# -*- coding: utf-8 -*-
from __future__ import annotations

"""
Construction du graphe logique du pipeline (nœuds, textes, masques par étage).

Principe unifié : **toute** logique surfaces (masques, ombres, blanchiments, checkpoints, sorties)
part de l’**image de base** du dessin (préparation, quantification, alignement sur les paliers Forerunner) et des **options** du rendu.
Sur le graphe, la chaîne **image de base → options pipeline** puis l’**éventail** des étages 1 à 5 matérialise ces données ;
le moteur enchaîne les paliers dans l’ordre croissant (ombres et blanchiments empilés par étage).
Référence : ``traitement._process_per_floor_surfaces``.
"""

import re
from typing import Any, Dict, List, Optional, Tuple

from effets.forerunner_floors import FORERUNNER_FLOOR_GRAYS, FORERUNNER_FLOOR_TITLES
from ui.settings import (
    DEFAULT_FUSION_FORCE_INEXM,
    DEFAULT_FUSION_FORCE_INM,
    DEFAULT_FUSION_MODE,
    UI_ANCRAGE_PIPELINE_COURT,
    UI_ANCRAGE_PIPELINE_LONG,
    UI_GRAPHE_TITRE_APRES_BLANCHIMENTS,
    UI_GRAPHE_TITRE_APRES_OMBRES,
    UI_GRAPHE_TITRE_IMAGE_BASE_AVANT_PALIERS,
    UI_GRAPHE_TITRE_IMAGE_BASE_PALIERS,
    UI_GRAPHE_TITRE_IMAGE_BASE_PREP,
    DEFAULT_GLOW_EX_INTENSITY,
    DEFAULT_GLOW_EX_RADIUS,
    DEFAULT_GLOW_EX_THRESHOLD,
    DEFAULT_GLOW_IN_INTENSITY,
    DEFAULT_GLOW_IN_RADIUS,
    DEFAULT_GLOW_IN_THRESHOLD,
    DEFAULT_MASQUE_BLANC_SEUIL,
    DEFAULT_MASQUE_NOIR_SEUIL,
    DEFAULT_SEUIL_FLOU,
    DEFAULT_SNAP_ONLY_SKIP_CONTOUR,
)

_ANCRAGE_TBQ_OPTS = UI_ANCRAGE_PIPELINE_COURT
_ANCRAGE_TBQ_OPTS_LONG = UI_ANCRAGE_PIPELINE_LONG


def _split_captured_masks_by_floor(
    captured_masks: List[Tuple[str, Any]],
) -> Dict[int, List[Tuple[str, Any]]]:
    """
    Regroupe les masques capturés par numéro d’étage (titres « Étage N … »).
    Les entrées sans numéro d'étage sont dans le groupe 0 (affichées sur la rangée 0).
    Ces clichés sont produits dans le moteur **à partir de l’image de base (paliers)** et **opts** (même run que le pipeline).
    """
    floor_re = re.compile(r"Étage\s+(\d+)", re.UNICODE)
    out: Dict[int, List[Tuple[str, Any]]] = {}
    for title, sim in captured_masks:
        m = floor_re.search(title or "")
        if m:
            k = int(m.group(1))
            out.setdefault(k, []).append((title, sim))
        else:
            out.setdefault(0, []).append((title, sim))
    return out


def _split_floor_masks_shadow_bleach(
    floor_masks: List[Tuple[str, Any]],
) -> Tuple[List[Tuple[str, Any]], List[Tuple[str, Any]]]:
    """
    Sépare les clichés debug chaîne ombre (vers palier récepteur) et chaîne blanchiment (palier courant)
    pour afficher deux chaînes parallèles reliées par un nœud de jonction sur le graphe.
    """
    shadow: List[Tuple[str, Any]] = []
    bleach: List[Tuple[str, Any]] = []
    for title, sim in floor_masks:
        t = title or ""
        if "ombre —" in t or "palier récepteur ombre" in t:
            shadow.append((title, sim))
        elif "blanchiment —" in t or "palier blanchiment" in t:
            bleach.append((title, sim))
    return shadow, bleach


def _floor_palier_short(title_floor: str) -> str:
    """Partie lisible du titre d’étage (ex. « Étage 5 — Blanc » → « Blanc »)."""
    if "—" in title_floor:
        return title_floor.split("—", 1)[1].strip()
    return title_floor.strip()


def _fmt_dict(d: Any, indent: int = 0) -> str:
    if d is None:
        return "(aucun)"
    if not isinstance(d, dict):
        return str(d)
    lines: List[str] = []
    pad = "  " * indent
    for k in sorted(d.keys(), key=lambda x: str(x)):
        v = d[k]
        if isinstance(v, dict):
            lines.append(f"{pad}{k}:")
            lines.append(_fmt_dict(v, indent + 1))
        else:
            lines.append(f"{pad}{k}: {v!r}")
    return "\n".join(lines) if lines else "{}"


def _native_pipeline_block() -> str:
    return (
        "Valeurs de référence (code / défauts settings.py) :\n"
        f"  fusion_mode: {DEFAULT_FUSION_MODE!r}\n"
        f"  fusion_force_inm (opacité ombre): {DEFAULT_FUSION_FORCE_INM}\n"
        f"  fusion_force_inexm (opacité blanchiment): {DEFAULT_FUSION_FORCE_INEXM}\n"
        f"  masque_blanc_seuil: {DEFAULT_MASQUE_BLANC_SEUIL}\n"
        f"  masque_noir_seuil: {DEFAULT_MASQUE_NOIR_SEUIL}\n"
        f"  seuil_flou (détection texture nette): {DEFAULT_SEUIL_FLOU}\n"
        f"  snap_only_skip_contour (défaut): {DEFAULT_SNAP_ONLY_SKIP_CONTOUR}\n"
    )


def _native_floor_block() -> str:
    return (
        "Référence glow / grain (defaults) :\n"
        f"  glow_ex: radius={DEFAULT_GLOW_EX_RADIUS}, threshold={DEFAULT_GLOW_EX_THRESHOLD}, "
        f"intensity={DEFAULT_GLOW_EX_INTENSITY}\n"
        f"  glow_in: radius={DEFAULT_GLOW_IN_RADIUS}, threshold={DEFAULT_GLOW_IN_THRESHOLD}, "
        f"intensity={DEFAULT_GLOW_IN_INTENSITY}\n"
    )


def _spoiler_global(opts: dict) -> str:
    po = opts.get("process_floor_indices")
    po_s = repr(list(po)) if po is not None else "None (tous les étages 1–5 éligibles)"
    return (
        _ANCRAGE_TBQ_OPTS_LONG
        + "Détails techniques (natif)\n"
        "— Le pipeline surfaces empile d’abord les calques d’ombre, puis les calques de blanchiment, "
        "sur la base métal ; la fusion « grain_merge » somme les écarts RVB dans l’ordre "
        "où les étages sont traités (ordre croissant des paliers).\n"
        "— process_floor_indices restreint quels étages sont recalculés ; les autres reçoivent "
        f"une empreinte __skipped__ côté moteur. Valeur actuelle : {po_s}\n"
        "— Les grain_* reçoivent une graine dérivée de _grain_seed_base si seed absent.\n"
    )


def _spoiler_floor(k: int) -> str:
    gk = FORERUNNER_FLOOR_GRAYS[k] if k < len(FORERUNNER_FLOOR_GRAYS) else "?"
    return (
        _ANCRAGE_TBQ_OPTS
        + f"Détails étage {k} (natif)\n"
        f"— Palier luminance Forerunner cible : L = {gk}.\n"
        "— Chaîne ombre : image de base alignée sur les paliers → glow sur cette image (si glow_on_tb) → masques manuels … → "
        "couleur vers alpha (blanc) pour la couche ombre fusionnée.\n"
        "— Chaîne blanchiment : inversion → glow → grains → masques … → "
        "couleur vers alpha (noir) pour la couche blanchiment fusionnée.\n"
        "— Si le cache calques (_floor_layer_cache) est valide, le moteur peut réutiliser "
        "les calques ombre/blanchiment fusionnés sans recalculer les N/B (message « cache calques »).\n"
    )


def list_graph_nodes(
    payload: dict,
    opts: Optional[dict],
) -> List[Dict[str, Any]]:
    """
    Nœuds du pipeline : id, parent_id, layout (main | branch | mask_chain | …).

    Layout ``input`` : image de base du dessin, texture métal (si présente) — pas de nœud séparé par étage : les réglages
    ``opts['floors'][k]`` sont décrits dans chaque branche ``n_sec_k`` (éventail).

    Tronc : image de base (avant / après alignement paliers) → **Options pipeline** → **éventail** : une branche par étage (``fan`` depuis
    ``n_opts``) ; masques debug et cliché « après blanchiments » sur chaque rayon ; puis sur le tronc horizontal
    : diagnostic ombre / blanchiment → états intermédiaires (après ombres, après blanchiments) → final — toujours **à partir de l’image de base + opts**.

    Les checkpoints « après blanchiments » par étage ne reçoivent **pas** d’arête ``base_feed`` : la base entre via l’**état après ombres**
    ; les blanchiments sont fusionnés cumulativement (pile déjà empilée). Seul le nœud de sortie
    **après blanchiments** final est relié à « Base surfaces » à titre de rappel du fond.

    ``input_channel`` : vrai pour les canaux d'entrée logiciel — image de base et fond surfaces (TM/blanc) uniquement.

    ``layout`` : ``input`` = entrées pipeline ; ``main`` = tronc ; ``fan`` = branche étage depuis ``n_opts``.
    """
    opts = opts or {}
    floors: List[dict] = list(opts.get("floors") or [])
    process_only = opts.get("process_floor_indices")
    tm = opts.get("tm")
    nodes: List[Dict[str, Any]] = []
    prev: Optional[str] = None

    def _append_input(spec: Dict[str, Any], nid: str) -> None:
        row = dict(spec)
        row["id"] = nid
        row["layout"] = "input"
        row["parent_id"] = None
        row["branch_anchor_id"] = None
        row.setdefault("input_channel", True)
        nodes.append(row)

    def _append(
        spec: Dict[str, Any],
        *,
        nid: str,
        spine: bool = True,
        branch_parent: Optional[str] = None,
        main_parent: Optional[str] = None,
    ) -> None:
        nonlocal prev
        row = dict(spec)
        row["id"] = nid
        if spine:
            row["layout"] = "main"
            row["parent_id"] = main_parent if main_parent is not None else prev
            row["branch_anchor_id"] = None
            prev = nid
        else:
            row["layout"] = "branch"
            row["parent_id"] = branch_parent
            row["branch_anchor_id"] = branch_parent
        nodes.append(row)

    def _append_mask_placeholder(
        spec: Dict[str, Any],
        nid: str,
        parent_id: str,
        *,
        mask_row: int = 0,
    ) -> None:
        row = dict(spec)
        row["id"] = nid
        row["layout"] = "mask_placeholder"
        row["parent_id"] = parent_id
        row["branch_anchor_id"] = None
        row["mask_row"] = mask_row
        nodes.append(row)

    def _append_mask_chain_row(
        spec: Dict[str, Any],
        nid: str,
        parent_id: str,
        mask_row: int,
    ) -> None:
        row = dict(spec)
        row["id"] = nid
        row["layout"] = "mask_chain"
        row["parent_id"] = parent_id
        row["branch_anchor_id"] = None
        row["mask_row"] = int(mask_row)
        nodes.append(row)

    def _append_mask_merge(
        spec: Dict[str, Any],
        nid: str,
        parent_id: str,
        merge_parent_ids: List[str],
        mask_row: int,
    ) -> None:
        row = dict(spec)
        row["id"] = nid
        row["layout"] = "mask_merge"
        row["parent_id"] = parent_id
        row["merge_parent_ids"] = list(merge_parent_ids)
        row["branch_anchor_id"] = None
        row["mask_row"] = int(mask_row)
        nodes.append(row)

    def _append_fan(spec: Dict[str, Any], nid: str, hub_id: str, *, lane_k: int) -> None:
        """Branche d’étage depuis le hub (éventail) ; ne met pas à jour ``prev`` du tronc."""
        row = dict(spec)
        row["id"] = nid
        row["layout"] = "fan"
        row["parent_id"] = hub_id
        row["branch_anchor_id"] = None
        row["fan_lane_k"] = int(lane_k)
        nodes.append(row)

    checkpoints: Dict[int, Any] = payload.get("_checkpoints") or {}
    captured_masks = list(payload.get("captured_masks") or [])
    masks_by_floor = _split_captured_masks_by_floor(captured_masks)
    snap0 = checkpoints.get(0)
    has_base_ckpt = bool(snap0 and snap0.get("INEXM") is not None)

    # Entrées pipeline — image de base, texture métal, puis une entrée par étage k (y compris inactifs)
    _append_input(
        {
            "type": "image",
            "title": UI_GRAPHE_TITRE_IMAGE_BASE_PREP,
            "image": payload.get("TB"),
            "user_text": (
                _ANCRAGE_TBQ_OPTS
                + "Entrée principale : image source après chargement / normalisation. "
                "Alimente la suite : quantification / paliers → options, puis l’éventail et les sorties."
            ),
            "native_text": "",
            "spoiler_text": "",
            "dead_end": False,
            "dead_reason": "",
            "order_note": "",
            "input_channel": True,
        },
        "n_tb",
    )
    if has_base_ckpt:
        im0 = snap0["INEXM"]
        _append_input(
            {
                "type": "image",
                "title": "Base surfaces — texture métal (TM) ou blanc (entrée pipeline)",
                "image": im0,
                "user_text": (
                    _ANCRAGE_TBQ_OPTS
                    + "Entrée : texture métallique (redimensionnée comme l’image de base à paliers) ou fond blanc si pas de TM. "
                    "Le moteur initialise les états **après ombres** et **après blanchiments** **sur cette base** avant la boucle par étage.\n\n"
                    "Sur le graphe, un fil « base_feed » va uniquement vers le nœud de sortie **"
                    + UI_GRAPHE_TITRE_APRES_BLANCHIMENTS
                    + "** (rappel : la composition finale **repose sur** ce fond). Les clichés intermédiaires **après blanchiments** "
                    "(un par étage traité) ne sont **pas** reliés à ce nœud : ils montrent l’état cumulé après ombres puis fusion "
                    "**de tous les blanchiments** des paliers déjà traités jusqu’à l’étage indiqué — la base entre dans le calcul "
                    "via l’image **après ombres** sur TM/blanc, puis `composite_grain_contributions_from_base(base_ombres, bleach_stack)` pour les blanchiments."
                ),
                "native_text": "",
                "spoiler_text": "",
                "dead_end": False,
                "dead_reason": "",
                "order_note": "Point de départ des piles shadow_stack et bleach_stack.",
                "input_channel": True,
            },
            "n_base_ckpt",
        )

    # Tronc de traitement : premier lien depuis l’image de base préparée
    _append(
        {
            "type": "image",
            "title": UI_GRAPHE_TITRE_IMAGE_BASE_AVANT_PALIERS,
            "image": payload.get("TB_q_presnap"),
            "user_text": (
                _ANCRAGE_TBQ_OPTS
                + "Étape dérivée de l’image de base préparée : prétraitement (contours, quantification) avant alignement sur les paliers. "
                "Enchaîne vers l’image alignée sur les paliers, puis les options (même jeu opts)."
            ),
            "native_text": "",
            "spoiler_text": "",
            "dead_end": False,
            "dead_reason": "",
            "order_note": "",
        },
        nid="n_tbq_pre",
        main_parent="n_tb",
    )
    _append(
        {
            "type": "image",
            "title": UI_GRAPHE_TITRE_IMAGE_BASE_PALIERS,
            "image": payload.get("TB_q_snapped"),
            "user_text": (
                _ANCRAGE_TBQ_OPTS_LONG
                + "Image de base après alignement des luminances sur les paliers Forerunner. "
                "Les étages 1 à 5 sont pilotés par opts et enchaînés dans l’ordre au moteur "
                "(fusion séquentielle des paliers). "
                "Sur le graphe : après « Options pipeline », l’éventail montre chaque étage en parallèle visuelle."
            ),
            "native_text": "",
            "spoiler_text": "",
            "dead_end": False,
            "dead_reason": "",
            "order_note": "",
        },
        nid="n_tbq_snap",
    )

    _opts_intro = (
        _ANCRAGE_TBQ_OPTS_LONG
        + "Ce nœud résume **opts** tel que lu pour ce rendu (image de base déjà alignée sur les paliers). "
        "Il ne calcule rien : carte d’identité du traitement (TM, fusion, périmètre des étages, "
        "graines de grain, etc.).\n\n"
        "Détail des clés :\n"
    )
    user_opts = (
        _opts_intro
        + f"Texture métallique (tm): {tm!r}\n"
        f"snap_only_skip_contour: {opts.get('snap_only_skip_contour')}\n"
        f"allow_conversion: {opts.get('allow_conversion')}\n"
        f"fusion_mode: {opts.get('fusion_mode')}\n"
        f"process_floor_indices: {process_only!r}\n"
        f"_grain_seed_base: {opts.get('_grain_seed_base')!r}\n"
    )
    _append(
        {
            "type": "code",
            "title": "Options pipeline (inscrites / effectives)",
            "image": None,
            "user_text": user_opts,
            "native_text": _native_pipeline_block(),
            "spoiler_text": _spoiler_global(opts),
            "dead_end": False,
            "dead_reason": "",
            "order_note": (
                "Hub graphe : toutes les branches par étage et les sorties diagnostic se rattachent à l’image de base (paliers) + options. "
                "Ordre moteur : shadow_stack puis bleach_stack pour les étages 1 à 5 ; à chaque étage, ombre puis blanchiment."
            ),
        },
        nid="n_opts",
    )

    if not captured_masks:
        _append_mask_placeholder(
            {
                "type": "code",
                "title": "Aucun masque capturé",
                "image": None,
                "user_text": (
                    _ANCRAGE_TBQ_OPTS
                    + "Activez « Diagnostic → Capturer masques et étapes (rendu plus lent) » "
                    "puis relancez un traitement (même image de base à paliers et options)."
                ),
                "native_text": "",
                "spoiler_text": "",
                "dead_end": False,
                "dead_reason": "",
                "order_note": "",
            },
            "n_dbg_masks_empty",
            "n_base_ckpt" if has_base_ckpt else "n_opts",
            mask_row=0,
        )

    # Masques sans préfixe « Étage k » (groupe 0) : chaîne sous la base ou les options
    bucket0 = masks_by_floor.get(0) or []
    if bucket0:
        prev_m = "n_base_ckpt" if has_base_ckpt else "n_opts"
        for i, (mtitle, sim) in enumerate(bucket0):
            nid = f"n_dbg_mask_0_{i}"
            _append_mask_chain_row(
                {
                    "type": "image",
                    "title": mtitle,
                    "image": sim,
                    "user_text": (
                        _ANCRAGE_TBQ_OPTS_LONG
                        + "Masque debug sans préfixe « Étage … » dans le titre : même image de base (paliers) et opts que le run ; "
                        "réassignation des indices (paliers exclus du pipeline via opts). "
                        "RVB intermédiaires souvent opaques ; RGBA (blanc/noir→α) en damier dans la visionneuse Qt."
                    ),
                    "native_text": "",
                    "spoiler_text": "",
                    "dead_end": False,
                    "dead_reason": "",
                    "order_note": "",
                },
                nid,
                prev_m,
                mask_row=0,
            )
            prev_m = nid

    for k in range(1, 6):
        title_floor = (
            FORERUNNER_FLOOR_TITLES[k]
            if k < len(FORERUNNER_FLOOR_TITLES)
            else f"Étage {k}"
        )
        fd = floors[k] if k < len(floors) else {}
        enabled = bool(fd.get("enabled", True))
        skipped = process_only is not None and k not in process_only

        if not enabled:
            sec_user = (
                _ANCRAGE_TBQ_OPTS
                + f"Branche surfaces · étage {k} ({_floor_palier_short(title_floor)}). "
                "Cet étage n’est pas traité : « Traiter cet étage » est décoché dans opts. "
                "Pas de contribution surfaces pour cet étage."
            )
            sec_native = ""
            sec_spoiler = _spoiler_floor(k)
            sec_order = ""
        elif skipped:
            sec_user = (
                _ANCRAGE_TBQ_OPTS
                + f"Branche surfaces · étage {k} ({_floor_palier_short(title_floor)}). "
                "Hors process_floor_indices pour ce rendu : pas de recalcul pour cet étage "
                "(empreinte __skipped__ ; opts restreint les étages actifs pour la même image de base à paliers)."
            )
            sec_native = f"process_floor_indices = {process_only!r}"
            sec_spoiler = _spoiler_floor(k)
            sec_order = ""
        else:
            user_fd = _fmt_dict(fd)
            sec_user = (
                _ANCRAGE_TBQ_OPTS_LONG
                + f"Branche surfaces · étage {k} · palier « {title_floor} » — opts['floors'][{k}] (même image de base à paliers).\n"
                "Ombres, blanchiment, grains, opacités pour ce palier.\n\n"
                "Valeurs actuelles :\n"
                + user_fd
                + "\n\nMasques debug et cliché intermédiaire « après blanchiments » sur ce rayon ; distinct du bloc « Diagnostic : chaîne ombre / blanchiment » sur le tronc."
            )
            sec_native = _native_floor_block()
            sec_spoiler = _spoiler_floor(k)
            sec_order = (
                "Lu dans opts après l’image de base alignée sur les paliers ; empilement moteur par palier croissant, ombre puis blanchiment à chaque étage."
            )

        _append_fan(
            {
                "type": "section",
                "title": (
                    f"═══ Branche surfaces · étage {k} · {_floor_palier_short(title_floor)} ═══"
                ),
                "image": None,
                "user_text": sec_user,
                "native_text": sec_native,
                "spoiler_text": sec_spoiler,
                "dead_end": False,
                "dead_reason": "",
                "order_note": sec_order,
            },
            f"n_sec_{k}",
            "n_opts",
            lane_k=k,
        )

        if not enabled:
            _append(
                {
                    "type": "dead",
                    "title": f"Étage {k} — traitement désactivé (cul-de-sac)",
                    "image": None,
                    "user_text": (
                        _ANCRAGE_TBQ_OPTS
                        + "opts['floors'][k].enabled=False : aucune ombre ni blanchiment pour ce palier."
                    ),
                    "native_text": "",
                    "spoiler_text": _spoiler_floor(k),
                    "dead_end": True,
                    "dead_reason": "enabled=False dans opts['floors'][k].",
                    "order_note": "",
                },
                nid=f"n_dead_dis_{k}",
                spine=False,
                branch_parent=f"n_sec_{k}",
            )
            continue

        if skipped:
            _append(
                {
                    "type": "dead",
                    "title": f"Étage {k} — exclu du périmètre (cul-de-sac)",
                    "image": None,
                    "user_text": (
                        _ANCRAGE_TBQ_OPTS
                        + "Hors process_floor_indices dans opts : pas de surfaces pour cet étage avec cette image de base à paliers."
                    ),
                    "native_text": f"process_floor_indices = {process_only!r}",
                    "spoiler_text": _spoiler_floor(k),
                    "dead_end": True,
                    "dead_reason": "Empreinte __skipped__ : étage ignoré pour ce rendu.",
                    "order_note": "",
                },
                nid=f"n_dead_skip_{k}",
                spine=False,
                branch_parent=f"n_sec_{k}",
            )
            continue

        ombrage_on = bool(fd.get("ombrage_enabled", True))
        fin_inm = float(fd.get("fusion_force_inm", DEFAULT_FUSION_FORCE_INM))
        glow_ex_on = fd.get("glow_ex_enabled")
        if glow_ex_on is None:
            glow_ex_on = fd.get("blanchiment_enabled", True)
        grain_b = fd.get("grain_blanchiment") or {}
        grain_a = fd.get("grain_affleurement") or {}
        grain_on = bool(grain_b.get("enabled")) or bool(grain_a.get("enabled"))
        fin_ex = float(fd.get("fusion_force_inexm", DEFAULT_FUSION_FORCE_INEXM))

        if not ombrage_on:
            _append(
                {
                    "type": "dead",
                    "title": f"Étage {k} — ombrage désactivé (cul-de-sac)",
                    "image": None,
                    "user_text": (
                        _ANCRAGE_TBQ_OPTS
                        + "opts['floors'][k].ombrage_enabled=False : pas de chaîne d’ombre ni calque ombre fusionné pour cet étage."
                    ),
                    "native_text": "ombrage_enabled=False",
                    "spoiler_text": "",
                    "dead_end": True,
                    "dead_reason": "ombrage_enabled=False",
                    "order_note": "",
                },
                nid=f"n_dead_omb_{k}",
                spine=False,
                branch_parent=f"n_sec_{k}",
            )
        elif fin_inm <= 0.0:
            _append(
                {
                    "type": "dead",
                    "title": f"Étage {k} — opacité ombre nulle (cul-de-sac)",
                    "image": None,
                    "user_text": (
                        _ANCRAGE_TBQ_OPTS
                        + "fusion_force_inm ≤ 0 dans opts pour cet étage : contribution ombre ignorée."
                    ),
                    "native_text": f"fusion_force_inm={fin_inm}",
                    "spoiler_text": "",
                    "dead_end": True,
                    "dead_reason": "Opacité des ombrages nulle pour ce calque.",
                    "order_note": "",
                },
                nid=f"n_dead_finm_{k}",
                spine=False,
                branch_parent=f"n_sec_{k}",
            )

        if not glow_ex_on and not grain_on:
            _append(
                {
                    "type": "dead",
                    "title": f"Étage {k} — blanchiment / grain désactivés (cul-de-sac)",
                    "image": None,
                    "user_text": (
                        "Glow de blanchiment et grains (blanchiment + affleurement) inactifs : "
                        "pas de calque blanchiment fusionné pour ce palier."
                    ),
                    "native_text": "glow_ex_enabled=False et grains désactivés",
                    "spoiler_text": "",
                    "dead_end": True,
                    "dead_reason": "Pas de calque blanchiment fusionné pour cet étage.",
                    "order_note": "",
                },
                nid=f"n_dead_gr_{k}",
                spine=False,
                branch_parent=f"n_sec_{k}",
            )
        elif fin_ex <= 0.0:
            _append(
                {
                    "type": "dead",
                    "title": f"Étage {k} — opacité blanchiment nulle (cul-de-sac)",
                    "image": None,
                    "user_text": (
                        _ANCRAGE_TBQ_OPTS
                        + "fusion_force_inexm ≤ 0 dans opts pour cet étage : pas de blanchiment fusionné."
                    ),
                    "native_text": f"fusion_force_inexm={fin_ex}",
                    "spoiler_text": "",
                    "dead_end": True,
                    "dead_reason": "Opacité du blanchiment nulle pour ce calque.",
                    "order_note": "",
                },
                nid=f"n_dead_fex_{k}",
                spine=False,
                branch_parent=f"n_sec_{k}",
            )

        # Masques N/B : deux chaînes parallèles (ombre 3.x → palier récepteur ; blanchiment 4.x → palier k),
        # puis jonction (fusion selon la carte des paliers / piles ombre + blanchiment) avant le cliché « après blanchiments ».
        floor_masks = masks_by_floor.get(k, [])
        shadow_m, bleach_m = _split_floor_masks_shadow_bleach(floor_masks)
        sec_id = f"n_sec_{k}"
        prev_mk = sec_id

        def _chain_masks(items: List[Tuple[str, Any]], lane: str) -> str:
            p = sec_id
            for i, (mtitle, sim) in enumerate(items):
                nid = f"n_dbg_mask_{k}_{lane}_{i}"
                _append_mask_chain_row(
                    {
                        "type": "image",
                        "title": mtitle,
                        "image": sim,
                        "user_text": (
                            _ANCRAGE_TBQ_OPTS_LONG
                            + "Même image de base (paliers) et réassignation des indices de palier (opts). Chaîne **"
                            + ("ombre" if lane == "in" else "blanchiment")
                            + "** en parallèle de l’autre branche ; la fusion selon la carte des paliers a lieu au nœud **jonction**. "
                            "Transparence (blanc/noir→α) en damier dans la visionneuse Qt."
                        ),
                        "native_text": "",
                        "spoiler_text": "",
                        "dead_end": False,
                        "dead_reason": "",
                        "order_note": "",
                    },
                    nid,
                    p,
                    mask_row=k,
                )
                p = nid
            return p

        if not shadow_m and not bleach_m and floor_masks:
            for i, (mtitle, sim) in enumerate(floor_masks):
                nid = f"n_dbg_mask_{k}_{i}"
                _append_mask_chain_row(
                    {
                        "type": "image",
                        "title": mtitle,
                        "image": sim,
                        "user_text": (
                            _ANCRAGE_TBQ_OPTS_LONG
                            + "Même image de base (paliers) et réassignation des indices (opts) ; étapes masques pour cet étage. "
                            "(Titres non reconnus pour le découpage ombre / blanchiment : chaîne plate.)"
                        ),
                        "native_text": "",
                        "spoiler_text": "",
                        "dead_end": False,
                        "dead_reason": "",
                        "order_note": "",
                    },
                    nid,
                    prev_mk,
                    mask_row=k,
                )
                prev_mk = nid
        else:
            last_in = _chain_masks(shadow_m, "in") if shadow_m else ""
            last_ex = _chain_masks(bleach_m, "ex") if bleach_m else ""
            if shadow_m and bleach_m:
                merge_id = f"n_dbg_merge_{k}"
                _append_mask_merge(
                    {
                        "type": "code",
                        "title": f"Étage {k} — jonction surfaces (piles ombre + blanchiment)",
                        "image": None,
                        "user_text": (
                            _ANCRAGE_TBQ_OPTS_LONG
                            + "À ce stade le moteur a produit le **calque d’ombre** (pour le palier qui reçoit l’ombre) "
                            "et le **calque de blanchiment** (pour le palier courant). Ils sont **empilés séparément** "
                            "(`shadow_stack` / `bleach_stack`) puis fusionnés sur la base : **après ombres** = ombres sur TM/blanc, "
                            "**après blanchiments** = blanchiments via `composite_grain_contributions_from_base(base_ombres, bleach_stack)` — "
                            "pas une fusion pixel-à-pixel des deux clichés ci-dessus, mais la même logique d’empilement que le run."
                        ),
                        "native_text": "",
                        "spoiler_text": "",
                        "dead_end": False,
                        "dead_reason": "",
                        "order_note": "Convergence des chaînes ombre et blanchiment avant le cliché intermédiaire « après blanchiments ».",
                    },
                    merge_id,
                    last_in,
                    [last_ex],
                    mask_row=k,
                )
                prev_mk = merge_id
            elif shadow_m:
                prev_mk = last_in
            elif bleach_m:
                prev_mk = last_ex
            else:
                prev_mk = sec_id

        snap_k = checkpoints.get(k)
        if snap_k and snap_k.get("INEXM") is not None:
            _append(
                {
                    "type": "image",
                    "title": f"{UI_GRAPHE_TITRE_APRES_BLANCHIMENTS} (cumul jusqu’à l’étage {k})",
                    "image": snap_k["INEXM"],
                    "user_text": (
                        _ANCRAGE_TBQ_OPTS
                        + "Instantané **après blanchiments**, cumulé jusqu’à cet étage : même formule que le moteur — "
                        "à ce stade l’image contient la **base + les ombres** des paliers jusqu’à l’étage indiqué, puis la fusion cumulative de **tous les "
                        "calques de blanchiment** sur la même plage (`composite_grain_contributions_from_base(base_ombres, bleach_stack)`). "
                        "La texture métal / blanc n’est pas un « parent » graphique de ce cliché : elle est déjà intégrée dans l’**état après ombres**.\n\n"
                        "Chaque étage traité produit un cliché intermédiaire ; le **dernier étage traité** correspond au dernier cliché "
                        "**avant** les nœuds de sortie globaux (après ombres / après blanchiments / final sur le tronc)."
                    ),
                    "native_text": "",
                    "spoiler_text": "",
                    "dead_end": False,
                    "dead_reason": "",
                    "order_note": (
                        "Cliché après cet étage ; les sorties globales du pipeline reflètent l’état **après tous les étages** traités."
                    ),
                },
                nid=f"n_ckpt_{k}",
                main_parent=prev_mk,
            )

    _out_user = {
        "INM": _ANCRAGE_TBQ_OPTS_LONG
        + "**Après ombres** : état **intermédiaire** une fois les **ombres** (masques et calques d’ombre) fusionnés sur la texture métal TM "
        "(ou blanc si pas de TM), **avant** tout blanchiment. "
        "Dans le moteur : base + contributions ombre (fusion « grains »). "
        "Utile en **diagnostic** ; ce n’est pas l’image finale exportée.\n\n"
        "Sur le graphe, ce nœud suit la **chaîne de traitement** (image de base à paliers / opts / surfaces) ; la base TM/blanc entre ici "
        "dans la logique moteur, pas via un fil séparé « Base surfaces » (réservé à la sortie **après blanchiments**).",
        "INEXM": _ANCRAGE_TBQ_OPTS_LONG
        + "**Après blanchiments** : état une fois les **blanchiments** (calques clairs) fusionnés **sur** l’image déjà traitée côté ombres, "
        "toujours relativement à la **base** TM/blanc. À la fin du run, c’est le résultat **cumulé sur tous les étages** "
        "traités dans ce rendu (pas seulement un seul étage).\n\n"
        "Le fil depuis « Base surfaces » vers ce nœud signifie : la composition **repose sur** ce fond — ce n’est pas "
        "une passe unique « base → image finale », mais *toutes* les étapes surfaces utilisent ce support.",
        "final": _ANCRAGE_TBQ_OPTS_LONG
        + "Image finale RGB exportable ; cohérente avec cette image de base à paliers et ce snapshot d’opts.",
    }
    for key, title, nid in [
        ("INM", UI_GRAPHE_TITRE_APRES_OMBRES, "n_out_inm"),
        ("INEXM", UI_GRAPHE_TITRE_APRES_BLANCHIMENTS, "n_out_inexm"),
        ("final", "Résultat final (RGB)", "n_out_final"),
    ]:
        im = payload.get(key)
        if im is not None:
            _notes = {
                "INM": "Diagnostic : ombres sur TM/blanc, avant blanchiments.",
                "INEXM": "Diagnostic : ombres et blanchiments cumulés (tous les étages du run) ; proche du dernier cliché intermédiaire si tout a tourné.",
                "final": "Texture finale affichée / exportée.",
            }
            _append(
                {
                    "type": "image",
                    "title": title,
                    "image": im,
                    "user_text": _out_user[key],
                    "native_text": "",
                    "spoiler_text": "",
                    "dead_end": False,
                    "dead_reason": "",
                    "order_note": _notes[key],
                },
                nid=nid,
            )

    return nodes


def estimate_node_height(node: Dict[str, Any]) -> int:
    # Hauteurs alignées sur la visionneuse Qt (vignette ~200 px + bloc texte + chrome + ports).
    t = node.get("type")
    if t == "section":
        return 56 if node.get("layout") == "mask_header" else 200
    if t == "dead":
        return 200
    if t == "code":
        if node.get("layout") == "input":
            tit = (node.get("title") or "")
            if "inactif" in tit or "hors périmètre" in tit:
                return 200
        return 380 if node.get("id") == "n_opts" else 340
    if t == "image":
        return 320
    return 140


def graph_edges_from_nodes(nodes: List[Dict[str, Any]]) -> List[Tuple[str, str, str]]:
    """Types d’arêtes : main_h, fan_out, mask_*, branch_side, base_feed."""
    out: List[Tuple[str, str, str]] = []
    by_id = {n["id"]: n for n in nodes}
    for n in nodes:
        pid = n.get("parent_id")
        if not pid:
            continue
        lay = n.get("layout")
        parent = by_id.get(pid, {})
        pl = parent.get("layout")
        if lay == "branch":
            kind = "branch_side"
        elif lay == "fan":
            kind = "fan_out"
        elif lay == "mask_header":
            kind = "mask_drop"
        elif lay == "mask_chain":
            if pl == "mask_header":
                kind = "mask_drop"
            elif pl in ("main", "input"):
                kind = "mask_drop"
            elif pl == "fan":
                kind = "mask_h"
            else:
                kind = "mask_h"
        elif lay == "mask_placeholder":
            kind = "mask_drop"
        elif lay == "mask_merge":
            kind = "mask_h"
        else:
            kind = "main_h"
        out.append((pid, n["id"], kind))

    # Deuxième parent des nœuds jonction (chaîne blanchiment → fusion).
    for n in nodes:
        if n.get("layout") != "mask_merge":
            continue
        nid = n["id"]
        for mp in n.get("merge_parent_ids") or []:
            out.append((mp, nid, "mask_h"))

    # Base surfaces → sortie « après blanchiments » finale uniquement (pas les checkpoints par étage : la base est déjà dans l’état après ombres en amont).
    base_id = "n_base_ckpt"
    if base_id in by_id:
        if "n_out_inexm" in by_id:
            out.append((base_id, "n_out_inexm", "base_feed"))

    return out


def _resolve_node_overlaps(
    positions: Dict[str, Tuple[float, float]],
    sizes: Dict[str, Tuple[float, float]],
    *,
    margin: float = 24.0,
    passes: int = 5,
) -> None:
    """Repousse légèrement les nœuds dont les boîtes se chevauchent (pas d’API de layout dans NodeGraphQt)."""
    ids = list(positions.keys())
    for _ in range(passes):
        moved = False
        for i, a in enumerate(ids):
            ax, ay = positions[a]
            aw, ah = sizes[a]
            for b in ids[i + 1 :]:
                bx, by = positions[b]
                bw, bh = sizes[b]
                if (
                    ax < bx + bw + margin
                    and ax + aw + margin > bx
                    and ay < by + bh + margin
                    and ay + ah + margin > by
                ):
                    dy = (ay + ah + margin) - by
                    if dy > 0.5:
                        positions[b] = (bx, by + dy)
                        moved = True
        if not moved:
            break


def _linear_chain_edges(edges: List[Tuple[str, str, str]], kind: str) -> List[Tuple[str, str]]:
    """Arêtes ``kind`` sans bifurcation (un seul fil sortant du parent, un seul entrant sur l’enfant)."""
    filtered = [(s, d) for s, d, k in edges if k == kind]
    if not filtered:
        return []
    out_n: Dict[str, int] = {}
    in_n: Dict[str, int] = {}
    for s, d in filtered:
        out_n[s] = out_n.get(s, 0) + 1
        in_n[d] = in_n.get(d, 0) + 1
    return [
        (s, d)
        for s, d in filtered
        if out_n.get(s, 0) == 1 and in_n.get(d, 0) == 1
    ]


def _place_base_surfaces_from_children(
    nodes: List[Dict[str, Any]],
    positions: Dict[str, Tuple[float, float]],
    sizes: Dict[str, Tuple[float, float]],
    edges: List[Tuple[str, str, str]],
    *,
    gap: float,
    x0_min: float,
    set_size: Any,
) -> None:
    """
    Place ``n_base_ckpt`` à gauche du bloc des nœuds cibles ``base_feed``, centrée verticalement
    sur leur encombrement — plus de colonne d’entrée figée au bord de la scène.
    """
    base_id = "n_base_ckpt"
    by_id = {n["id"]: n for n in nodes}
    if base_id not in by_id:
        return
    children = [d for s, d, k in edges if s == base_id and k == "base_feed"]
    children = [d for d in children if d in positions]
    if not children:
        return
    set_size(base_id)
    bw, bh = sizes[base_id]
    min_cx = min(positions[d][0] for d in children)
    min_top = min(positions[d][1] for d in children)
    max_bot = max(positions[d][1] + sizes[d][1] for d in children)
    cy = (min_top + max_bot) / 2.0
    x = min_cx - gap - bw
    if x < x0_min:
        x = x0_min
    positions[base_id] = (x, cy - bh / 2.0)


def _layout_merge_fanin(
    nodes: List[Dict[str, Any]],
    positions: Dict[str, Tuple[float, float]],
    sizes: Dict[str, Tuple[float, float]],
    edges: List[Tuple[str, str, str]],
    *,
    gap: float,
    stack_gap: float,
    set_size: Any,
) -> None:
    """
    Nœuds avec plusieurs fils entrants (fusion « fan-in » vers un même enfant) : parents déplaçables alignés **en colonne**
    (même X), enfant à droite avec ``gap``, centré sur l’axe vertical du groupe de parents.

    Ne **repositionne pas** un parent qui est déjà source d’une arête vers un nœud placé **avant**
    l’enfant courant dans l’ordre de ``list_graph_nodes`` (lecture de la chaîne).
    """
    node_index = {n["id"]: i for i, n in enumerate(nodes)}
    by_id = {n["id"]: n for n in nodes}
    incoming: Dict[str, List[str]] = {}
    for s, d, k in edges:
        # ``base_feed`` est une dépendance de fond (annotation de support TM/blanc),
        # pas une vraie jonction géométrique à packer comme un fan-in.
        # Si on l'inclut, des nœuds de sortie (ex. n_out_inexm) sont traités
        # comme fusions multi-parents et perdent leur alignement de chaîne.
        if k == "base_feed":
            continue
        incoming.setdefault(d, []).append(s)

    merge_children = [d for d, srcs in incoming.items() if len(srcs) > 1]
    merge_children.sort(key=lambda nid: node_index.get(nid, 10**9))

    def _tbr(pid: str) -> Tuple[float, float, float]:
        x, y = positions[pid]
        w, h = sizes[pid]
        return y, y + h, x + w

    def _pack_boxes_around_center(ids: List[str], center_y: float, gap_y: float) -> None:
        """
        Packing vertical simple par boîtes : place les nœuds en pile autour de ``center_y``
        sans recouvrement (hauteurs réelles + espacement ``gap_y``).
        """
        if not ids:
            return
        total_h = sum(sizes[p][1] for p in ids) + gap_y * (len(ids) - 1)
        y0 = center_y - total_h / 2.0
        for p in ids:
            w, h = sizes[p]
            x_old, _ = positions[p]
            positions[p] = (x_old, y0)
            y0 += h + gap_y

    for c in merge_children:
        if c not in positions:
            continue
        c_layout = (by_id.get(c) or {}).get("layout")
        idx_c = node_index.get(c, 10**9)
        locked_sources: set = set()
        for s, d, k in edges:
            if k == "base_feed":
                continue
            if d == c:
                continue
            if node_index.get(d, 10**9) < idx_c:
                locked_sources.add(s)

        raw = incoming[c]
        preds = list(dict.fromkeys(raw))
        preds.sort(key=lambda pid: node_index.get(pid, 10**9))
        if not preds:
            continue

        for p in preds:
            set_size(p)
        set_size(c)

        # Cas des jonctions de chaînes masques (ombre / blanchiment) :
        # on ne repositionne PAS les parents ; la jonction se place à droite,
        # et en Y sur la moyenne des centres des parents.
        if c_layout == "mask_merge":
            # Pour les jonctions de chaînes masques, ne pas laisser une chaîne
            # fortement ramifiée contraindre l'autre en vertical :
            # - X : à droite du parent le plus avancé
            # - Y : moyenne des centres des parents.
            max_rx = max(_tbr(p)[2] for p in preds)
            cy = sum((_tbr(p)[0] + _tbr(p)[1]) / 2.0 for p in preds) / len(preds)
            _cw, ch = sizes[c]
            positions[c] = (max_rx + gap, cy - ch / 2.0)
            continue

        base_id = "n_base_ckpt"
        movable = [p for p in preds if p not in locked_sources and p != base_id]
        locked_here = [p for p in preds if p in locked_sources or p == base_id]

        if not movable:
            min_top = min(_tbr(p)[0] for p in preds)
            max_bot = max(_tbr(p)[1] for p in preds)
            max_rx = max(_tbr(p)[2] for p in preds)
            cy = (min_top + max_bot) / 2.0
            _cw, ch = sizes[c]
            positions[c] = (max_rx + gap, cy - ch / 2.0)
            continue

        if locked_here:
            max_rx_l = max(_tbr(p)[2] for p in locked_here)
            x_stack = max_rx_l + gap
            cy_ref = (min(_tbr(p)[0] for p in locked_here) + max(_tbr(p)[1] for p in locked_here)) / 2.0
        else:
            x_stack = min(positions[p][0] for p in preds)
            cy_ref = sum((_tbr(p)[0] + _tbr(p)[1]) / 2.0 for p in preds) / len(preds)

        for p in movable:
            w, h = sizes[p]
            _x_old, _y_old = positions[p]
            positions[p] = (x_stack, _y_old)
        _pack_boxes_around_center(movable, cy_ref, stack_gap)

        # Jonction sur la moyenne des parents (et non min/max), puis placement à droite.
        max_rx = max(_tbr(p)[2] for p in preds)
        cy = sum((_tbr(p)[0] + _tbr(p)[1]) / 2.0 for p in preds) / len(preds)
        _cw, ch = sizes[c]
        positions[c] = (max_rx + gap, cy - ch / 2.0)


def _layout_bifurcation_on_children_mean(
    positions: Dict[str, Tuple[float, float]],
    sizes: Dict[str, Tuple[float, float]],
    edges: List[Tuple[str, str, str]],
) -> None:
    """
    Positionne verticalement une bifurcation (parent avec >=2 sorties) sur la
    moyenne des centres de ses enfants, pour stabiliser les embranchements.
    """
    out_map: Dict[str, List[str]] = {}
    for s, d, k in edges:
        # On couvre toutes les ramifications de lecture usuelles du graphe
        # (y compris les chaînes de commentaires/explications), sauf la dépendance base_feed.
        if k not in ("branch_side", "mask_h", "mask_drop", "fan_out", "main_h"):
            continue
        out_map.setdefault(s, []).append(d)

    for p, kids in out_map.items():
        if len(kids) < 2 or p not in positions or p not in sizes:
            continue
        kids = [k for k in kids if k in positions and k in sizes]
        if len(kids) < 2:
            continue
        cy = sum(positions[k][1] + sizes[k][1] / 2.0 for k in kids) / len(kids)
        px, _py = positions[p]
        _pw, ph = sizes[p]
        positions[p] = (px, cy - ph / 2.0)


def _layout_macro_blocks_from_active_lanes(
    positions: Dict[str, Tuple[float, float]],
    sizes: Dict[str, Tuple[float, float]],
    *,
    gap_left_to_sections: float,
) -> None:
    """
    Ajustements macro inspirés des corrections manuelles:
    - bloc tronc (image de base → paliers → opts) à gauche des sections d'étage;
    - tronc centré verticalement sur la zone des sections;
    - base/out alignés sur le dernier checkpoint actif.
    """
    sec_ids = [nid for nid in positions if nid.startswith("n_sec_")]
    if sec_ids:
        sec_min_x = min(positions[nid][0] for nid in sec_ids)
        sec_min_y = min(positions[nid][1] for nid in sec_ids)
        sec_max_y = max(positions[nid][1] for nid in sec_ids)
    else:
        sec_min_x = None
        sec_min_y = None
        sec_max_y = None

    trunk_ids = [
        nid
        for nid in ("n_tb", "n_tbq_pre", "n_tbq_snap")
        if nid in positions and nid in sizes
    ]
    if trunk_ids and sec_min_x is not None:
        trunk_max_r = max(positions[nid][0] + sizes[nid][0] for nid in trunk_ids)
        desired_max_r = sec_min_x - gap_left_to_sections
        dx = desired_max_r - trunk_max_r
        if abs(dx) > 0.5:
            for nid in trunk_ids:
                x, y = positions[nid]
                positions[nid] = (x + dx, y)

    if trunk_ids and sec_min_y is not None and sec_max_y is not None:
        # Centrage vertical du tronc sur la médiane de la zone sections.
        target_y = sec_min_y + 0.50 * (sec_max_y - sec_min_y)
        cur_y = sum(positions[nid][1] for nid in trunk_ids) / float(len(trunk_ids))
        dy = target_y - cur_y
        if abs(dy) > 0.5:
            for nid in trunk_ids:
                x, y = positions[nid]
                positions[nid] = (x, y + dy)

    ckpt_ids = sorted(
        [nid for nid in positions if nid.startswith("n_ckpt_") and nid in sizes]
    )
    if ckpt_ids:
        y_term = max(positions[nid][1] for nid in ckpt_ids)
        for nid in ("n_base_ckpt", "n_out_inm", "n_out_inexm", "n_out_final"):
            if nid in positions:
                x, _y = positions[nid]
                positions[nid] = (x, y_term)


def _layout_branch_side_linear(
    positions: Dict[str, Tuple[float, float]],
    sizes: Dict[str, Tuple[float, float]],
    edges: List[Tuple[str, str, str]],
    *,
    gap: float,
) -> None:
    """
    Arêtes ``branch_side`` sans bifurcation : enfant à droite du parent, **centres verticaux alignés**
    (fil horizontal out→in entre milieux de bord).
    """
    linear = _linear_chain_edges(edges, "branch_side")
    for s, d in linear:
        if s not in positions or d not in positions:
            continue
        sx, sy = positions[s]
        sw, sh = sizes[s]
        _, dh = sizes[d]
        positions[d] = (sx + sw + gap, sy + sh / 2.0 - dh / 2.0)


def _layout_linear_chains_horizontal(
    positions: Dict[str, Tuple[float, float]],
    sizes: Dict[str, Tuple[float, float]],
    edges: List[Tuple[str, str, str]],
    *,
    kind: str,
    gap: float,
) -> None:
    """
    Chaînes **sans bifurcation** (un seul fil ``kind`` sortant du parent, un seul entrant sur l’enfant) :
    même ordonnée que la tête de chaîne, et espacement horizontal minimal ``gap`` entre bord droit du
    parent et bord gauche de l’enfant (fil out→in lisible).
    """
    linear = _linear_chain_edges(edges, kind)
    if not linear:
        return
    succ = {s: d for s, d in linear}
    preds = {d: s for s, d in linear}
    nodes_in = set(succ.keys()) | set(succ.values())
    heads = [n for n in nodes_in if n not in preds]

    for h in heads:
        if h not in positions:
            continue
        hy = positions[h][1]
        cur = h
        while cur in succ:
            nxt = succ[cur]
            if cur not in positions or nxt not in positions:
                break
            cx, _ = positions[cur]
            cw, _ = sizes[cur]
            new_x = cx + cw + gap
            positions[nxt] = (new_x, hy)
            cur = nxt


def _enforce_fan_section_dx(
    positions: Dict[str, Tuple[float, float]],
    edges: List[Tuple[str, str, str]],
    *,
    hub_id: str,
    fan_section_dx: float,
) -> None:
    """
    Force un décalage horizontal final entre le hub fan (n_opts) et chaque
    section n_sec_k, en décalant la section et tout son sous-graphe.
    """
    if hub_id not in positions:
        return
    hub_x, _ = positions[hub_id]
    out_map: Dict[str, List[str]] = {}
    for s, d, _k in edges:
        out_map.setdefault(s, []).append(d)

    for k in range(1, 6):
        sec_id = f"n_sec_{k}"
        if sec_id not in positions:
            continue
        target_x = hub_x + fan_section_dx
        cur_x, _cur_y = positions[sec_id]
        dx = target_x - cur_x
        if abs(dx) < 0.5:
            continue

        stack = [sec_id]
        seen: set = set()
        while stack:
            nid = stack.pop()
            if nid in seen:
                continue
            seen.add(nid)
            if nid in positions:
                x, y = positions[nid]
                positions[nid] = (x + dx, y)
            for ch in out_map.get(nid, []):
                # On limite au sous-graphe de la branche d'étage.
                if ch.startswith(
                    (
                        f"n_dbg_mask_{k}_",
                        f"n_dbg_merge_{k}",
                        f"n_ckpt_{k}",
                        f"n_dead_dis_{k}",
                        f"n_dead_skip_{k}",
                        f"n_dead_omb_{k}",
                        f"n_dead_finm_{k}",
                        f"n_dead_gr_{k}",
                        f"n_dead_fex_{k}",
                    )
                ) or ch == sec_id:
                    stack.append(ch)


def _layout_terminal_surfaces_junction(
    positions: Dict[str, Tuple[float, float]],
    sizes: Dict[str, Tuple[float, float]],
    *,
    chain_gap: float,
    junction_gap: float,
) -> None:
    """
    Règles de placement explicites pour la fin de chaîne surfaces :
    - n_ckpt_5 -> n_out_inm : espacement horizontal ``chain_gap``
    - n_out_inm et n_base_ckpt en colonne (même X), espacés verticalement de ``chain_gap``
    - n_out_inexm à ``junction_gap`` horizontal des parents, centré verticalement entre eux
    - n_out_final à ``chain_gap`` de n_out_inexm, aligné horizontalement (même Y)
    """
    req = ("n_ckpt_5", "n_out_inm", "n_base_ckpt", "n_out_inexm", "n_out_final")
    if not all(nid in positions and nid in sizes for nid in req):
        return

    ck_x, ck_y = positions["n_ckpt_5"]
    out_inm_w, out_inm_h = sizes["n_out_inm"]
    base_w, base_h = sizes["n_base_ckpt"]
    out_inexm_w, out_inexm_h = sizes["n_out_inexm"]
    _out_final_w, _out_final_h = sizes["n_out_final"]

    # 1) Chaîne depuis le dernier checkpoint vers "après ombres".
    out_inm_x = ck_x + chain_gap
    out_inm_y = ck_y
    positions["n_out_inm"] = (out_inm_x, out_inm_y)

    # 2) Parent "base surfaces" en colonne avec "après ombres" (espacement vertical standard).
    # On place la base sous "après ombres" pour une lecture stable et répétable.
    base_x = out_inm_x
    base_y = out_inm_y + chain_gap
    positions["n_base_ckpt"] = (base_x, base_y)

    # 3) Jonction "après blanchiments" : centrée verticalement entre ses 2 parents.
    p1_cy = out_inm_y + out_inm_h / 2.0
    p2_cy = base_y + base_h / 2.0
    junction_cy = (p1_cy + p2_cy) / 2.0
    out_inexm_x = out_inm_x + junction_gap
    out_inexm_y = junction_cy - out_inexm_h / 2.0
    positions["n_out_inexm"] = (out_inexm_x, out_inexm_y)

    # 4) Résultat final : suite directe, même Y.
    out_final_x = out_inexm_x + chain_gap
    out_final_y = out_inexm_y
    positions["n_out_final"] = (out_final_x, out_final_y)


def layout_pipeline_graph(
    nodes: List[Dict[str, Any]],
    *,
    node_width: int = 300,
    main_y: float = 48.0,
    mask_y: float = 600.0,
    dx_main: float = 500.0,
    dx_mask: float = 500.0,
    x0: float = 40.0,
    main_offset_x: float = 540.0,
    input_dy: float = 500.0,
    fan_ckpt_gap: float = 1500.0,
    fan_mask_row_gap: float = 1500.0,
    fan_hub_below_gap: float = 1500.0,
    fan_column_top_gap: float = 1500.0,
    fan_section_dx: float = 1500.0,
    fan_section_gap: float = 1500.0,
    fan_mask_h_gap: float = 1500.0,
    fan_parallel_lane_dy: float = 1500.0,
    mask_row_dy: float = 500.0,
    overlap_margin: float = 500.0,
    overlap_passes: int = 5,
    input_gap_to_main: float = 500.0,
    linear_chain_gap: float = 500.0,
    merge_gap_from_parents: float = 1500.0,
    merge_stack_gap: float = 1500.0,
    branch_gap_from_parent: float = 1500.0,
    branch_gap_sibling: float = 1500.0,
) -> Tuple[Dict[str, Tuple[float, float]], Dict[str, Tuple[float, float]]]:
    """
    Placement des nœuds pour la visionneuse Qt (NodeGraphQt n’expose pas de grille auto).

    Espacements paramétrables par type de structure :
    - Chaînes simples : ``linear_chain_gap``
    - Jonctions (fan-in) : ``merge_gap_from_parents``, ``merge_stack_gap``
    - Bifurcations / cul-de-sac : ``branch_gap_from_parent``, ``branch_gap_sibling``
    - Éventail des sections d'étage : paramètres ``fan_*``
    Le placement des sections par étage (≥1) est une colonne dédiée sous le hub.

    - ``input`` : entrée image de base — colonne à gauche du tronc (``input_gap_to_main``). « Base surfaces »
      (``n_base_ckpt``) n’y figure pas : placée **après** le reste, à gauche de l’ensemble des nœuds
      cibles des ``base_feed``, selon ``linear_chain_gap``.
    - Tronc ``main`` : image de base (paliers) → options → sorties.
    - Sections « Branche surfaces » (depuis ``n_opts``) : **colonne** décalée horizontalement de
      ``fan_section_dx`` depuis l’axe du hub ; masques
      et cliché intermédiaire « après blanchiments » sur une rangée horizontale à droite de chaque section ; cul-de-sacs empilés
      à droite de la section si plusieurs.
    - Masques groupe 0 : rangée sous le tronc.
    - Branches cul-de-sac : à droite du parent ; pile **verticale** centrée sur l’axe vertical du parent
      (``branch_gap_from_parent``, ``branch_gap_sibling``).

    Chaînes **sans bifurcation** (un seul fil ``main_h`` / ``mask_h`` sortant / entrant) : après
    résolution des chevauchements, réalignement horizontal avec espacement ``linear_chain_gap`` entre
    nœuds consécutifs (voir ``_layout_linear_chains_horizontal``). Cul-de-sac seul : ``branch_side``
    (voir ``_layout_branch_side_linear``). **Fusion fan-in** (plusieurs parents vers un enfant) :
    ``_layout_merge_fanin`` piloté par ``merge_gap_from_parents`` et ``merge_stack_gap``.
    """
    positions: Dict[str, Tuple[float, float]] = {}
    sizes: Dict[str, Tuple[float, float]] = {}
    by_id = {n["id"]: n for n in nodes}
    use_fan = any(n.get("layout") == "fan" for n in nodes)

    branch_by_parent: Dict[str, List[str]] = {}
    for n in nodes:
        if n.get("layout") != "branch":
            continue
        pid = n.get("parent_id")
        if pid:
            branch_by_parent.setdefault(pid, []).append(n["id"])

    def set_size(nid: str) -> None:
        n = by_id[nid]
        h = float(estimate_node_height(n))
        w = float(node_width)
        sizes[nid] = (w, h)

    # Entrées : image de base (et entrées hors fond TM) — colonne à gauche du tronc. ``n_base_ckpt`` est exclu :
    # multi-parents ``base_feed`` → position dérivée des enfants (voir fin du layout).
    main_start_x = x0 + main_offset_x
    input_x = main_start_x - float(node_width) - input_gap_to_main
    if input_x < x0:
        input_x = x0
    iy = main_y
    for n in nodes:
        if n.get("layout") != "input":
            continue
        nid = n["id"]
        if nid == "n_base_ckpt":
            continue
        set_size(nid)
        positions[nid] = (input_x, iy)
        iy += sizes[nid][1] + input_dy

    input_column_bottom = iy

    # Tronc principal — pas les checkpoints d’étage (placés sur les rayons de l’éventail)
    x = main_start_x
    for n in nodes:
        if n.get("layout") != "main":
            continue
        nid = n["id"]
        if use_fan and nid.startswith("n_ckpt_"):
            continue
        set_size(nid)
        positions[nid] = (x, main_y)
        x += dx_main

    # Rangées masques (groupe 0 uniquement si éventail : les étages k≥1 suivent les rayons)
    x_snap = positions.get("n_tbq_snap", (main_start_x + dx_main, main_y))[0]
    mask_y_use = max(mask_y, input_column_bottom + 96.0)
    dy_mask_row = mask_row_dy
    mask_row_mx: Dict[int, float] = {}
    for n in nodes:
        lay = n.get("layout")
        if lay not in ("mask_header", "mask_chain", "mask_placeholder"):
            continue
        nid = n["id"]
        floor_k = int(n.get("mask_row") or 0)
        if use_fan and lay == "mask_chain" and floor_k >= 1:
            continue
        set_size(nid)
        my = mask_y_use + floor_k * dy_mask_row
        if floor_k not in mask_row_mx:
            mask_row_mx[floor_k] = x_snap
        mx = mask_row_mx[floor_k]
        positions[nid] = (mx, my)
        mask_row_mx[floor_k] = mx + dx_mask

    # Colonne « Branche surfaces · k » sous le hub (même X) ; masques + ckpt à droite ; cul-de-sac intégrés.
    hub_id = "n_opts"
    if use_fan and hub_id in positions:
        px, py = positions[hub_id]
        pw, ph = sizes[hub_id]
        hub_cx = px + pw / 2.0
        hub_by = py + ph + fan_hub_below_gap
        fan_col_x = hub_cx - float(node_width) / 2.0 + fan_section_dx
        # Le gap vertical "hub -> première section" doit rester lisible et piloté
        # par une seule valeur effective (sinon fan_hub_below_gap + fan_column_top_gap
        # se cumulent et éloignent excessivement les branches).
        y_sec = hub_by + max(0.0, fan_column_top_gap - fan_hub_below_gap)

        def _fan_row_mask_ids(k: int) -> Tuple[List[str], List[str], Optional[str]]:
            ins: List[str] = []
            i = 0
            while True:
                mid = f"n_dbg_mask_{k}_in_{i}"
                if mid not in by_id:
                    break
                ins.append(mid)
                i += 1
            exs: List[str] = []
            i = 0
            while True:
                mid = f"n_dbg_mask_{k}_ex_{i}"
                if mid not in by_id:
                    break
                exs.append(mid)
                i += 1
            mj = f"n_dbg_merge_{k}"
            merge = mj if mj in by_id else None
            if not ins and not exs:
                flat_mask_ids: List[str] = []
                j = 0
                while True:
                    mid = f"n_dbg_mask_{k}_{j}"
                    if mid not in by_id:
                        break
                    flat_mask_ids.append(mid)
                    j += 1
                if flat_mask_ids:
                    return flat_mask_ids, [], merge
            return ins, exs, merge

        for k in range(1, 6):
            sec = f"n_sec_{k}"
            masks_in, masks_ex, merge_id = _fan_row_mask_ids(k)
            ckpt_id = f"n_ckpt_{k}"
            ckpt = ckpt_id if ckpt_id in by_id else None

            if sec not in by_id:
                continue

            set_size(sec)
            sw, sh = sizes[sec]
            positions[sec] = (fan_col_x, y_sec)
            sec_x, sec_y = positions[sec]
            sec_cy = sec_y + sh / 2.0

            # Cul-de-sac d’abord (fil branch_side depuis la section), puis masques + ckpt sans chevauchement.
            cids = branch_by_parent.get(sec, [])
            branch_right = sec_x + sw
            if cids:
                px_s, py_s = sec_x, sec_y
                pw_s, ph_s = sw, sh
                pcy_s = py_s + ph_s / 2.0
                for cid in cids:
                    set_size(cid)
                gap_h = branch_gap_from_parent
                gap_v = branch_gap_sibling
                total_h = sum(sizes[cid][1] for cid in cids) + gap_v * (len(cids) - 1)
                stack_top = pcy_s - total_h / 2.0
                cur_y = stack_top
                base_x = px_s + pw_s + gap_h
                for cid in cids:
                    bw, bh = sizes[cid]
                    positions[cid] = (base_x, cur_y)
                    branch_right = max(branch_right, base_x + bw)
                    cur_y += bh + gap_v

            cur_x_base = branch_right + fan_mask_row_gap
            lane_half = float(fan_parallel_lane_dy) / 2.0

            def _place_lane(mids: List[str], y_cy: float) -> float:
                cx = cur_x_base
                for i, mid in enumerate(mids):
                    set_size(mid)
                    mw, mh = sizes[mid]
                    positions[mid] = (cx, y_cy - mh / 2.0)
                    cx += mw + (fan_mask_h_gap if i + 1 < len(mids) else 0.0)
                return cx

            if masks_in:
                y_in = sec_cy - lane_half if masks_ex else sec_cy
                end_in = _place_lane(masks_in, y_in)
            else:
                end_in = cur_x_base
            end_ex = _place_lane(masks_ex, sec_cy + lane_half) if masks_ex else cur_x_base
            cur_x = max(end_in, end_ex)

            if merge_id is not None:
                set_size(merge_id)
                mw_m, mh_m = sizes[merge_id]
                cur_x += fan_mask_row_gap
                positions[merge_id] = (cur_x, sec_cy - mh_m / 2.0)
                cur_x += mw_m

            if ckpt is not None:
                set_size(ckpt_id)
                cw, ch = sizes[ckpt_id]
                if masks_in or masks_ex or merge_id is not None:
                    cur_x += fan_ckpt_gap
                else:
                    cur_x = branch_right + fan_ckpt_gap
                positions[ckpt_id] = (cur_x, sec_cy - ch / 2.0)

            row_bottom = sec_y + sh
            if cids:
                for cid in cids:
                    cy = positions[cid][1]
                    ch = sizes[cid][1]
                    row_bottom = max(row_bottom, cy + ch)
            for mid in masks_in + masks_ex:
                my = positions[mid][1]
                mh = sizes[mid][1]
                row_bottom = max(row_bottom, my + mh)
            if merge_id is not None:
                cy = positions[merge_id][1]
                mh = sizes[merge_id][1]
                row_bottom = max(row_bottom, cy + mh)
            if ckpt is not None:
                cy = positions[ckpt_id][1]
                ch = sizes[ckpt_id][1]
                row_bottom = max(row_bottom, cy + ch)

            y_sec = row_bottom + fan_section_gap

    for pid, cids in branch_by_parent.items():
        if use_fan and pid.startswith("n_sec_"):
            continue
        if pid not in positions:
            continue
        px, py = positions[pid]
        pw, ph = sizes[pid]
        pcy = py + ph / 2.0
        for cid in cids:
            set_size(cid)
        gap_h = branch_gap_from_parent
        gap_v = branch_gap_sibling
        total_h = sum(sizes[cid][1] for cid in cids) + gap_v * (len(cids) - 1)
        stack_top = pcy - total_h / 2.0
        cur_y = stack_top
        for cid in cids:
            _, h = sizes[cid]
            positions[cid] = (px + pw + gap_h, cur_y)
            cur_y += h + gap_v

    edges_for_align = graph_edges_from_nodes(nodes)

    _resolve_node_overlaps(
        positions,
        sizes,
        margin=overlap_margin,
        passes=overlap_passes,
    )

    _layout_linear_chains_horizontal(
        positions,
        sizes,
        edges_for_align,
        kind="main_h",
        gap=linear_chain_gap,
    )
    _layout_linear_chains_horizontal(
        positions,
        sizes,
        edges_for_align,
        kind="mask_h",
        gap=linear_chain_gap,
    )
    _layout_branch_side_linear(
        positions,
        sizes,
        edges_for_align,
        gap=branch_gap_from_parent,
    )
    _place_base_surfaces_from_children(
        nodes,
        positions,
        sizes,
        edges_for_align,
        gap=linear_chain_gap,
        x0_min=x0,
        set_size=set_size,
    )
    _layout_merge_fanin(
        nodes,
        positions,
        sizes,
        edges_for_align,
        gap=merge_gap_from_parents,
        stack_gap=merge_stack_gap,
        set_size=set_size,
    )
    # Après repositionnement des jonctions (fan-in), certaines chaînes simples
    # (ex. n_dbg_merge_k -> n_ckpt_k) ne sont plus parfaitement horizontales.
    # On relance l'alignement linéaire pour recoller visuellement ces paires.
    _layout_linear_chains_horizontal(
        positions,
        sizes,
        edges_for_align,
        kind="main_h",
        gap=linear_chain_gap,
    )
    _layout_linear_chains_horizontal(
        positions,
        sizes,
        edges_for_align,
        kind="mask_h",
        gap=linear_chain_gap,
    )
    _layout_bifurcation_on_children_mean(
        positions,
        sizes,
        edges_for_align,
    )
    _layout_macro_blocks_from_active_lanes(
        positions,
        sizes,
        gap_left_to_sections=linear_chain_gap,
    )
    # Dernier passage d'alignement : certaines étapes macro/bifurcation peuvent
    # recasser localement une chaîne simple du tronc (ex. TB_q_snap -> n_opts).
    _layout_linear_chains_horizontal(
        positions,
        sizes,
        edges_for_align,
        kind="main_h",
        gap=linear_chain_gap,
    )
    _layout_linear_chains_horizontal(
        positions,
        sizes,
        edges_for_align,
        kind="mask_h",
        gap=linear_chain_gap,
    )
    _enforce_fan_section_dx(
        positions,
        edges_for_align,
        hub_id="n_opts",
        fan_section_dx=fan_section_dx,
    )
    _layout_terminal_surfaces_junction(
        positions,
        sizes,
        chain_gap=linear_chain_gap,
        junction_gap=merge_gap_from_parents,
    )

    return positions, sizes
