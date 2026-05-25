"""
Paramètres et constantes configurables pour Texture Halo.
"""

# Seuils par défaut (pour compatibilité)
DEFAULT_GLOW_THRESHOLD = 220
DEFAULT_GLOW_RADIUS = 15
DEFAULT_GLOW_INTENSITY = 1.2

# Paramètres glow — chaîne blanchiment (image inversée + glow)
DEFAULT_GLOW_EX_RADIUS = 18
DEFAULT_GLOW_EX_THRESHOLD = 200
DEFAULT_GLOW_EX_INTENSITY = 1.4

# Paramètres glow — chaîne ombre (sur l’image de base)
DEFAULT_GLOW_IN_RADIUS = 25
DEFAULT_GLOW_IN_THRESHOLD = 250
DEFAULT_GLOW_IN_INTENSITY = 2.0

# Spread Grain (filtre "Éparpiller")
DEFAULT_SPREAD_GRAIN_X = 2.0
DEFAULT_SPREAD_GRAIN_Y = 2.0

# Seuils de masques
DEFAULT_MASQUE_BLANC_SEUIL = 250
DEFAULT_MASQUE_NOIR_SEUIL = 5

# Fusion de grains
# Mode "Fusion de grain" de GIMP par défaut (formule: E = I + M - 128)
# Référence: https://docs.gimp.org/2.6/fr/gimp-concepts-layer-modes.html
DEFAULT_FUSION_MODE = 'grain_merge'
DEFAULT_FUSION_FORCE = 1.0
DEFAULT_FUSION_FORCE_INM = 0.5  # Opacité fusion ombres sur la base (voir fusion_force_inm)
DEFAULT_FUSION_FORCE_INEXM = 0.2  # Opacité fusion blanchiments (voir fusion_force_inexm)

# Libellés interface (les clés dict résultat moteur restent « INM » / « INEXM »)
UI_PROGRESS_FUSION_OMBRES = "Ombres fusionnées sur la base (diagnostic)"
UI_PROGRESS_FUSION_BLANCHIMENTS = "Ombres et blanchiments fusionnés (diagnostic)"
UI_GRAPHE_TITRE_APRES_OMBRES = "Surfaces — après ombres (sur base métal)"
UI_GRAPHE_TITRE_APRES_BLANCHIMENTS = "Surfaces — après blanchiments"

# Libellés « TB / TB_q » (clés payload : TB, TB_q, TB_q_presnap, TB_q_snapped)
UI_ANCRAGE_PIPELINE_COURT = (
    "Référence : **image de base** (étape du nœud) + **options du pipeline** pour ce rendu. "
)
UI_ANCRAGE_PIPELINE_LONG = (
    "Toute la logique affichée se lit à partir de l’**image de base** du dessin (préparation puis, selon le nœud, "
    "quantification / alignement sur les paliers Forerunner) et des **options** (étages, fusion, texture métal, "
    "périmètre, graines). "
)
UI_GRAPHE_TITRE_IMAGE_BASE_PREP = "Image de base — préparation (entrée pipeline)"
UI_GRAPHE_TITRE_IMAGE_BASE_AVANT_PALIERS = "Image de base — avant alignement sur les paliers"
UI_GRAPHE_TITRE_IMAGE_BASE_PALIERS = "Image de base — paliers Forerunner appliqués"

# Messages progression — calques ombre / blanchiment (clés moteur EXOK, INOK, *TR inchangées)
UI_PROGRESS_CALQUE_BLANCHIMENT_MASQUE = "Calque blanchiment prêt (avant transparence)"
UI_PROGRESS_CALQUE_OMBRE_MASQUE = "Calque ombre prêt (avant transparence)"
UI_PROGRESS_CALQUE_OMBRE_FUSION_METAL = "Ombre : blanc → transparence pour fusion sur métal"
UI_PROGRESS_CALQUE_BLANCHIMENT_FUSION_METAL = "Blanchiment : noir → transparence pour fusion sur métal"

# Quantification
DEFAULT_NIVEAUX = 4

# Renforcement contours
DEFAULT_CONTOUR_METHODE = 'pillow'
# Score composite detecter_flou (global + contours) ; en dessous = flou / transitions trop douces
DEFAULT_SEUIL_FLOU = 18.0

# Dessin flou : masque flou (UnsharpMask PIL) avant contours + snap — vise aplats + arêtes adoucies
DEFAULT_AUTO_SHARPEN_BLUR = True
DEFAULT_DEBLUR_UNSHARP_RADIUS = 2.0
DEFAULT_DEBLUR_UNSHARP_PERCENT = 200
DEFAULT_DEBLUR_UNSHARP_THRESHOLD = 2
DEFAULT_DEBLUR_UNSHARP_PASSES = 1

# Après défloutage ou si « transitions douces » détectées : cible les franges 1–3 px (tracé avec AA)
DEFAULT_AUTO_ANTIALIAS_PASS = True
DEFAULT_AA_UNSHARP_RADIUS = 1.0
DEFAULT_AA_UNSHARP_PERCENT = 220
DEFAULT_AA_UNSHARP_THRESHOLD = 0
DEFAULT_AA_UNSHARP_PASSES = 1

# Snap : par défaut, nombre de paliers déduit du contenu (histogramme), pas les 6 forcés
DEFAULT_SNAP_USE_FULL_PALETTE = False

# Si False (défaut) : ne pas snapper sur les paliers Forerunner lorsque l’image de base n’a reçu aucune
# correction de netteté (anti-flou / passe AA) — texture déjà nette, inutile de quantifier.
# Mettre True pour toujours snapper (ex. calque déjà net mais pas encore sur les paliers).
# Le mode snap_only (mise au propre / éditeur) snappe toujours ; cette option ne s’applique pas.
DEFAULT_FORCE_FORERUNNER_SNAP = False

# Mise au propre (snap_only) : sans unsharp sur les contours (limite une frange ~1 px aux transitions)
DEFAULT_SNAP_ONLY_SKIP_CONTOUR = True

