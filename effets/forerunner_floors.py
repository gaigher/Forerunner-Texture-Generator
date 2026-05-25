"""
Valeurs de gris des étages Forerunner (alignées sur ui/construction_editor).
"""
from typing import Tuple

# Indices 0 = plus bas / fond (noir), 5 = plus haut (blanc).
# Ombre de l'étage k : fusion masquée sur le gris de l'étage k-1 (le fond reçoit l'ombre de k=1).
# Aucun étage ne cible le gris 255 : le palier blanc ne reçoit pas d'ombre dans ce schéma.
FORERUNNER_FLOOR_GRAYS: Tuple[int, ...] = (0, 51, 102, 153, 204, 255)

FORERUNNER_FLOOR_TITLES: Tuple[str, ...] = (
    "Étage 0 — Noir (fond)",
    "Étage 1 — Gris très foncé",
    "Étage 2 — Gris moyen-foncé",
    "Étage 3 — Gris moyen",
    "Étage 4 — Gris clair",
    "Étage 5 — Blanc",
)
