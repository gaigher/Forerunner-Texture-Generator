"""
Presets de configuration pour Texture Halo.
"""
from typing import Dict, Optional
from ui.logger import ProcessingError
from ui.settings import (
    DEFAULT_GLOW_RADIUS, DEFAULT_GLOW_THRESHOLD, DEFAULT_GLOW_INTENSITY,
    DEFAULT_SPREAD_GRAIN_X, DEFAULT_SPREAD_GRAIN_Y,
    DEFAULT_NIVEAUX
)


# Presets prédéfinis
PRESETS: Dict[str, Dict] = {
    'default': {
        'niveaux': DEFAULT_NIVEAUX,
        'glow': {
            'enabled': False,
            'radius': DEFAULT_GLOW_RADIUS,
            'threshold': DEFAULT_GLOW_THRESHOLD,
            'intensity': DEFAULT_GLOW_INTENSITY
        },
        'spread_grain': {
            'enabled': False,
            'spread_x': DEFAULT_SPREAD_GRAIN_X,
            'spread_y': DEFAULT_SPREAD_GRAIN_Y,
            'seed': None
        },
        'do_contour': True,
        'glow_on_tb': False,
        'preserve_input_format': True
    },

    'glow_soft': {
        'niveaux': 4,
        'glow': {
            'enabled': True,
            'radius': 10,
            'threshold': 200,
            'intensity': 1.0
        },
        'spread_grain': {
            'enabled': False,
            'spread_x': DEFAULT_SPREAD_GRAIN_X,
            'spread_y': DEFAULT_SPREAD_GRAIN_Y,
            'seed': None
        },
        'do_contour': True,
        'glow_on_tb': False,
        'preserve_input_format': True
    },

    'glow_intense': {
        'niveaux': 4,
        'glow': {
            'enabled': True,
            'radius': 20,
            'threshold': 220,
            'intensity': 1.5
        },
        'spread_grain': {
            'enabled': False,
            'spread_x': DEFAULT_SPREAD_GRAIN_X,
            'spread_y': DEFAULT_SPREAD_GRAIN_Y,
            'seed': None
        },
        'do_contour': True,
        'glow_on_tb': False,
        'preserve_input_format': True
    },

    'spread_grain_subtle': {
        'niveaux': 4,
        'glow': {
            'enabled': False,
            'radius': DEFAULT_GLOW_RADIUS,
            'threshold': DEFAULT_GLOW_THRESHOLD,
            'intensity': DEFAULT_GLOW_INTENSITY
        },
        'spread_grain': {
            'enabled': True,
            'spread_x': 1.5,
            'spread_y': 1.5,
            'seed': None
        },
        'do_contour': True,
        'glow_on_tb': False,
        'preserve_input_format': True
    },

    'spread_grain_strong': {
        'niveaux': 4,
        'glow': {
            'enabled': False,
            'radius': DEFAULT_GLOW_RADIUS,
            'threshold': DEFAULT_GLOW_THRESHOLD,
            'intensity': DEFAULT_GLOW_INTENSITY
        },
        'spread_grain': {
            'enabled': True,
            'spread_x': 4.0,
            'spread_y': 4.0,
            'seed': None
        },
        'do_contour': True,
        'glow_on_tb': False,
        'preserve_input_format': True
    },

    'full_effects': {
        'niveaux': 6,
        'glow': {
            'enabled': True,
            'radius': 15,
            'threshold': 220,
            'intensity': 1.2
        },
        'spread_grain': {
            'enabled': True,
            'spread_x': 2.0,
            'spread_y': 2.0,
            'seed': None
        },
        'do_contour': True,
        'glow_on_tb': False,
        'preserve_input_format': True
    },

    'quant_only': {
        'niveaux': 8,
        'glow': {
            'enabled': False,
            'radius': DEFAULT_GLOW_RADIUS,
            'threshold': DEFAULT_GLOW_THRESHOLD,
            'intensity': DEFAULT_GLOW_INTENSITY
        },
        'spread_grain': {
            'enabled': False,
            'spread_x': DEFAULT_SPREAD_GRAIN_X,
            'spread_y': DEFAULT_SPREAD_GRAIN_Y,
            'seed': None
        },
        'do_contour': True,
        'glow_on_tb': False,
        'preserve_input_format': True
    }
}


def load_preset(name: str) -> Dict:
    """
    Charge un preset par son nom.

    Paramètres :
        name: Nom du preset

    Renvoie :
        Dictionnaire d'options du preset

    Lève :
        ProcessingError: Si le preset n'existe pas
    """
    if name not in PRESETS:
        available = ', '.join(PRESETS.keys())
        raise ProcessingError(
            f"Preset '{name}' introuvable. Presets disponibles: {available}"
        )

    # Retourner une copie pour éviter les modifications
    import copy
    return copy.deepcopy(PRESETS[name])


def list_presets() -> list:
    """
    Liste tous les presets disponibles.

    Renvoie :
        Liste des noms de presets
    """
    return list(PRESETS.keys())
