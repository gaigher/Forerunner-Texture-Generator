"""
Outils d'erreurs partagés pour Texture Halo.
"""


def ui_log(msg: str, level: str = 'info') -> None:
    """
    Fonction conservée pour compatibilité, sans sortie de log.

    Paramètres :
        msg: Message ignoré
        level: Niveau ignoré
    """
    return None


# Exceptions personnalisées
class TextureHaloError(Exception):
    """Exception de base pour Texture Halo."""
    pass


class ConversionNotAllowed(TextureHaloError):
    """Levée quand une conversion n'est pas autorisée."""
    pass


class ProcessingError(TextureHaloError):
    """Levée lors d'une erreur de traitement."""
    pass

