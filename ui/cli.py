"""
Interface en ligne de commande pour Texture Halo.
"""
import sys
import click
from pathlib import Path
from effets.traitement import process_image
from ui.logger import ui_log, ProcessingError
from ui.presets import load_preset, list_presets
from ui.settings import (
    DEFAULT_FUSION_FORCE_INM,
    DEFAULT_FUSION_FORCE_INEXM,
    DEFAULT_SEUIL_FLOU,
)


@click.command()
@click.argument('input_path', type=click.Path(exists=True))
@click.argument('output_path', type=click.Path())
@click.option('--niveaux', type=int, default=0, help='Quantification en N niveaux (0=none)')
@click.option('--glow/--no-glow', default=False, help='Appliquer effet glow')
@click.option('--glow-radius', type=int, default=15, help='Rayon du glow')
@click.option('--glow-threshold', type=int, default=220, help='Seuil de luminosité pour glow (0-255)')
@click.option('--glow-intensity', type=float, default=1.2, help='Intensité du glow')
@click.option('--spread-grain/--no-spread-grain', default=False, help='Appliquer effet Spread Grain (filtre "Éparpiller")')
@click.option('--spread-grain-x', type=float, default=2.0, help='Distance maximale de déplacement horizontal (pixels)')
@click.option('--spread-grain-y', type=float, default=2.0, help='Distance maximale de déplacement vertical (pixels)')
@click.option('--spread-grain-seed', type=int, default=None, help='Graine aléatoire pour Spread Grain')
@click.option('--tm', type=click.Path(exists=True), help='Chemin vers texture métallique')
@click.option('--fusion-mode', type=click.Choice(['grain_merge', 'multiply', 'overlay', 'soft_light']),
              default='grain_merge', help='Mode de fusion des grains (grain_merge = mode "Fusion de grain" de GIMP)')
@click.option('--fusion-force-inm', type=float, default=DEFAULT_FUSION_FORCE_INM,
              help=f'Opacité de fusion des ombres sur la base métal (défaut: {DEFAULT_FUSION_FORCE_INM})')
@click.option('--fusion-force-inexm', type=float, default=DEFAULT_FUSION_FORCE_INEXM,
              help=f'Opacité de fusion des blanchiments sur l’image déjà ombrée (défaut: {DEFAULT_FUSION_FORCE_INEXM})')
@click.option('--palette', type=click.Path(exists=True), help='Chemin vers palette personnalisée')
@click.option('--allow-conversion', is_flag=True, help='Autoriser conversion de mode automatique')
@click.option('--do-quant/--no-quant', default=True, help='Activer/désactiver quantification')
@click.option('--glow-on-tb', is_flag=True, help='Appliquer le glow sur l’image de base (au lieu de la couche inversée seule)')
@click.option(
    '--seuil-flou',
    type=float,
    default=DEFAULT_SEUIL_FLOU,
    help='Seuil score netteté (alerte si inférieur ; voir effets.utilitaires.detecter_flou)',
)
@click.option('--contour-methode', type=click.Choice(['pillow', 'opencv']),
              default='pillow', help='Méthode de renforcement des contours')
@click.option('--preset', type=str, help='Nom du preset à utiliser')
@click.option('--batch-dir', type=click.Path(exists=True, file_okay=False),
              help='Traiter tous les fichiers d\'un répertoire')
@click.option('--preview-mode', is_flag=True, help='Mode aperçu (affiche les étapes)')
@click.option('--masque-blanc-seuil', type=int, default=250, help='Seuil pour masque blanc vers alpha')
@click.option('--masque-noir-seuil', type=int, default=5, help='Seuil pour masque noir vers alpha')
@click.option(
    '--use-inok',
    is_flag=True,
    help='Exporter l’image finale à partir du calque ombre plutôt que du calque blanchiment',
)
@click.option('--preserve-format/--no-preserve-format', default=True,
              help='Préserver le format d\'entrée')
def main(input_path, output_path, niveaux, glow, glow_radius, glow_threshold, glow_intensity,
         spread_grain, spread_grain_x, spread_grain_y, spread_grain_seed,
         tm, fusion_mode, fusion_force_inm, fusion_force_inexm, palette, allow_conversion, do_quant,
         glow_on_tb, seuil_flou, contour_methode, preset, batch_dir, preview_mode,
         masque_blanc_seuil, masque_noir_seuil, use_inok, preserve_format):
    """Texture Halo - Générateur de textures Forerunner."""

    # Charger preset si spécifié
    opts = {}
    if preset:
        try:
            opts = load_preset(preset)
            ui_log(f"Preset '{preset}' chargé", 'info')
        except Exception as e:
            click.echo(f"Erreur lors du chargement du preset: {e}", err=True)
            sys.exit(1)

    # Construire les options depuis les arguments CLI
    opts.update({
        'niveaux': niveaux if do_quant else 0,
        'glow': {
            'enabled': glow,
            'radius': glow_radius,
            'threshold': glow_threshold,
            'intensity': glow_intensity
        },
        'spread_grain': {
            'enabled': spread_grain,
            'spread_x': spread_grain_x,
            'spread_y': spread_grain_y,
            'seed': spread_grain_seed
        },
        'tm': str(tm) if tm else None,
        'tm_fit_mode': 'anisotropic',
        'fusion_mode': fusion_mode,
        'fusion_force_inm': fusion_force_inm,
        'fusion_force_inexm': fusion_force_inexm,
        'palette': str(palette) if palette else None,
        'allow_conversion': allow_conversion,
        'do_contour': False,
        'glow_on_tb': glow_on_tb,
        'seuil_flou': seuil_flou,
        'contour_methode': contour_methode,
        'masque_blanc_seuil': masque_blanc_seuil,
        'masque_noir_seuil': masque_noir_seuil,
        'use_inok': use_inok,
        'preserve_input_format': preserve_format,
        'output_path': output_path
    })

    # Rappel de progression
    def progress_cb(percent: int, message: str):
        click.echo(f"[{percent}%] {message}", err=True)

    # Traitement par lot ou fichier unique
    if batch_dir:
        batch_path = Path(batch_dir)
        image_extensions = {'.png', '.jpg', '.jpeg', '.tiff', '.tif', '.bmp'}
        image_files = [f for f in batch_path.iterdir()
                      if f.suffix.lower() in image_extensions]

        if not image_files:
            click.echo(f"Aucune image trouvée dans {batch_dir}", err=True)
            sys.exit(1)

        click.echo(f"Traitement de {len(image_files)} fichiers...")
        for img_file in image_files:
            output_file = img_file.parent / f"{img_file.stem}_processed{img_file.suffix}"
            opts['output_path'] = str(output_file)
            try:
                result_path = process_image(str(img_file), opts, progress_cb)
                click.echo(f"✓ {img_file.name} -> {result_path}")
            except Exception as e:
                click.echo(f"✗ Erreur sur {img_file.name}: {e}", err=True)
    else:
        # Fichier unique
        try:
            if preview_mode:
                # Mode aperçu: utiliser process_image_stepwise
                from effets.traitement import process_image_stepwise
                from effets.utilitaires import charger_image
                img = charger_image(input_path)
                results = process_image_stepwise(img, opts, progress_cb)
                click.echo("\nÉtapes disponibles:", err=True)
                for key in results.keys():
                    click.echo(f"  - {key}", err=True)
                # Sauvegarder quand même le résultat final
                result_path = process_image(input_path, opts, progress_cb)
            else:
                result_path = process_image(input_path, opts, progress_cb)

            click.echo(f"Image traitée: {result_path}")
        except ProcessingError as e:
            click.echo(f'Erreur de traitement: {e}', err=True)
            sys.exit(1)
        except Exception as e:
            click.echo(f'Erreur: {e}', err=True)
            sys.exit(1)

if __name__ == '__main__':
    main()
