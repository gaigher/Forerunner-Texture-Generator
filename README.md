Forerunner Texture Generator
============================

Forerunner Texture Generator est une application de bureau :
elle utilise deux images d'entrée — une image de lignes de construction, qui sert de support aux calculs des effets (ombre, blanchiment, grain), et une image de métal, sur laquelle les effets sont appliqués pour produire le rendu final.
Les lignes de construction peuvent être tracées dans l'éditeur intégré ou importées depuis un fichier.
Le résultat est une texture visuelle de style Forerunner, proche de celles de la série Halo.

L'architecture forerunner inspire une grande partie des assets de la licence. Après de nombreuses années à modder Halo, l'auteur a identifié les techniques couramment utilisées pour obtenir ce style ; plutôt qu'un long tutoriel, ce dépôt propose un logiciel qui encadre et automatise ce pipeline.

![Aperçu de l'application](images/interface.png)

<h2>🎥 Démonstration</h2>

<a href="https://www.youtube.com/watch?v=QzwhTnUbOB0" target="_blank">
  <img src="https://img.youtube.com/vi/QzwhTnUbOB0/maxresdefault.jpg"
       alt="Démonstration pv-scanner"
       width="600">
</a>

*Cliquez sur l’image pour visionner la vidéo de démonstration sur YouTube.*


Fenêtre principale (Texture Halo — Générateur de Textures Forerunner) :
panneau des paramètres par étage (surfaces, blanchiment / glow, grain, creux / ombres) et prévisualisation avec zoom.

images/interface-texture-halo.jpg


Éditeur de lignes de construction :
dessin des formes de base (outils crayon, ligne, rectangles, remplissage, etc.), mode carrelable pour textures sans couture, choix d'étage / niveau de gris.

images/interface-editeur-construction.jpg


--------------------------------------------------------------------------------
UTILISATION (Windows)
--------------------------------------------------------------------------------

1. Téléchargez l'archive ZIP du programme (page Releases du projet, quand elle sera disponible).
2. Décompressez le dossier complet sur votre disque.
3. Double-cliquez sur ForerunnerTextureGenerator.exe (ne déplacez pas l'exe sans le dossier qui l'accompagne).
4. Dans l'application :
   - chargez ou dessinez vos lignes de construction ;
   - chargez votre image de métal ;
   - réglez les paramètres par étage (curseurs, cases à cocher) ;
   - utilisez la prévisualisation pour ajuster le rendu ;
   - exportez l'image finale lorsque le résultat vous convient.



--------------------------------------------------------------------------------
RESSOURCES
--------------------------------------------------------------------------------

Des ressources sont à disposition dans les dossiers :

  metal_textures_base\
  lignes_construction_base\


--------------------------------------------------------------------------------
À PROPOS
--------------------------------------------------------------------------------

Développé par : www.gaigher.fr
GitHub : github.com/gaigher

Coordonnées (e-mail, etc.) : menu "À propos" dans l'application.


--------------------------------------------------------------------------------
LICENCE
--------------------------------------------------------------------------------

MIT — voir le fichier LICENSE.


================================================================================
ANNEXE — pour qui souhaite lancer le projet avec Python (développeurs)
================================================================================

Prérequis : Python 3.10 ou plus récent (3.11 recommandé sous Windows).

À la racine du projet, dans PowerShell :

  python -m venv .venv
  .\.venv\Scripts\Activate.ps1
  pip install -r requirements.txt
  python main.py

Créer l'exécutable Windows (PyInstaller) :

  powershell -ExecutionPolicy Bypass -File .\build_exe.ps1

Sortie : dossier dist\ contenant ForerunnerTextureGenerator.exe et les fichiers associés.
