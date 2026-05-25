"""
Interface graphique pour Texture Halo avec prévisualisation en temps réel.
"""
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import threading
import pickle
import subprocess
import sys
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import webbrowser
import secrets
import numpy as np
import tempfile

from ui.metal_procedural import make_seamless_metal_texture
from effets.traitement import (
    process_image_stepwise,
    build_forerunner_snap_cache_state,
    process_forerunner_surfaces_only,
    forerunner_presnap_cache_key,
    surface_pipeline_global_key,
    floor_surface_fingerprint,
    first_changed_floor_index,
    floor_removal_requires_full_surface_rebuild,
    copy_floor_surface_state,
    image_to_preview_rgb,
)
from effets.utilitaires import charger_image, detect_used_forerunner_floor_indices, detecter_flou
from ui.settings import (
    DEFAULT_GLOW_THRESHOLD, DEFAULT_GLOW_RADIUS, DEFAULT_GLOW_INTENSITY,
    DEFAULT_GLOW_EX_RADIUS, DEFAULT_GLOW_EX_THRESHOLD, DEFAULT_GLOW_EX_INTENSITY,
    DEFAULT_GLOW_IN_RADIUS, DEFAULT_GLOW_IN_THRESHOLD, DEFAULT_GLOW_IN_INTENSITY,
    DEFAULT_SPREAD_GRAIN_X, DEFAULT_SPREAD_GRAIN_Y,
    DEFAULT_FUSION_MODE, DEFAULT_FUSION_FORCE_INM, DEFAULT_FUSION_FORCE_INEXM,
    DEFAULT_NIVEAUX, DEFAULT_CONTOUR_METHODE, DEFAULT_SEUIL_FLOU,
    DEFAULT_SNAP_ONLY_SKIP_CONTOUR,
    DEFAULT_AUTO_SHARPEN_BLUR,
    DEFAULT_AUTO_ANTIALIAS_PASS,
)
from effets.forerunner_floors import FORERUNNER_FLOOR_TITLES

# Délai après le dernier changement de paramètre avant de lancer le pipeline (ms).
# Utilisé aussi pendant le glisser-déposer des curseurs : un seul rendu après la pause.
PREVIEW_DEBOUNCE_MS = 200

# En « Mod Rapide », le plus grand côté de l’image d’aperçu ne dépasse pas cette valeur (px).
FAST_PREVIEW_MAX_EDGE = 1024

# Libellé affiché quand la TM est une génération procédurale (fichier temporaire réel dans current_tm_path).
METAL_PROCEDURAL_DISPLAY = "Texture métallique procédurale (sans couture, liée à l’image de base)"


class TextureHaloGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Texture Halo - Générateur de Textures Forerunner")
        self.root.geometry("1400x900")
        
        # Variables d'état
        self.current_image: Optional[Image.Image] = None
        self.current_tm_path: Optional[str] = None
        # TM procédurale : recalcul automatique quand la taille de l’image de base change.
        self._metal_procedural_spec: Optional[Dict[str, int]] = None  # {"seed": int}
        self._tm_procedural_temp_path: Optional[str] = None
        self.preview_image: Optional[Image.Image] = None
        self.update_timer: Optional[str] = None
        self.is_processing = False
        self._preview_pending = False  # Relancer le traitement si des params ont changé pendant un run
        self._last_progress_msg = ""  # Dernier message pipeline (suffixe « en attente » si params changent)
        self._status_anim_job: Optional[str] = None
        self._status_busy = False
        self._status_has_progress = False
        self._status_phase = 0
        self._status_progress_pct = 0
        
        # Cache incrémental : image de base + étapes paliers (avant/après snap Forerunner) si seuls les étages / TM / masques changent
        self._source_image_generation = 0
        self._snap_cache_key: Optional[tuple] = None
        self._snap_cache_tb = None
        self._snap_cache_tb_q_presnap = None
        self._snap_cache_tb_q_snapped = None
        self._snap_cache_infer = None
        
        # Cache incrémental surfaces par étage (empreintes + checkpoints après chaque k)
        self._surface_global_key: Optional[tuple] = None
        self._floor_fingerprints: Optional[tuple] = None
        self._floor_checkpoints_after: Optional[dict] = None
        self._floor_layer_cache: dict = {}
        # Étapes image de base / surfaces dans la prévisualisation : uniquement après import ou validation éditeur
        self._pipeline_visual_on_first_base_load: bool = False
        # Données pipeline (graphe Qt) : dernier traitement avec capture
        self._last_pipeline_state: Optional[dict] = None
        self._last_pipeline_opts: Optional[dict] = None
        self.pipeline_capture_details = tk.BooleanVar(value=False)
        # Ajustement TM → taille de l'image de base (pipeline + éditeur de construction)
        self.tm_fit_mode_var = tk.StringVar(value="anisotropic")
        
        # Variables pour zoom et pan
        self.zoom_level = 1.0
        self.pan_x = 0
        self.pan_y = 0
        self.pan_start_x = 0
        self.pan_start_y = 0
        self.is_panning = False
        self.preview_resampling = Image.Resampling.NEAREST  # Toujours NEAREST pour la prévisualisation
        self.fast_preview_var = tk.BooleanVar(value=False)
        
        # Créer l'interface
        self._create_ui()
        
        # Charger les valeurs par défaut
        self._load_default_values()
    
    def _create_ui(self):
        """Crée l'interface utilisateur."""
        # Barre de menus
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)
        
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Fichier", menu=file_menu)
        file_menu.add_command(label="Charger une image...", command=self._load_image)
        file_menu.add_command(label="Charger texture métallique...", command=self._load_metal_texture)
        file_menu.add_separator()
        file_menu.add_command(label="Exporter le résultat...", command=self._export_result)
        file_menu.add_separator()
        file_menu.add_command(label="Quitter", command=self.root.quit)
        
        # Menu Pipeline (masques et étapes)
        diag_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Pipeline", menu=diag_menu)
        diag_menu.add_checkbutton(
            label="Capturer masques et étapes (rendu plus lent)",
            variable=self.pipeline_capture_details,
        )
        diag_menu.add_separator()
        diag_menu.add_command(
            label="Ouvrir dans NodeGraphQt (Qt)…",
            command=self._open_pipeline_graph_qt_from_menu,
        )
        
        # Menu Aide
        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Aide", menu=help_menu)
        help_menu.add_command(label="À propos...", command=self._show_about)
        
        # Panneau principal : languette native (pas de calque ⋮ = pas de conflit avec les scrollbars)
        _paned_bg = ttk.Style().lookup("TFrame", "background", default="") or "#e8e8e8"
        main_panel = tk.PanedWindow(
            self.root,
            orient=tk.HORIZONTAL,
            sashwidth=10,
            sashpad=2,
            sashrelief=tk.FLAT,
            sashcursor="sb_h_double_arrow",
            background=_paned_bg,
            bd=0,
        )
        main_panel.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Panneau gauche : Paramètres (scrollable)
        left_container = ttk.Frame(main_panel)
        main_panel.add(left_container, minsize=260, stretch="always")
        
        # Frame pour les paramètres avec bordure
        left_frame = ttk.LabelFrame(left_container, text="Paramètres", padding=5)
        left_frame.pack(fill=tk.BOTH, expand=True)
        
        # Canevas avec barre de défilement pour les paramètres
        canvas_container = ttk.Frame(left_frame)
        canvas_container.pack(fill=tk.BOTH, expand=True)
        
        canvas = tk.Canvas(canvas_container, highlightthickness=0)
        scrollbar = ttk.Scrollbar(canvas_container, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        def update_scrollregion(event=None):
            canvas.update_idletasks()
            bbox = canvas.bbox("all")
            if bbox:
                canvas.configure(scrollregion=bbox)
        
        scrollable_frame.bind("<Configure>", update_scrollregion)
        
        # Mettre à jour la scrollregion quand le contenu change
        def bind_scroll_update(widget):
            widget.bind("<Configure>", update_scrollregion)
            for child in widget.winfo_children():
                bind_scroll_update(child)
        
        bind_scroll_update(scrollable_frame)
        
        canvas_window = canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        
        def configure_canvas_width(event):
            canvas_width = event.width
            canvas.itemconfig(canvas_window, width=canvas_width)
        
        canvas.bind('<Configure>', configure_canvas_width)
        canvas.configure(yscrollcommand=scrollbar.set)

        def on_params_wheel(event):
            if getattr(event, "delta", 0):
                canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
            return "break"

        def on_params_wheel_up(_event):
            canvas.yview_scroll(-3, "units")
            return "break"

        def on_params_wheel_down(_event):
            canvas.yview_scroll(3, "units")
            return "break"

        def bind_params_wheel_recursive(widget):
            widget.bind("<MouseWheel>", on_params_wheel)
            widget.bind("<Button-4>", on_params_wheel_up)
            widget.bind("<Button-5>", on_params_wheel_down)
            for ch in widget.winfo_children():
                bind_params_wheel_recursive(ch)
        
        canvas.grid(row=0, column=0, sticky="nsew")
        scrollbar.grid(row=0, column=1, sticky="ns")
        canvas_container.grid_rowconfigure(0, weight=1)
        canvas_container.grid_columnconfigure(0, weight=1)
        
        # Panneau droit : Prévisualisation
        right_container = ttk.Frame(main_panel)
        main_panel.add(right_container, minsize=320, stretch="always")
        
        # Frame pour la prévisualisation avec bordure
        right_frame = ttk.LabelFrame(right_container, text="Prévisualisation", padding=5)
        right_frame.pack(fill=tk.BOTH, expand=True)
        
        # Label de prévisualisation avec contrôles de zoom
        preview_header = ttk.Frame(right_frame)
        preview_header.pack(pady=5)
        
        preview_label = ttk.Label(preview_header, text="Prévisualisation", font=("Arial", 14, "bold"))
        preview_label.pack(side=tk.LEFT, padx=5)
        self.preview_step_var = tk.StringVar(value="")
        self.preview_step_label = ttk.Label(
            preview_header,
            textvariable=self.preview_step_var,
            font=("Arial", 10),
            foreground="#444",
        )
        self.preview_step_label.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=8)
        
        # Contrôles de zoom
        zoom_frame = ttk.Frame(preview_header)
        zoom_frame.pack(side=tk.RIGHT, padx=5)
        
        ttk.Label(zoom_frame, text="Zoom:").pack(side=tk.LEFT, padx=2)
        zoom_out_btn = ttk.Button(zoom_frame, text="-", width=3, command=self._zoom_out)
        zoom_out_btn.pack(side=tk.LEFT, padx=2)
        self.zoom_label = ttk.Label(zoom_frame, text="100%", width=6)
        self.zoom_label.pack(side=tk.LEFT, padx=2)
        zoom_in_btn = ttk.Button(zoom_frame, text="+", width=3, command=self._zoom_in)
        zoom_in_btn.pack(side=tk.LEFT, padx=2)
        reset_zoom_btn = ttk.Button(zoom_frame, text="Reset", width=6, command=self._reset_zoom)
        reset_zoom_btn.pack(side=tk.LEFT, padx=2)
        
        ttk.Separator(zoom_frame, orient=tk.VERTICAL).pack(side=tk.LEFT, padx=6, fill=tk.Y)
        ttk.Checkbutton(
            zoom_frame,
            text="Mod Rapide",
            variable=self.fast_preview_var,
            command=self._on_fast_preview_toggle,
        ).pack(side=tk.LEFT, padx=2)
        
        # Canevas pour l'image avec barres de défilement
        canvas_frame = ttk.Frame(right_frame)
        canvas_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Scrollbars
        h_scrollbar = ttk.Scrollbar(canvas_frame, orient=tk.HORIZONTAL)
        v_scrollbar = ttk.Scrollbar(canvas_frame, orient=tk.VERTICAL)
        
        self.preview_canvas = tk.Canvas(canvas_frame, bg="gray90", width=800, height=600,
                                       xscrollcommand=h_scrollbar.set,
                                       yscrollcommand=v_scrollbar.set)
        
        h_scrollbar.config(command=self.preview_canvas.xview)
        v_scrollbar.config(command=self.preview_canvas.yview)
        
        self.preview_canvas.grid(row=0, column=0, sticky="nsew")
        h_scrollbar.grid(row=1, column=0, sticky="ew")
        v_scrollbar.grid(row=0, column=1, sticky="ns")
        
        canvas_frame.grid_rowconfigure(0, weight=1)
        canvas_frame.grid_columnconfigure(0, weight=1)
        
        # Bindings pour zoom et pan
        self.preview_canvas.bind("<ButtonPress-1>", self._on_canvas_press)
        self.preview_canvas.bind("<B1-Motion>", self._on_canvas_drag)
        self.preview_canvas.bind("<ButtonRelease-1>", self._on_canvas_release)
        self.preview_canvas.bind("<MouseWheel>", self._on_mousewheel)
        self.preview_canvas.bind("<Button-4>", self._on_mousewheel)  # Linux
        self.preview_canvas.bind("<Button-5>", self._on_mousewheel)  # Linux
        self.preview_canvas.focus_set()  # Pour recevoir les événements clavier
        
        # Barre de statut (fond vert = progression ; texte au premier plan)
        self.status_var = tk.StringVar(value="Prêt - Chargez une image pour commencer")
        status_frame = tk.Frame(self.root, bd=1, relief=tk.SUNKEN)
        status_frame.pack(side=tk.BOTTOM, fill=tk.X)
        self._status_canvas = tk.Canvas(
            status_frame,
            height=24,
            highlightthickness=0,
            bg="#e8e8e8",
        )
        self._status_canvas.pack(fill=tk.X)
        self._status_canvas.bind("<Configure>", lambda _e: self._draw_status_bar())
        self.status_var.trace_add("write", lambda *_a: self._draw_status_bar())
        
        # Créer les contrôles de paramètres
        self._create_parameter_controls(scrollable_frame)

        canvas.bind("<MouseWheel>", on_params_wheel)
        canvas.bind("<Button-4>", on_params_wheel_up)
        canvas.bind("<Button-5>", on_params_wheel_down)
        bind_params_wheel_recursive(scrollable_frame)
        
        # Mettre à jour la scrollbar après création des contrôles
        def update_after_creation():
            canvas.update_idletasks()
            canvas.configure(scrollregion=canvas.bbox("all"))
        self.root.after(100, update_after_creation)
    
    def _draw_status_bar(self) -> None:
        """Dessine la barre de statut pleine largeur (fond gris, remplissage vert si actif)."""
        c = getattr(self, "_status_canvas", None)
        if c is None or not c.winfo_exists():
            return
        w = max(int(c.winfo_width()), 2)
        h = max(int(c.winfo_height()), 2)
        c.delete("all")
        text = self.status_var.get()
        track = "#e0e0e0"
        green = "#27ae60"
        if self._status_busy:
            c.create_rectangle(0, 0, w, h, fill=track, outline="")
            if self._status_has_progress:
                fill_w = w * self._status_progress_pct / 100.0
                if fill_w > 0:
                    c.create_rectangle(0, 0, fill_w, h, fill=green, outline="")
            else:
                seg_w = max(60, min(w // 3, 200))
                x = (self._status_phase * (w + seg_w) / 120.0) % (w + seg_w) - seg_w
                c.create_rectangle(x, 0, x + seg_w, h, fill=green, outline="")
        else:
            c.create_rectangle(0, 0, w, h, fill="#e8e8e8", outline="")
        c.create_text(8, h // 2, text=text, anchor="w", fill="#101010")
    
    def _tick_status_anim(self) -> None:
        if not self._status_busy or self._status_has_progress:
            return
        self._status_phase = (self._status_phase + 2) % 120
        self._draw_status_bar()
        self._status_anim_job = self.root.after(40, self._tick_status_anim)
    
    def _start_status_busy(self) -> None:
        self._status_busy = True
        self._status_has_progress = False
        self._status_phase = 0
        self._status_progress_pct = 0
        if self._status_anim_job:
            self.root.after_cancel(self._status_anim_job)
            self._status_anim_job = None
        self._tick_status_anim()
    
    def _stop_status_busy(self) -> None:
        self._status_busy = False
        self._status_has_progress = False
        self._status_progress_pct = 0
        if self._status_anim_job:
            self.root.after_cancel(self._status_anim_job)
            self._status_anim_job = None
        self._draw_status_bar()
    
    def _on_worker_progress(self, pct: int, msg: str) -> None:
        """Appelé sur le fil d'interface (via after) pendant le traitement."""
        if not self._status_busy:
            return
        self._status_has_progress = True
        if self._status_anim_job:
            self.root.after_cancel(self._status_anim_job)
            self._status_anim_job = None
        self._status_progress_pct = max(0, min(100, pct))
        self._last_progress_msg = msg
        if self._preview_pending:
            msg = f"{msg}  ·  Paramètres modifiés — nouveau rendu ensuite"
        self.status_var.set(msg)
    
    def _create_parameter_controls(self, parent):
        """Crée tous les contrôles de paramètres."""
        # Variables pour les paramètres
        self.vars = {}
        
        # Section : Textures de bases
        textures_base_frame = ttk.LabelFrame(parent, text="Textures de bases", padding=10)
        textures_base_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Section : Sélection de l'image de base
        image_select_frame = ttk.Frame(textures_base_frame)
        image_select_frame.pack(fill=tk.X, pady=5)
        
        # Première ligne : Label, Entry et bouton "..."
        first_line = ttk.Frame(image_select_frame)
        first_line.pack(fill=tk.X, pady=(0, 5))
        
        ttk.Label(first_line, text="Image de base:").pack(side=tk.LEFT, padx=(0, 5))
        
        # Barre de lien (Entry en lecture seule pour afficher le chemin)
        self.image_path_var = tk.StringVar(value="Aucune image sélectionnée")
        self.image_path_entry = ttk.Entry(
            first_line,
            textvariable=self.image_path_var,
            state="readonly",
            width=40,
        )
        self.image_path_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))
        
        # Bouton "..."
        browse_btn = ttk.Button(first_line, text="...", width=3, command=self._browse_image)
        browse_btn.pack(side=tk.LEFT, padx=2)
        
        # Deuxième ligne : Bouton "Généré avec l'éditeur de construction"
        second_line = ttk.Frame(image_select_frame)
        second_line.pack(fill=tk.X)
        
        # Bouton "Généré avec l'éditeur de construction" pour ouvrir l'éditeur de construction
        edit_btn = ttk.Button(second_line, text="Généré avec l'éditeur de construction", command=self._open_construction_editor)
        edit_btn.pack(anchor=tk.CENTER, expand=True)
        
        # Séparateur entre Image de base et Image de métal
        separator = ttk.Separator(textures_base_frame, orient=tk.HORIZONTAL)
        separator.pack(fill=tk.X, padx=5, pady=10)
        
        # Section : Sélection de l'image de métal
        metal_select_frame = ttk.Frame(textures_base_frame)
        metal_select_frame.pack(fill=tk.X, pady=5)
        
        # Première ligne : Label, Entry et bouton "..."
        metal_first_line = ttk.Frame(metal_select_frame)
        metal_first_line.pack(fill=tk.X, pady=(0, 5))
        
        ttk.Label(metal_first_line, text="Image de métal:").pack(side=tk.LEFT, padx=(0, 5))
        
        # Barre de lien (Entry en lecture seule pour afficher le chemin)
        self.metal_path_var = tk.StringVar(value="Aucune image sélectionnée")
        self.metal_path_entry = ttk.Entry(
            metal_first_line, textvariable=self.metal_path_var, state="readonly", width=40
        )
        self.metal_path_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))
        
        # Bouton "..."
        browse_metal_btn = ttk.Button(metal_first_line, text="...", width=3, command=self._browse_metal_image)
        browse_metal_btn.pack(side=tk.LEFT, padx=2)
        
        metal_second_line = ttk.Frame(metal_select_frame)
        metal_second_line.pack(fill=tk.X)
        ttk.Button(
            metal_second_line,
            text="Générer une texture métallique",
            command=self._generate_metal_texture,
        ).pack(anchor=tk.CENTER, expand=True)
        
        self.vars["snap_only_skip_contour"] = tk.BooleanVar(value=DEFAULT_SNAP_ONLY_SKIP_CONTOUR)
        self.vars["limit_snap_to_detected_floors"] = tk.BooleanVar(value=True)

        self.surface_blocks_host = ttk.Frame(parent)
        self.surface_blocks_host.pack(fill=tk.X, padx=0, pady=0)
        n_floors = len(FORERUNNER_FLOOR_TITLES)
        self.detected_floor_indices: Optional[List[int]] = None
        self.floor_surface_vars = [None] * n_floors
        self._refresh_surface_parameter_blocks()

        # Bouton d'export
        export_btn = ttk.Button(parent, text="Exporter le résultat", command=self._export_result)
        export_btn.pack(pady=10)

    def _attach_param_tooltip(self, widget, text: str, delay_ms: int = 450) -> None:
        """Infobulle au survol (curseur un instant sur le contrôle)."""
        if not text:
            return
        st: Dict[str, Any] = {"after_id": None, "win": None}

        def cancel():
            aid = st["after_id"]
            if aid is not None:
                try:
                    widget.after_cancel(aid)
                except tk.TclError:
                    pass
                st["after_id"] = None
            tw = st["win"]
            if tw is not None:
                try:
                    tw.destroy()
                except tk.TclError:
                    pass
                st["win"] = None

        def show():
            st["after_id"] = None
            if not widget.winfo_exists():
                return
            try:
                if str(widget.cget("state")) in (tk.DISABLED, "disabled"):
                    return
            except tk.TclError:
                pass
            tw = tk.Toplevel(widget)
            tw.wm_overrideredirect(True)
            try:
                tw.wm_attributes("-topmost", True)
            except tk.TclError:
                pass
            lbl = tk.Label(
                tw,
                text=text,
                background="#ffffe0",
                relief=tk.SOLID,
                borderwidth=1,
                justify=tk.LEFT,
                wraplength=400,
                font=("Segoe UI", 9),
            )
            lbl.pack(ipadx=4, ipady=2)
            tw.update_idletasks()
            x = widget.winfo_rootx() + 12
            y = widget.winfo_rooty() + widget.winfo_height() + 4
            sw = tw.winfo_screenwidth()
            sh = tw.winfo_screenheight()
            tw_w = tw.winfo_width()
            tw_h = tw.winfo_height()
            if x + tw_w > sw - 8:
                x = max(8, sw - tw_w - 8)
            if y + tw_h > sh - 8:
                y = max(8, widget.winfo_rooty() - tw_h - 4)
            tw.geometry(f"+{x}+{y}")
            st["win"] = tw

        def on_enter(_event=None):
            cancel()
            st["after_id"] = widget.after(delay_ms, show)

        def on_leave(_event=None):
            cancel()

        widget.bind("<Enter>", on_enter)
        widget.bind("<Leave>", on_leave)

    def _create_floor_zero_surface_info(self, parent) -> None:
        """Étage 0 (fond) : ne porte pas d'ombre vers le bas ; reçoit l'ombre de l'étage 1."""
        title = FORERUNNER_FLOOR_TITLES[0]
        fb = ttk.LabelFrame(parent, text=f"Étage 0 — {title.split(' — ', 1)[-1]}", padding=10)
        fb.pack(fill=tk.X, padx=5, pady=5)
        hdr = ttk.Frame(fb)
        hdr.pack(fill=tk.X, anchor=tk.W)
        inner = ttk.Frame(fb)
        state = {'expanded': False}

        def toggle_zero():
            state['expanded'] = not state['expanded']
            if state['expanded']:
                inner.pack(fill=tk.X, anchor=tk.W)
                zbtn.configure(text='▲')
            else:
                inner.pack_forget()
                zbtn.configure(text='▼')

        zbtn = ttk.Button(hdr, text='▼', width=3, command=toggle_zero)
        zbtn.pack(side=tk.LEFT)
        ttk.Button(hdr, text="?", width=3, command=self._show_floor_zero_help).pack(side=tk.LEFT, padx=(8, 0))
        ttk.Label(inner, text="Aucun réglage pour ce palier.").pack(anchor=tk.W)

    def _create_floor_surface_block(self, parent, floor_i: int) -> dict:
        """Contrôles surface : blanchiment (glow + grain), grain affleurement (calque séparé), creux."""
        title = FORERUNNER_FLOOR_TITLES[floor_i]
        fb = ttk.LabelFrame(parent, text=f"Paramètres de surface — {title}", padding=10)
        fb.pack(fill=tk.X, padx=5, pady=5)
        fv = {}
        r = 0

        hdr_row = ttk.Frame(fb)
        hdr_row.grid(row=r, column=0, columnspan=3, sticky=tk.W)
        r += 1

        fv['floor_enabled'] = tk.BooleanVar(value=True)
        cb_floor = ttk.Checkbutton(
            hdr_row, text="Traiter cet étage", variable=fv['floor_enabled'],
            command=self._on_parameter_change
        )
        cb_floor.pack(side=tk.LEFT)
        self._attach_param_tooltip(
            cb_floor,
            "Inclut ce palier dans le pipeline surfaces (ombres et blanchiments empilés par étage).",
        )

        inner = ttk.Frame(fb)
        inner_row = r
        r += 1
        collapse_state = {'expanded': False}

        def toggle_floor_body():
            collapse_state['expanded'] = not collapse_state['expanded']
            if collapse_state['expanded']:
                inner.grid(row=inner_row, column=0, columnspan=3, sticky=tk.EW)
                collapse_btn.configure(text='▲')
            else:
                inner.grid_remove()
                collapse_btn.configure(text='▼')

        collapse_btn = ttk.Button(hdr_row, text='▼', width=3, command=toggle_floor_body)
        collapse_btn.pack(side=tk.LEFT, padx=(8, 0))

        ir = 0
        lum_fr = ttk.LabelFrame(
            inner,
            text="Blanchiment",
            padding=6,
        )
        lum_fr.grid(row=ir, column=0, columnspan=3, sticky=tk.EW, pady=(6, 4))
        ir += 1
        rr = 0
        grain_blanchiment_ctrls: list = []

        def apply_grain_blanchiment_sensitivity(*_):
            glow_on = fv['glow_ex_enabled'].get()
            st = tk.NORMAL if glow_on else tk.DISABLED
            if not glow_on and fv['grain_blanchiment_enabled'].get():
                fv['grain_blanchiment_enabled'].set(False)
            for w in grain_blanchiment_ctrls:
                w.configure(state=st)

        def on_glow_ex_toggled():
            apply_grain_blanchiment_sensitivity()
            self._on_parameter_change()

        hdr_ex = ttk.Frame(lum_fr)
        hdr_ex.grid(row=rr, column=0, columnspan=3, sticky=tk.W)
        fv['glow_ex_enabled'] = tk.BooleanVar(value=True)
        cb_glow_ex = ttk.Checkbutton(
            hdr_ex, text="Activer glow de blanchiment", variable=fv['glow_ex_enabled'],
            command=on_glow_ex_toggled,
        )
        cb_glow_ex.pack(side=tk.LEFT)
        self._attach_param_tooltip(
            cb_glow_ex,
            "Glow sur le masque noir et blanc du blanchiment (relief des affleurements). "
            "Désactiver grise aussi le grain blanchiment (même chaîne).",
        )
        ttk.Button(hdr_ex, text="?", width=3, command=self._show_glow_ex_help).pack(side=tk.LEFT, padx=5)
        rr += 1

        fv['fusion_force_inexm'] = tk.DoubleVar(value=0.5)
        lbl_ix = ttk.Label(lum_fr, text="Opacité fusion :")
        lbl_ix.grid(row=rr, column=0, sticky=tk.W, pady=2)
        tip_ix = (
            "Fusion du calque clair (blanchiment) vers la texture métal. "
            "Sans le grain affleurement (bloc séparé ci‑dessous)."
        )
        self._attach_param_tooltip(lbl_ix, tip_ix)
        sc_ix = ttk.Scale(lum_fr, from_=0.0, to=1.0, variable=fv['fusion_force_inexm'],
                          orient=tk.HORIZONTAL, length=160)
        sc_ix.grid(row=rr, column=1, sticky=tk.W, pady=2)
        self._attach_param_tooltip(sc_ix, tip_ix)
        en_ix = ttk.Entry(lum_fr, width=8)
        en_ix.grid(row=rr, column=2, sticky=tk.W, padx=4)
        self._attach_param_tooltip(en_ix, tip_ix)

        def upd_ix(*_a):
            en_ix.delete(0, tk.END)
            en_ix.insert(0, f"{fv['fusion_force_inexm'].get():.2f}")
        fv['fusion_force_inexm'].trace('w', upd_ix)

        def on_ix_motion(_e=None):
            upd_ix()
            self._on_parameter_change()

        sc_ix.bind('<ButtonRelease>', lambda e: self._on_parameter_change())
        sc_ix.bind('<B1-Motion>', on_ix_motion)
        upd_ix()
        rr += 1

        fv['glow_ex_radius'] = tk.IntVar(value=25)
        fv['glow_ex_threshold'] = tk.IntVar(value=127)
        fv['glow_ex_intensity'] = tk.DoubleVar(value=2.5)
        for label, key, vmin, vmax, is_int, tip in [
            ("Rayon :", 'glow_ex_radius', 1, 50, True, "Rayon du glow de blanchissement (zone d'influence, pixels)."),
            ("Seuil :", 'glow_ex_threshold', 0, 255, True, "Luminosité minimale (0–255) pour appliquer le glow sur le masque N/B."),
            ("Intensité :", 'glow_ex_intensity', 0.1, 5.0, False, "Force du blanchiment (glow sur les affleurements)."),
        ]:
            lb = ttk.Label(lum_fr, text=label)
            lb.grid(row=rr, column=0, sticky=tk.W, pady=2)
            self._attach_param_tooltip(lb, tip)
            sc = ttk.Scale(lum_fr, from_=vmin, to=vmax, variable=fv[key], orient=tk.HORIZONTAL, length=160)
            sc.grid(row=rr, column=1, sticky=tk.W, pady=2)
            self._attach_param_tooltip(sc, tip)
            ent = ttk.Entry(lum_fr, width=8)
            ent.grid(row=rr, column=2, sticky=tk.W, padx=4)
            self._attach_param_tooltip(ent, tip)

            def make_upd(k=key, entry=ent, integer=is_int):
                def upd(*_):
                    entry.delete(0, tk.END)
                    v = fv[k].get()
                    entry.insert(0, str(int(v)) if integer else f"{float(v):.2f}")
                return upd

            upd = make_upd()

            def make_commit(k=key, entry=ent, integer=is_int, lo=vmin, hi=vmax):
                def on_commit(_event=None):
                    try:
                        raw = entry.get()
                        v = int(raw) if integer else float(raw)
                        v = max(lo, min(hi, v))
                        fv[k].set(v)
                        self._on_parameter_change()
                    except ValueError:
                        pass
                return on_commit

            commit = make_commit()
            fv[key].trace('w', lambda *a, u=upd: u())
            ent.bind('<Return>', commit)
            ent.bind('<FocusOut>', commit)

            def on_glow_ex_scale_motion(_e=None, u=upd):
                u()
                self._on_parameter_change()

            sc.bind('<ButtonRelease>', lambda e: self._on_parameter_change())
            sc.bind('<B1-Motion>', on_glow_ex_scale_motion)
            upd()
            rr += 1

        ttk.Separator(lum_fr, orient=tk.HORIZONTAL).grid(
            row=rr, column=0, columnspan=3, sticky=tk.EW, pady=(10, 6)
        )
        rr += 1
        gh = ttk.Frame(lum_fr)
        gh.grid(row=rr, column=0, columnspan=3, sticky=tk.W)
        fv['grain_blanchiment_enabled'] = tk.BooleanVar(value=False)
        cb_grain_bl = ttk.Checkbutton(
            gh,
            text="Activer le grain blanchiment",
            variable=fv['grain_blanchiment_enabled'],
            command=self._on_parameter_change,
        )
        cb_grain_bl.pack(side=tk.LEFT)
        self._attach_param_tooltip(
            cb_grain_bl,
            "Spread (éparpillage) sur le calque après le glow ; puis normalisation par palier. "
            "Voir le « ? » pour le détail.",
        )
        btn_grain_bl_help = ttk.Button(gh, text="?", width=3, command=self._show_grain_blanchiment_help)
        btn_grain_bl_help.pack(side=tk.LEFT, padx=5)
        grain_blanchiment_ctrls.extend([cb_grain_bl, btn_grain_bl_help])
        rr += 1

        fv['grain_blanchiment_intensity'] = tk.DoubleVar(value=5.0)
        tip_sg = "Intensité du spread (éparpillage) sur X et Y, en pixels, pour le grain blanchiment."
        lbl_sg = ttk.Label(lum_fr, text="Intensité grain :")
        lbl_sg.grid(row=rr, column=0, sticky=tk.W, pady=2)
        self._attach_param_tooltip(lbl_sg, tip_sg)
        sc_sg = ttk.Scale(
            lum_fr,
            from_=0.0,
            to=10.0,
            variable=fv['grain_blanchiment_intensity'],
            orient=tk.HORIZONTAL,
            length=160,
        )
        sc_sg.grid(row=rr, column=1, sticky=tk.W, pady=2)
        self._attach_param_tooltip(sc_sg, tip_sg)
        en_sg = ttk.Entry(lum_fr, width=8)
        en_sg.grid(row=rr, column=2, sticky=tk.W, padx=4)
        self._attach_param_tooltip(en_sg, tip_sg)
        grain_blanchiment_ctrls.extend([lbl_sg, sc_sg, en_sg])

        def upd_sg(*_):
            en_sg.delete(0, tk.END)
            en_sg.insert(0, f"{fv['grain_blanchiment_intensity'].get():.2f}")

        fv['grain_blanchiment_intensity'].trace('w', upd_sg)

        def on_sg_motion(_e=None):
            upd_sg()
            self._on_parameter_change()

        sc_sg.bind('<ButtonRelease>', lambda e: self._on_parameter_change())
        sc_sg.bind('<B1-Motion>', on_sg_motion)
        upd_sg()
        apply_grain_blanchiment_sensitivity()

        affleur_fr = ttk.LabelFrame(
            inner,
            text="Grain affleurement",
            padding=6,
        )
        affleur_fr.grid(row=ir, column=0, columnspan=3, sticky=tk.EW, pady=(2, 6))
        ir += 1
        ar = 0
        gah = ttk.Frame(affleur_fr)
        gah.grid(row=ar, column=0, columnspan=3, sticky=tk.W)
        fv['grain_affleurement_enabled'] = tk.BooleanVar(value=True)
        cb_grain_af = ttk.Checkbutton(
            gah,
            text="Activer le grain affleurement",
            variable=fv['grain_affleurement_enabled'],
            command=self._on_parameter_change,
        )
        cb_grain_af.pack(side=tk.LEFT)
        self._attach_param_tooltip(
            cb_grain_af,
            "Grain sur une couche séparée, limitée au palier courant, puis mélangée au blanchiment.",
        )
        ttk.Button(gah, text="?", width=3, command=self._show_grain_affleurement_help).pack(side=tk.LEFT, padx=5)
        ar += 1
        fv['grain_affleurement_intensity'] = tk.DoubleVar(value=5.0)
        tip_ga = "Intensité du spread (éparpillage) sur X et Y pour le grain limité au palier courant."
        lbl_ga = ttk.Label(affleur_fr, text="Intensité :")
        lbl_ga.grid(row=ar, column=0, sticky=tk.W, pady=2)
        self._attach_param_tooltip(lbl_ga, tip_ga)
        sc_ga = ttk.Scale(
            affleur_fr,
            from_=0.0,
            to=10.0,
            variable=fv['grain_affleurement_intensity'],
            orient=tk.HORIZONTAL,
            length=160,
        )
        sc_ga.grid(row=ar, column=1, sticky=tk.W, pady=2)
        self._attach_param_tooltip(sc_ga, tip_ga)
        en_ga = ttk.Entry(affleur_fr, width=8)
        en_ga.grid(row=ar, column=2, sticky=tk.W, padx=4)
        self._attach_param_tooltip(en_ga, tip_ga)

        def upd_ga(*_):
            en_ga.delete(0, tk.END)
            en_ga.insert(0, f"{fv['grain_affleurement_intensity'].get():.2f}")

        fv['grain_affleurement_intensity'].trace('w', upd_ga)

        def on_ga_motion(_e=None):
            upd_ga()
            self._on_parameter_change()

        sc_ga.bind('<ButtonRelease>', lambda e: self._on_parameter_change())
        sc_ga.bind('<B1-Motion>', on_ga_motion)
        upd_ga()
        ar += 1

        fv['grain_affleurement_opacity'] = tk.DoubleVar(value=0.5)
        tip_gao = "Mélange du calque grain affleurement avec le blanchiment (0 = blanchiment seul, 1 = affleurement max)."
        lbl_gao = ttk.Label(affleur_fr, text="Opacité mélange :")
        lbl_gao.grid(row=ar, column=0, sticky=tk.W, pady=2)
        self._attach_param_tooltip(lbl_gao, tip_gao)
        sc_gao = ttk.Scale(
            affleur_fr,
            from_=0.0,
            to=1.0,
            variable=fv['grain_affleurement_opacity'],
            orient=tk.HORIZONTAL,
            length=160,
        )
        sc_gao.grid(row=ar, column=1, sticky=tk.W, pady=2)
        self._attach_param_tooltip(sc_gao, tip_gao)
        en_gao = ttk.Entry(affleur_fr, width=8)
        en_gao.grid(row=ar, column=2, sticky=tk.W, padx=4)
        self._attach_param_tooltip(en_gao, tip_gao)

        def upd_gao(*_):
            en_gao.delete(0, tk.END)
            en_gao.insert(0, f"{fv['grain_affleurement_opacity'].get():.2f}")

        fv['grain_affleurement_opacity'].trace('w', upd_gao)

        def on_gao_motion(_e=None):
            upd_gao()
            self._on_parameter_change()

        sc_gao.bind('<ButtonRelease>', lambda e: self._on_parameter_change())
        sc_gao.bind('<B1-Motion>', on_gao_motion)
        upd_gao()

        ttk.Separator(inner, orient=tk.HORIZONTAL).grid(row=ir, column=0, columnspan=3, sticky=tk.EW, pady=8)
        ir += 1
        tip_creux = (
            "Ombre projetée par ce palier vers le palier inférieur (creux). "
            "Voir aussi l’aide « ? » à côté de « Activer ombrages »."
        )
        lbl_creux = ttk.Label(inner, text="Ombrages", font=("Arial", 9, "bold"))
        lbl_creux.grid(row=ir, column=0, columnspan=3, sticky=tk.W)
        self._attach_param_tooltip(lbl_creux, tip_creux)
        ir += 1
        hdr_in = ttk.Frame(inner)
        hdr_in.grid(row=ir, column=0, columnspan=3, sticky=tk.W)
        fv['ombrage_enabled'] = tk.BooleanVar(value=True)
        cb_omb = ttk.Checkbutton(
            hdr_in, text="Activer ombrages", variable=fv['ombrage_enabled'],
            command=self._on_parameter_change
        )
        cb_omb.pack(side=tk.LEFT)
        self._attach_param_tooltip(cb_omb, "Chaîne d’ombre de ce palier vers le palier inférieur (creux).")
        ttk.Button(hdr_in, text="?", width=3, command=self._show_glow_in_help).pack(
            side=tk.LEFT, padx=(8, 4)
        )
        ir += 1

        fv['fusion_force_inm'] = tk.DoubleVar(value=0.5)
        tip_im = "Opacité de fusion du calque d’ombre (creux) sur la cible sous ce palier."
        lbl_im = ttk.Label(inner, text="Opacité ombre :")
        lbl_im.grid(row=ir, column=0, sticky=tk.W, pady=2)
        self._attach_param_tooltip(lbl_im, tip_im)
        sc_im = ttk.Scale(inner, from_=0.0, to=1.0, variable=fv['fusion_force_inm'],
                          orient=tk.HORIZONTAL, length=160)
        sc_im.grid(row=ir, column=1, sticky=tk.W, pady=2)
        self._attach_param_tooltip(sc_im, tip_im)
        en_im = ttk.Entry(inner, width=8)
        en_im.grid(row=ir, column=2, sticky=tk.W, padx=4)
        self._attach_param_tooltip(en_im, tip_im)

        def upd_im(*_):
            en_im.delete(0, tk.END)
            en_im.insert(0, f"{fv['fusion_force_inm'].get():.2f}")
        fv['fusion_force_inm'].trace('w', upd_im)

        def on_im_motion(_e=None):
            upd_im()
            self._on_parameter_change()

        sc_im.bind('<ButtonRelease>', lambda e: self._on_parameter_change())
        sc_im.bind('<B1-Motion>', on_im_motion)
        upd_im()
        ir += 1

        fv['glow_in_radius'] = tk.IntVar(value=25)
        fv['glow_in_threshold'] = tk.IntVar(value=127)
        fv['glow_in_intensity'] = tk.DoubleVar(value=2.5)
        for label, key, vmin, vmax, is_int, tip in [
            ("Rayon :", 'glow_in_radius', 1, 50, True, "Taille de la zone d’influence du glow d’ombre (pixels)."),
            ("Seuil :", 'glow_in_threshold', 0, 255, True, "Luminosité (0–255) à partir de laquelle l’ombre s’applique sur le masque."),
            ("Intensité :", 'glow_in_intensity', 0.1, 5.0, False, "Force du glow d’ombre (assombrissement des creux)."),
        ]:
            lb = ttk.Label(inner, text=label)
            lb.grid(row=ir, column=0, sticky=tk.W, pady=2)
            self._attach_param_tooltip(lb, tip)
            sc = ttk.Scale(inner, from_=vmin, to=vmax, variable=fv[key], orient=tk.HORIZONTAL, length=160)
            sc.grid(row=ir, column=1, sticky=tk.W, pady=2)
            self._attach_param_tooltip(sc, tip)
            ent = ttk.Entry(inner, width=8)
            ent.grid(row=ir, column=2, sticky=tk.W, padx=4)
            self._attach_param_tooltip(ent, tip)

            def make_upd(k=key, entry=ent, integer=is_int):
                def upd(*_):
                    entry.delete(0, tk.END)
                    v = fv[k].get()
                    entry.insert(0, str(int(v)) if integer else f"{float(v):.2f}")
                return upd

            upd = make_upd()

            def make_commit(k=key, entry=ent, integer=is_int, lo=vmin, hi=vmax):
                def on_commit(_event=None):
                    try:
                        raw = entry.get()
                        v = int(raw) if integer else float(raw)
                        v = max(lo, min(hi, v))
                        fv[k].set(v)
                        self._on_parameter_change()
                    except ValueError:
                        pass
                return on_commit

            commit = make_commit()
            fv[key].trace('w', lambda *a, u=upd: u())
            ent.bind('<Return>', commit)
            ent.bind('<FocusOut>', commit)

            def on_glow_in_scale_motion(_e=None, u=upd):
                u()
                self._on_parameter_change()

            sc.bind('<ButtonRelease>', lambda e: self._on_parameter_change())
            sc.bind('<B1-Motion>', on_glow_in_scale_motion)
            upd()
            ir += 1

        return fv

    def _refresh_surface_parameter_blocks(self) -> None:
        """Recrée les blocs surface selon l'image et l'option « étages détectés » (variables internes)."""
        if not hasattr(self, 'surface_blocks_host'):
            return
        for w in self.surface_blocks_host.winfo_children():
            w.destroy()
        n = len(FORERUNNER_FLOOR_TITLES)
        self.floor_surface_vars = [None] * n

        if self.current_image is not None:
            self.detected_floor_indices = detect_used_forerunner_floor_indices(self.current_image)
            adapt = self.vars['limit_snap_to_detected_floors'].get()
            if adapt:
                surface_only = [i for i in self.detected_floor_indices if i > 0]
                if not surface_only:
                    ttk.Label(
                        self.surface_blocks_host,
                        text=(
                            "Aucun palier avec paramètres de surface "
                            "(image sans étages 1–5 reconnus, ou seul le noir)."
                        ),
                        wraplength=320,
                        justify=tk.LEFT,
                    ).pack(anchor=tk.W, padx=5, pady=5)
                else:
                    if 0 in self.detected_floor_indices:
                        self._create_floor_zero_surface_info(self.surface_blocks_host)
                    for floor_i in sorted(surface_only):
                        self.floor_surface_vars[floor_i] = self._create_floor_surface_block(
                            self.surface_blocks_host, floor_i
                        )
            else:
                if 0 in self.detected_floor_indices:
                    self._create_floor_zero_surface_info(self.surface_blocks_host)
                for floor_i in range(1, n):
                    self.floor_surface_vars[floor_i] = self._create_floor_surface_block(
                        self.surface_blocks_host, floor_i
                    )
        else:
            self.detected_floor_indices = None
            for floor_i in range(1, n):
                self.floor_surface_vars[floor_i] = self._create_floor_surface_block(
                    self.surface_blocks_host, floor_i
                )
    
    def _load_default_values(self):
        """Charge les valeurs par défaut depuis settings.py."""
        # Les valeurs sont déjà initialisées dans _create_parameter_controls
        pass
    
    def _show_floor_zero_help(self):
        """Aide : étage 0 (fond), pas de réglages surface."""
        help_text = """Étage 0 (fond)

Aucun réglage ici : le fond ne projette pas d’ombre vers un palier plus bas. Les pixels au gris du fond reçoivent en revanche l’ombre portée par l’étage au-dessus (étage 1) — activez et réglez les ombrages dans le bloc « Étage 1 »."""
        messagebox.showinfo("Aide - Étage 0 (fond)", help_text)

    def _show_grain_blanchiment_help(self):
        """Aide : grain sur la couche blanchiment avant découpe."""
        help_text = """Grain blanchiment (spread / « éparpiller »)

Après le glow, une étape intermédiaire (``blanch_02b_prepare_sous_palier_avant_grain``, logique ``analyze_grain.remplir_etages_dessous_reference_affleurement``) uniformise la couleur des étages **sous** le palier avec la teinte du premier pixel du palier, **puis** le spread agit sur tout le calque. Ensuite la normalisation par palier (teinte sous palier / noir au-dessus / dégradé sur le palier) s’applique comme d’habitude.

Un module de préparation du grain est utilisé en interne pendant le traitement.

À régler séparément du « grain affleurement » (autre bloc — voir son aide).

Équivalent GIMP : Éparpiller."""
        messagebox.showinfo("Aide - Grain blanchiment", help_text)

    def _show_grain_affleurement_help(self):
        """Aide : grain sur la zone affleurement seule."""
        help_text = """Grain affleurement (logique séparée du blanchiment)

En bref : masque blanc/noir par palier → spread → ne garder que le palier courant ; fusion avec le blanchiment (réglage par opacité).

1) Image test : blanc sous le palier traité, noir sur ce palier et au-dessus.
2) Spread (éparpiller) sur toute l’image.
3) Noir partout sauf sur le palier courant (le grain ne reste que là → bouillie près des bords).
4) Fusion avec le résultat du blanchiment : l’opacité règle combien le grain affleurement se mélange au blanchiment (0 = blanchiment seul, 1 = prise en compte maximale du calque affleurement).

5) Le noir devient transparent à la fusion métal comme d’habitude.

Indépendant du grain appliqué sur la couche de blanchiment (glow + collage par palier)."""
        messagebox.showinfo("Aide - Grain affleurement", help_text)
    
    def _show_glow_ex_help(self):
        """Affiche l'aide pour le blanchiment des affleurements."""
        help_text = """Blanchiment des affleurements

Éclairage sur les zones élevées (affleurements), pour le relief.

• Rayon : zone d'influence du glow (px)
• Seuil : luminosité à partir de laquelle l'effet s'applique (0–255)
• Intensité : force du blanchiment (0.1–5.0)
• Opacité (fusion) : fusion du calque clair (blanchiment) vers la texture métal (0.0–1.0). Ce réglage ne concerne pas le grain affleurement (bloc séparé).

Le grain sur le masque N/B (avant découpe) est réglé sous cette section ; le grain « affleurement » a son propre bloc plus bas."""
        messagebox.showinfo("Aide - Blanchiment des affleurements", help_text)
    
    def _show_glow_in_help(self):
        """Affiche l'aide pour les ombrages."""
        help_text = """Ombrages

Cet effet applique un assombrissement sur les zones creuses (ombres) de la texture, renforçant les creux et les détails.

• Rayon : Contrôle la taille de la zone d'influence de l'ombre (en pixels)
• Seuil : Détermine le niveau de luminosité à partir duquel l'effet s'applique (0-255)
• Intensité : Force de l'effet d'ombrage (0.1-5.0)
• Opacité : Transparence de l'effet lors de la fusion avec l'image originale (0.0-1.0)

Des valeurs élevées créent des ombres plus profondes, tandis que des valeurs faibles donnent un effet subtil.

Palier blanc (étage 5) : le palier le plus haut ne reçoit pas d’ombre dans le pipeline ; seuls les gris des étages 0 à 4 sont cibles de réception d’ombre depuis le palier au-dessus."""
        messagebox.showinfo("Aide - Ombrages", help_text)
    
    def _show_about(self):
        """Affiche la fenêtre À propos avec les informations de contact."""
        # Créer une fenêtre personnalisée pour permettre les liens cliquables
        about_window = tk.Toplevel(self.root)
        about_window.title("À propos")
        about_window.geometry("500x300")
        about_window.resizable(False, False)
        
        # Centrer la fenêtre
        about_window.transient(self.root)
        about_window.grab_set()
        
        # Frame principal
        main_frame = ttk.Frame(about_window, padding=20)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Titre
        title_label = ttk.Label(main_frame, text="Forerunner Texture Generator", 
                               font=("Arial", 14, "bold"))
        title_label.pack(pady=(0, 10))
        
        # Description
        desc_label = ttk.Label(main_frame, text="Générateur de textures Forerunner pour Halo",
                              font=("Arial", 10))
        desc_label.pack(pady=(0, 20))
        
        # Informations de contact
        contact_frame = ttk.Frame(main_frame)
        contact_frame.pack(fill=tk.X, pady=10)
        
        ttk.Label(contact_frame, text="Contact et ressources :", font=("Arial", 9, "bold")).pack(anchor=tk.W, pady=5)
        
        # E-mail
        email_frame = ttk.Frame(contact_frame)
        email_frame.pack(fill=tk.X, pady=2)
        ttk.Label(email_frame, text="E-mail :").pack(side=tk.LEFT, padx=5)
        email_label = ttk.Label(email_frame, text="gaigher@hotmail.fr", 
                               foreground="blue", cursor="hand2")
        email_label.pack(side=tk.LEFT)
        email_label.bind("<Button-1>", lambda e: self._open_email())
        
        # Site web
        web_frame = ttk.Frame(contact_frame)
        web_frame.pack(fill=tk.X, pady=2)
        ttk.Label(web_frame, text="Site web :").pack(side=tk.LEFT, padx=5)
        web_label = ttk.Label(web_frame, text="www.gaigher.fr", 
                             foreground="blue", cursor="hand2")
        web_label.pack(side=tk.LEFT)
        web_label.bind("<Button-1>", lambda e: webbrowser.open("http://www.gaigher.fr"))
        
        # GitHub
        github_frame = ttk.Frame(contact_frame)
        github_frame.pack(fill=tk.X, pady=2)
        ttk.Label(github_frame, text="GitHub :").pack(side=tk.LEFT, padx=5)
        github_label = ttk.Label(github_frame, text="https://github.com/gaigher", 
                                foreground="blue", cursor="hand2")
        github_label.pack(side=tk.LEFT)
        github_label.bind("<Button-1>", lambda e: webbrowser.open("https://github.com/gaigher"))
        
        # YouTube
        youtube_frame = ttk.Frame(contact_frame)
        youtube_frame.pack(fill=tk.X, pady=2)
        ttk.Label(youtube_frame, text="YouTube :").pack(side=tk.LEFT, padx=5)
        youtube_label = ttk.Label(youtube_frame, 
                                 text="https://www.youtube.com/channel/UCI_Vauxg9rZlKI0A_5-ZOsA/videos", 
                                 foreground="blue", cursor="hand2")
        youtube_label.pack(side=tk.LEFT)
        youtube_label.bind("<Button-1>", lambda e: webbrowser.open("https://www.youtube.com/channel/UCI_Vauxg9rZlKI0A_5-ZOsA/videos"))
    
    def _open_email(self):
        """Ouvre le client de messagerie avec l'adresse e-mail."""
        import subprocess
        import platform
        
        email = "gaigher@hotmail.fr"
        if platform.system() == "Windows":
            # Windows
            try:
                subprocess.run(["start", f"mailto:{email}"], shell=True, check=True)
            except:
                # Repli : copier dans le presse-papiers
                self.root.clipboard_clear()
                self.root.clipboard_append(email)
                messagebox.showinfo("E-mail", f"Adresse e-mail copiée dans le presse-papiers :\n{email}")
        elif platform.system() == "Darwin":
            # macOS
            subprocess.run(["open", f"mailto:{email}"])
        else:
            # Linux
            subprocess.run(["xdg-email", email])
    
    def _set_image_path_display(self, text: str) -> None:
        """Affiche le libellé ou chemin de l'image de base.

        Sous Windows, un ttk.Entry en lecture seule ne repeint pas toujours l'affichage
        si l'on ne fait que StringVar.set() (le texte n'apparaît qu'après un autre
        événement d'interface). On force donc un cycle normal → lecture seule.
        """
        self.image_path_var.set(text)
        e = getattr(self, "image_path_entry", None)
        if e is not None:
            e.configure(state="normal")
            e.configure(state="readonly")
            self.root.update_idletasks()

    def _on_source_image_replaced(self):
        """Invalide le cache snap Forerunner (nouvelle image source)."""
        self._source_image_generation += 1
        self._snap_cache_key = None
        self._snap_cache_tb = None
        self._snap_cache_tb_q_presnap = None
        self._snap_cache_tb_q_snapped = None
        self._snap_cache_infer = None
        self._surface_global_key = None
        self._floor_fingerprints = None
        self._floor_checkpoints_after = None
        self._floor_layer_cache.clear()
        if self._metal_procedural_spec is not None:
            self._materialize_procedural_metal_file()

    def _metal_target_dimensions(self) -> Tuple[int, int]:
        """Largeur × hauteur pour la TM : celle de l’image de base, ou 1024×1024 si pas de base."""
        if self.current_image is not None:
            w, h = self.current_image.size
            return (w, h)
        return (1024, 1024)

    def _reference_metal_file_pixel_size(self) -> Optional[Tuple[int, int]]:
        """
        Taille native du fichier TM (pas la TM procédurale : elle suit déjà l’image de base).
        Utilisé pour comparer à la construction importée / fabriquée.
        """
        if not self.current_tm_path or not Path(self.current_tm_path).exists():
            return None
        if self._metal_procedural_spec is not None:
            return None
        try:
            return charger_image(self.current_tm_path).size
        except Exception:
            return None

    def _apply_default_tm_fit_construction_over_metal(self) -> None:
        """
        L’image de base (lignes de construction) prime sur le fichier métal : si les tailles
        diffèrent, le mode d’adaptation TM par défaut est mosaïque (tuiles au plus proche puis étirement).
        """
        if self.current_image is None:
            return
        tm_wh = self._reference_metal_file_pixel_size()
        if tm_wh is None:
            return
        iw, ih = self.current_image.size
        if (iw, ih) != tm_wh:
            self.tm_fit_mode_var.set("tile_approx_stretch")

    def _clear_procedural_metal_state(self) -> None:
        """Quitte le mode TM procédurale et supprime le fichier temporaire associé."""
        self._metal_procedural_spec = None
        if self._tm_procedural_temp_path and Path(self._tm_procedural_temp_path).exists():
            try:
                os.unlink(self._tm_procedural_temp_path)
            except OSError:
                pass
        self._tm_procedural_temp_path = None

    def _materialize_procedural_metal_file(self) -> None:
        """Écrit la PNG procédurale (dimensions courantes) sans lancer le pipeline."""
        assert self._metal_procedural_spec is not None
        seed = self._metal_procedural_spec["seed"]
        w, h = self._metal_target_dimensions()
        img = make_seamless_metal_texture(w, h, seed=seed)
        if self._tm_procedural_temp_path is None:
            fd, path = tempfile.mkstemp(suffix=".png")
            os.close(fd)
            self._tm_procedural_temp_path = path
        img.save(self._tm_procedural_temp_path, format="PNG")
        self.current_tm_path = self._tm_procedural_temp_path
        if hasattr(self, "metal_path_var"):
            self.metal_path_var.set(METAL_PROCEDURAL_DISPLAY)
            entry = getattr(self, "metal_path_entry", None)
            if entry is not None:
                entry.configure(state="normal")
                entry.configure(state="readonly")

    def _debounced_process_image(self) -> None:
        """Invoqué après le délai de regroupement : libère l'identifiant du temporisateur puis lance le pipeline."""
        self.update_timer = None
        self._process_image()

    def _on_parameter_change(self, event=None):
        """Appelé quand un paramètre change — relance le pipeline après un court délai (regroupement / anti-rebond)."""
        if self.current_image is None:
            return
        
        # Annuler le timer précédent
        if self.update_timer:
            try:
                self.root.after_cancel(self.update_timer)
            except tk.TclError:
                pass
            self.update_timer = None

        # Pendant un rendu : marquer tout de suite la file d’attente (sans attendre le regroupement temporel).
        if self.is_processing:
            self._preview_pending = True
            if self._status_has_progress and self._last_progress_msg:
                self.status_var.set(
                    f"{self._last_progress_msg}  ·  Paramètres modifiés — nouveau rendu ensuite"
                )
            else:
                self.status_var.set(
                    "Traitement en cours — paramètres modifiés, nouveau rendu ensuite…"
                )
            return

        self.status_var.set("Paramètres modifiés - Mise à jour en cours...")
        self.update_timer = self.root.after(PREVIEW_DEBOUNCE_MS, self._debounced_process_image)
    
    def _on_fast_preview_toggle(self) -> None:
        """Bascule l’aperçu pleine résolution / réduit ; invalide le cache via nouvelle taille de travail."""
        if self.current_image is not None:
            self._on_parameter_change()
        elif self.current_tm_path:
            self._preview_metal_texture_only()
    
    def _image_for_preview_pipeline(self) -> Image.Image:
        """Image envoyée au pipeline pour la prévisualisation (réduite si « Mod Rapide »)."""
        assert self.current_image is not None
        img = self.current_image
        if not self.fast_preview_var.get():
            return img.copy()
        w, h = img.size
        max_edge = FAST_PREVIEW_MAX_EDGE
        if max(w, h) <= max_edge:
            return img.copy()
        scale = max_edge / float(max(w, h))
        nw = max(1, int(round(w * scale)))
        nh = max(1, int(round(h * scale)))
        return img.resize((nw, nh), Image.Resampling.LANCZOS)
    
    def _browse_image(self):
        """Ouvre un dialogue pour sélectionner une image."""
        file_path = filedialog.askopenfilename(
            title="Charger une image",
            filetypes=[
                ("Images", "*.png *.jpg *.jpeg *.tiff *.tif *.bmp"),
                ("Tous les fichiers", "*.*")
            ]
        )
        
        if file_path:
            self._load_image(file_path)
    
    def _mise_au_propre_opts(self) -> dict:
        """Options pipeline pour le mode mise au propre (snap Forerunner, sans surfaces ni TM)."""
        return {
            'niveaux': 0,
            'do_contour': False,
            'contour_methode': DEFAULT_CONTOUR_METHODE,
            'seuil_flou': DEFAULT_SEUIL_FLOU,
            'snap_only': True,
            'snap_only_skip_contour': self.vars['snap_only_skip_contour'].get(),
            'snap_forerunner_floors': True,
            'force_forerunner_snap': False,
            'snap_use_full_palette': False,
            'auto_sharpen_blur': DEFAULT_AUTO_SHARPEN_BLUR,
            'auto_antialias_pass': DEFAULT_AUTO_ANTIALIAS_PASS,
            'allow_conversion': True,
        }

    def _apply_mise_au_propre(self, img: Image.Image) -> Image.Image:
        """Snap Forerunner et corrections anti-flou si besoin, sans surfaces ni texture métal."""
        rgba = img.convert('RGBA')
        res = process_image_stepwise(rgba, self._mise_au_propre_opts())
        return res['final'].convert('RGB')

    def _open_construction_editor(self):
        """Ouvre l'éditeur de construction Forerunner en mode modal."""
        try:
            from ui.construction_editor import ConstructionEditor
            
            # Créer une nouvelle fenêtre pour l'éditeur
            editor_window = tk.Toplevel(self.root)
            editor_window.transient(self.root)
            editor_window.grab_set()
            
            ref_metal_wh: Optional[Tuple[int, int]] = None
            if self.current_tm_path and Path(self.current_tm_path).exists():
                try:
                    if self._metal_procedural_spec is not None:
                        ref_metal_wh = self._metal_target_dimensions()
                    else:
                        ref_metal_wh = charger_image(self.current_tm_path).size
                except Exception:
                    ref_metal_wh = None
            
            # Préparer l'image initiale si une image est déjà chargée
            initial_image = None
            if self.current_image:
                # Convertir en niveaux de gris si nécessaire
                if self.current_image.mode != 'L':
                    initial_image = self.current_image.convert('L')
                else:
                    initial_image = self.current_image.copy()
            
            # Construction importée ou déjà en mémoire : dimensions du dessin prioritaires ;
            # adaptation TM par défaut = mosaïque au plus proche puis étirement.
            initial_tm_fit = (
                "tile_approx_stretch" if initial_image is not None else self.tm_fit_mode_var.get()
            )
            # Créer l'éditeur en mode modal
            editor = ConstructionEditor(
                editor_window,
                modal=True,
                initial_image=initial_image,
                reference_metal_size=ref_metal_wh,
                initial_tm_fit_mode=initial_tm_fit,
            )
            
            # Attendre que la fenêtre soit fermée
            editor_window.wait_window()
            
            # Récupérer l'image résultante
            if editor.result_image is not None:
                if editor.result_tm_fit_mode:
                    self.tm_fit_mode_var.set(editor.result_tm_fit_mode)
                # L'éditeur retourne une image en niveaux de gris (mode 'L')
                if editor.result_image.mode == 'L':
                    brut_rgb = editor.result_image.convert('RGB')
                else:
                    brut_rgb = editor.result_image.convert('RGB')
                self.current_image = self._apply_mise_au_propre(brut_rgb)
                self._on_source_image_replaced()
                
                self._set_image_path_display(
                    "Image créée avec l'éditeur de construction (mise au propre)"
                )
                
                self.status_var.set(
                    "Éditeur : dessin enregistré avec snap Forerunner (mise au propre)."
                )
                self._refresh_surface_parameter_blocks()
                self._pipeline_visual_on_first_base_load = True
                self._process_image()
            else:
                self.status_var.set("Édition annulée")
        except ImportError:
            messagebox.showerror("Erreur", "Impossible d'importer l'éditeur de construction.\nVérifiez que ui/construction_editor.py est disponible.")
        except Exception as e:
            messagebox.showerror("Erreur", f"Erreur lors de l'ouverture de l'éditeur:\n{e}")
    
    def _load_image(self, file_path=None):
        """Charge une image depuis un fichier."""
        if file_path is None:
            file_path = filedialog.askopenfilename(
                title="Charger une image",
                filetypes=[
                    ("Images", "*.png *.jpg *.jpeg *.tiff *.tif *.bmp"),
                    ("Tous les fichiers", "*.*")
                ]
            )
        
        if file_path:
            try:
                self.current_image = charger_image(file_path)
                self._on_source_image_replaced()
                self._apply_default_tm_fit_construction_over_metal()
                im_rgba = self.current_image.convert("RGBA")
                flou_score = detecter_flou(im_rgba)
                if flou_score < DEFAULT_SEUIL_FLOU:
                    messagebox.showwarning(
                        "Dessin flou ou contours trop mou détecté",
                        "L'image semble peu nette : le score combine la finesse globale"
                        " et la netteté le long des changements de luminance (y compris sur un dessin "
                        "à aplats).\n\n"
                        "Lors du traitement, un anti-flou automatique (masque flou / netteté) est "
                        "appliqué avant les contours et le recollage Forerunner, puis le nettoyage des "
                        "étages parasites.\n\n"
                        f"Indicateur au chargement : {flou_score:.1f} (seuil : {DEFAULT_SEUIL_FLOU:.0f}).",
                    )
                self._set_image_path_display(file_path)
                self.status_var.set(f"Image chargée: {Path(file_path).name}")
                self._refresh_surface_parameter_blocks()
                self._pipeline_visual_on_first_base_load = True
                self._process_image()
            except Exception as e:
                messagebox.showerror("Erreur", f"Impossible de charger l'image:\n{e}")
    
    def _browse_metal_image(self):
        """Ouvre un dialogue pour sélectionner une image de métal."""
        file_path = filedialog.askopenfilename(
            title="Charger texture métallique",
            filetypes=[
                ("Images", "*.png *.jpg *.jpeg *.tiff *.tif *.bmp"),
                ("Tous les fichiers", "*.*")
            ]
        )
        
        if file_path:
            self._load_metal_texture(file_path)
    
    def _load_metal_texture(self, file_path=None):
        """Charge une texture métallique."""
        if file_path is None:
            file_path = filedialog.askopenfilename(
                title="Charger texture métallique",
                filetypes=[
                    ("Images", "*.png *.jpg *.jpeg *.tiff *.tif *.bmp"),
                    ("Tous les fichiers", "*.*")
                ]
            )
        
        if file_path:
            self._clear_procedural_metal_state()
            self.current_tm_path = file_path
            if hasattr(self, "metal_path_var"):
                self.metal_path_var.set(file_path)
                if hasattr(self, "metal_path_entry"):
                    self.metal_path_entry.configure(state="normal")
                    self.metal_path_entry.configure(state="readonly")
            self.status_var.set(f"Texture métallique chargée: {Path(file_path).name}")
            if self.current_image:
                self._apply_default_tm_fit_construction_over_metal()
                self._process_image()
            else:
                self._preview_metal_texture_only()
    
    def _preview_metal_texture_only(self) -> None:
        """Affiche la texture métallique dans l'aperçu lorsqu'aucune image de base n'est chargée."""
        p = self.current_tm_path
        if not p or not Path(p).exists():
            return
        try:
            tm = charger_image(p)
            rgb = image_to_preview_rgb(tm)
            if self.fast_preview_var.get():
                w, h = rgb.size
                max_edge = FAST_PREVIEW_MAX_EDGE
                if max(w, h) > max_edge:
                    scale = max_edge / float(max(w, h))
                    nw = max(1, int(round(w * scale)))
                    nh = max(1, int(round(h * scale)))
                    rgb = rgb.resize((nw, nh), Image.Resampling.LANCZOS)
            self.preview_image = rgb
            self.preview_step_var.set("Texture métallique (sans image de base)")
            self._update_preview()
        except Exception as e:
            messagebox.showerror("Erreur", f"Impossible d'afficher la texture métallique:\n{e}")
    
    def _generate_metal_texture(self) -> None:
        """Génère une texture métallique procédurale sans couture, calée sur l’image de base."""
        try:
            # Nouvelle graine à chaque clic ; le même seed est conservé lors d’un changement de taille de la base.
            self._metal_procedural_spec = {"seed": secrets.randbelow(2**31)}
            self._materialize_procedural_metal_file()
            w, h = self._metal_target_dimensions()
            self.status_var.set(f"Texture métallique procédurale {w}×{h} (carrelable)")
            if self.current_image is not None:
                self._process_image()
            else:
                self._preview_metal_texture_only()
        except Exception as e:
            messagebox.showerror("Erreur", f"Impossible de générer la texture métallique:\n{e}")
    
    def _build_options(self, seed_wh: Optional[Tuple[int, int]] = None) -> dict:
        """Construit le dictionnaire d'options depuis les variables.

        seed_wh : dimensions utilisées pour la graine de grain (défaut : image courante).
        """
        _noop_floor = {
            'enabled': False,
            'glow_ex_enabled': False,
            'ombrage_enabled': False,
            'glow_on_tb': False,
            'glow_ex': {'radius': 25, 'threshold': 127, 'intensity': 2.5},
            'glow_in': {'radius': 25, 'threshold': 127, 'intensity': 2.5},
            'grain_blanchiment': {
                'enabled': False, 'spread_x': 1.0, 'spread_y': 1.0, 'seed': None,
            },
            'grain_affleurement': {
                'enabled': False, 'spread_x': 1.0, 'spread_y': 1.0, 'opacity': 1.0, 'seed': None,
            },
            'fusion_force_inm': 0.0,
            'fusion_force_inexm': 0.0,
        }
        floors = []
        for fv in self.floor_surface_vars:
            if fv is None:
                floors.append(dict(_noop_floor))
                continue
            floors.append({
                'enabled': fv['floor_enabled'].get(),
                'glow_ex_enabled': fv['glow_ex_enabled'].get(),
                'ombrage_enabled': fv['ombrage_enabled'].get(),
                'glow_on_tb': True,
                'glow_ex': {
                    'radius': fv['glow_ex_radius'].get(),
                    'threshold': fv['glow_ex_threshold'].get(),
                    'intensity': fv['glow_ex_intensity'].get(),
                },
                'glow_in': {
                    'radius': fv['glow_in_radius'].get(),
                    'threshold': fv['glow_in_threshold'].get(),
                    'intensity': fv['glow_in_intensity'].get(),
                },
                'grain_blanchiment': {
                    'enabled': fv['grain_blanchiment_enabled'].get() and fv['glow_ex_enabled'].get(),
                    'spread_x': fv['grain_blanchiment_intensity'].get(),
                    'spread_y': fv['grain_blanchiment_intensity'].get(),
                    'seed': None,
                },
                'grain_affleurement': {
                    'enabled': fv['grain_affleurement_enabled'].get(),
                    'spread_x': fv['grain_affleurement_intensity'].get(),
                    'spread_y': fv['grain_affleurement_intensity'].get(),
                    'opacity': fv['grain_affleurement_opacity'].get(),
                    'seed': None,
                },
                'fusion_force_inm': fv['fusion_force_inm'].get(),
                'fusion_force_inexm': fv['fusion_force_inexm'].get(),
            })
        use_limit = self.vars['limit_snap_to_detected_floors'].get()
        if use_limit and self.detected_floor_indices:
            process_subset = [i for i in self.detected_floor_indices if i > 0]
        else:
            process_subset = None

        if seed_wh is not None:
            w, h = seed_wh
        else:
            w, h = self.current_image.size
        grain_seed_base = self._source_image_generation * (1 << 20) + (w << 10) + (h & 0x3FF)
        opts = {
            'niveaux': 0,
            'do_contour': False,
            'contour_methode': DEFAULT_CONTOUR_METHODE,
            'seuil_flou': DEFAULT_SEUIL_FLOU,
            'floors': floors,
            'snap_forerunner_floors': True,
            'force_forerunner_snap': False,
            'process_floor_indices': process_subset,
            'fusion_mode': DEFAULT_FUSION_MODE,
            'use_inok': False,
            'allow_conversion': True,
            'tm': self.current_tm_path,
            'tm_fit_mode': self.tm_fit_mode_var.get(),
            'snap_only': False,
            'snap_only_skip_contour': self.vars['snap_only_skip_contour'].get(),
            '_grain_seed_base': grain_seed_base,
            '_floor_layer_cache': self._floor_layer_cache,
            'captured_surface_masks': self.pipeline_capture_details.get(),
        }
        return opts
    
    def _compute_final_image(self, source: Image.Image, progress_cb=None) -> Image.Image:
        """Exécute le pipeline complet sur une image (export pleine résolution)."""
        opts = self._build_options(seed_wh=source.size)
        TB, TB_q_pre, TB_q_snap, infer = build_forerunner_snap_cache_state(source, opts, progress_cb)
        results = process_forerunner_surfaces_only(
            TB, TB_q_pre, TB_q_snap, infer, opts, progress_cb
        )
        return results['final']
    
    def _process_image(self):
        """Traite l'image avec les paramètres actuels."""
        if self.current_image is None:
            return
        
        if self.is_processing:
            self._preview_pending = True
            return
        
        work_image = self._image_for_preview_pipeline()
        opts_pre = self._build_options(seed_wh=work_image.size)
        cache_key_pre = (
            self._source_image_generation,
            forerunner_presnap_cache_key(opts_pre),
            work_image.size,
        )
        gk_pre = surface_pipeline_global_key(opts_pre)
        fp_pre = tuple(floor_surface_fingerprint(opts_pre, k) for k in range(1, 6))
        snap_hit_pre = (
            cache_key_pre == self._snap_cache_key
            and self._snap_cache_tb is not None
            and self._snap_cache_tb_q_presnap is not None
            and self._snap_cache_tb_q_snapped is not None
        )
        if snap_hit_pre and self._surface_global_key == gk_pre:
            jc_pre = first_changed_floor_index(self._floor_fingerprints, fp_pre)
        else:
            jc_pre = 1
        if snap_hit_pre and self._surface_global_key == gk_pre and jc_pre is None:
            self.status_var.set("Prêt")
            self._pipeline_visual_on_first_base_load = False
            return
        
        self.is_processing = True
        preview_label = (
            "Traitement en cours (aperçu rapide)…"
            if self.fast_preview_var.get()
            else "Traitement en cours…"
        )
        self.status_var.set(preview_label)
        self._last_progress_msg = preview_label
        self._start_status_busy()
        
        use_pipeline_visual = self._pipeline_visual_on_first_base_load
        self._pipeline_visual_on_first_base_load = False
        
        # Traiter dans un fil d'exécution séparé pour ne pas bloquer l'interface
        def process():
            def progress_cb(pct: int, msg: str) -> None:
                self.root.after(0, lambda p=pct, m=msg: self._on_worker_progress(p, m))

            def visual_cb(pil_img: Image.Image, label: str) -> None:
                img_copy = pil_img.copy()
                self.root.after(0, lambda i=img_copy, t=label: self._apply_preview_step(i, t))

            visual_cb_arg = visual_cb if use_pipeline_visual else None
            
            try:
                opts = self._build_options(seed_wh=work_image.size)
                cache_key = (
                    self._source_image_generation,
                    forerunner_presnap_cache_key(opts),
                    work_image.size,
                )
                gk = surface_pipeline_global_key(opts)
                fp_new = tuple(floor_surface_fingerprint(opts, k) for k in range(1, 6))
                snap_hit = (
                    cache_key == self._snap_cache_key
                    and self._snap_cache_tb is not None
                    and self._snap_cache_tb_q_presnap is not None
                    and self._snap_cache_tb_q_snapped is not None
                )
                if snap_hit and self._surface_global_key == gk:
                    jc = first_changed_floor_index(self._floor_fingerprints, fp_new)
                    if jc is not None and floor_removal_requires_full_surface_rebuild(
                        self._floor_fingerprints, fp_new
                    ):
                        jc = 1
                else:
                    jc = 1
                rebuild_snap = not snap_hit
                use_incremental = (
                    jc is not None
                    and jc >= 2
                    and snap_hit
                    and self._surface_global_key == gk
                    and self._floor_checkpoints_after is not None
                    and (jc - 1) in self._floor_checkpoints_after
                    and not self.pipeline_capture_details.get()
                )
                pf_kwargs: dict = {'save_checkpoints': True}
                if use_incremental:
                    pf_kwargs['k_start'] = jc
                    pf_kwargs['resume_state'] = copy_floor_surface_state(
                        self._floor_checkpoints_after[jc - 1]
                    )
                if (
                    cache_key == self._snap_cache_key
                    and self._snap_cache_tb is not None
                    and self._snap_cache_tb_q_presnap is not None
                    and self._snap_cache_tb_q_snapped is not None
                ):
                    results = process_forerunner_surfaces_only(
                        self._snap_cache_tb,
                        self._snap_cache_tb_q_presnap,
                        self._snap_cache_tb_q_snapped,
                        self._snap_cache_infer,
                        opts,
                        progress_cb,
                        visual_cb=visual_cb_arg,
                        **pf_kwargs,
                    )
                else:
                    TB, TB_q_pre, TB_q_snap, infer = build_forerunner_snap_cache_state(
                        work_image, opts, progress_cb, visual_cb=visual_cb_arg
                    )
                    self._snap_cache_key = cache_key
                    self._snap_cache_tb = TB
                    self._snap_cache_tb_q_presnap = TB_q_pre
                    self._snap_cache_tb_q_snapped = TB_q_snap
                    self._snap_cache_infer = infer
                    results = process_forerunner_surfaces_only(
                        TB, TB_q_pre, TB_q_snap, infer, opts, progress_cb,
                        k_start=1,
                        resume_state=None,
                        save_checkpoints=True,
                        visual_cb=visual_cb_arg,
                    )
                cp = results.get('_checkpoints')
                self.root.after(
                    0,
                    lambda r=results,
                    g=gk,
                    fp=fp_new,
                    rb=rebuild_snap,
                    c=cp,
                    o=opts: self._apply_process_success(r, g, fp, rb, c, o),
                )
            except Exception as e:
                self.root.after(0, lambda: messagebox.showerror("Erreur", f"Erreur de traitement:\n{e}"))
                self.root.after(0, lambda: self.status_var.set(f"Erreur: {e}"))
            finally:
                def finish():
                    self.preview_step_var.set("")
                    self._stop_status_busy()
                    self.is_processing = False
                    if self._preview_pending:
                        self._preview_pending = False
                        self._process_image()

                self.root.after(0, finish)
        
        threading.Thread(target=process, daemon=True).start()
    
    def _apply_process_success(
        self,
        results: dict,
        gk: tuple,
        fp_new: tuple,
        rebuild_snap: bool,
        cp: Optional[dict],
        opts: Optional[dict] = None,
    ) -> None:
        """Applique le résultat du traitement sur le fil d'interface (prévisualisation et caches)."""
        old_fp = self._floor_fingerprints
        self.preview_image = results['final']
        self._last_pipeline_opts = opts
        pipeline_cp = results.get('_checkpoints')
        if cp is not None:
            if rebuild_snap:
                self._floor_checkpoints_after = dict(cp)
                self._floor_layer_cache.clear()
            else:
                if self._floor_checkpoints_after is None:
                    self._floor_checkpoints_after = {}
                self._floor_checkpoints_after.update(cp)
                # Recalcul partiel depuis le 1er étage modifié : invalider les calques mis en cache
                # pour k et au-delà (ex. activation/désactivation « Traiter cet étage »).
                jc_inv = first_changed_floor_index(old_fp, fp_new)
                if jc_inv is not None:
                    for kk in list(self._floor_layer_cache.keys()):
                        if kk >= jc_inv:
                            del self._floor_layer_cache[kk]
            pipeline_cp = dict(self._floor_checkpoints_after or cp)
        self._last_pipeline_state = self._build_pipeline_state_payload(
            results,
            opts,
            checkpoints=pipeline_cp,
        )
        self._surface_global_key = gk
        self._floor_fingerprints = fp_new
        self._update_preview()
        if self.fast_preview_var.get():
            self.status_var.set(
                f"Traitement terminé (aperçu rapide, côté max {FAST_PREVIEW_MAX_EDGE}px)"
            )
        else:
            self.status_var.set("Traitement terminé")
    
    def _build_pipeline_state_payload(
        self,
        results: dict,
        opts: Optional[dict] = None,
        *,
        checkpoints: Optional[dict] = None,
    ) -> dict:
        tbqs = results.get('TB_q_snapped')
        if tbqs is None:
            tbqs = self._snap_cache_tb_q_snapped
        return {
            'TB': results.get('TB'),
            'TB_q_presnap': results.get('TB_q'),
            'TB_q_snapped': tbqs,
            'captured_masks': list(results.get('captured_surface_masks') or []),
            'INM': results.get('INM'),
            'INEXM': results.get('INEXM'),
            'final': results.get('final'),
            'EX': results.get('EX'),
            'EX_gl': results.get('EX_gl'),
            'EX_glgr': results.get('EX_glgr'),
            'EXOK': results.get('EXOK'),
            'IN_gl': results.get('IN_gl'),
            'IN_n': results.get('IN_n'),
            'INOK': results.get('INOK'),
            '_checkpoints': checkpoints if checkpoints is not None else results.get('_checkpoints'),
            '_pipeline_opts': opts,
        }
    
    def _open_pipeline_graph_qt_from_menu(self) -> None:
        """Ouvre la visionneuse NodeGraphQt avec le dernier résultat de traitement."""
        payload = self._last_pipeline_state
        if not payload or payload.get("final") is None:
            messagebox.showinfo(
                "Diagnostic pipeline",
                "Aucun résultat de traitement disponible.\n\n"
                "Lancez d’abord un traitement (aperçu). Pour inclure masques et étapes détaillés dans le graphe, "
                "cochez « Capturer masques et étapes (rendu plus lent) » puis relancez.",
            )
            return
        opts_use = payload.get("_pipeline_opts") or self._last_pipeline_opts or {}
        self._launch_pipeline_graph_qt(payload, opts_use)
    
    def _launch_pipeline_graph_qt(self, payload: dict, opts: Optional[dict]) -> None:
        """Lance la visionneuse NodeGraphQt dans un processus séparé (pickle temporaire)."""
        try:
            import NodeGraphQt  # noqa: F401
        except ImportError:
            messagebox.showinfo(
                "NodeGraphQt",
                "Dépendances optionnelles manquantes.\n\n"
                "Installez avec :\n"
                "  pip install -r requirements-optional-nodegraph.txt",
            )
            return
        project_root = Path(__file__).resolve().parent.parent
        fd, pkl_path = tempfile.mkstemp(suffix=".pkl")
        os.close(fd)
        try:
            with open(pkl_path, "wb") as f:
                pickle.dump({"payload": payload, "opts": opts or {}}, f)
        except Exception as e:
            messagebox.showerror("NodeGraphQt", f"Sérialisation impossible : {e}")
            try:
                os.remove(pkl_path)
            except OSError:
                pass
            return
        cwd = str(project_root)
        try:
            subprocess.Popen(
                [sys.executable, "-m", "ui.pipeline_graph_qt", pkl_path],
                cwd=cwd,
            )
        except OSError as e:
            try:
                os.remove(pkl_path)
            except OSError:
                pass
            messagebox.showerror("NodeGraphQt", f"Impossible de lancer la visionneuse : {e}")
    
    def _apply_preview_step(self, pil_img: Image.Image, step_text: str) -> None:
        """Met à jour la prévisualisation pendant le traitement (fil d'exécution principal)."""
        self.preview_image = pil_img
        self.preview_step_var.set(step_text)
        self._update_preview()
    
    def _update_preview(self):
        """Met à jour l'affichage de la prévisualisation."""
        if self.preview_image is None:
            return
        
        # Redimensionner l'image pour l'affichage
        canvas_width = self.preview_canvas.winfo_width()
        canvas_height = self.preview_canvas.winfo_height()
        
        if canvas_width <= 1 or canvas_height <= 1:
            # Canevas pas encore initialisé : report au prochain rafraîchissement
            self.root.after(100, self._update_preview)
            return
        
        # Calculer les dimensions avec zoom
        img_width, img_height = self.preview_image.size
        base_scale = min(canvas_width / img_width, canvas_height / img_height, 1.0)
        display_width = int(img_width * base_scale * self.zoom_level)
        display_height = int(img_height * base_scale * self.zoom_level)
        
        # Redimensionner l'image avec la méthode choisie
        display_image = self.preview_image.resize((display_width, display_height), self.preview_resampling)
        
        # Convertir en PhotoImage
        photo = ImageTk.PhotoImage(display_image)
        
        # Afficher sur le canevas
        self.preview_canvas.delete("all")
        
        # Position avec pan
        x = (canvas_width - display_width) // 2 + self.pan_x
        y = (canvas_height - display_height) // 2 + self.pan_y
        
        self.preview_canvas.create_image(x, y, anchor=tk.NW, image=photo)
        self.preview_canvas.image = photo  # Garder une référence
        
        # Mettre à jour la région de défilement
        self.preview_canvas.config(scrollregion=self.preview_canvas.bbox("all"))
        
        # Mettre à jour le label de zoom
        if hasattr(self, 'zoom_label'):
            self.zoom_label.config(text=f"{int(self.zoom_level * 100)}%")
    
    def _zoom_in(self):
        """Augmente le niveau de zoom."""
        self.zoom_level = min(self.zoom_level * 1.2, 5.0)  # Max 500%
        self._update_preview()
    
    def _zoom_out(self):
        """Diminue le niveau de zoom."""
        self.zoom_level = max(self.zoom_level / 1.2, 0.1)  # Min 10%
        self._update_preview()
    
    def _reset_zoom(self):
        """Réinitialise le zoom et la position."""
        self.zoom_level = 1.0
        self.pan_x = 0
        self.pan_y = 0
        self._update_preview()
    
    def _on_mousewheel(self, event):
        """Gère le zoom avec la molette de la souris."""
        if event.delta > 0 or event.num == 4:
            self._zoom_in()
        elif event.delta < 0 or event.num == 5:
            self._zoom_out()
    
    def _on_canvas_press(self, event):
        """Démarre le pan (déplacement) de l'image."""
        self.pan_start_x = event.x
        self.pan_start_y = event.y
        self.is_panning = True
        self.preview_canvas.config(cursor="hand2")
    
    def _on_canvas_drag(self, event):
        """Déplace l'image pendant le pan."""
        if self.is_panning:
            dx = event.x - self.pan_start_x
            dy = event.y - self.pan_start_y
            self.pan_x += dx
            self.pan_y += dy
            self.pan_start_x = event.x
            self.pan_start_y = event.y
            self._update_preview()
    
    def _on_canvas_release(self, event):
        """Termine le pan."""
        self.is_panning = False
        self.preview_canvas.config(cursor="")
    
    
    def _export_result(self):
        """Exporte le résultat final avec options."""
        if self.preview_image is None or self.current_image is None:
            messagebox.showwarning("Avertissement", "Aucun résultat à exporter")
            return
        
        # Créer une fenêtre de dialogue pour les options d'export
        export_window = tk.Toplevel(self.root)
        export_window.title("Options d'export")
        export_window.geometry("400x220")
        export_window.transient(self.root)
        export_window.grab_set()
        
        # Dimensions par défaut : résolution source (l'export est toujours calculé en pleine qualité)
        if self.current_image is not None:
            original_width = self.current_image.width
            original_height = self.current_image.height
        else:
            original_width = self.preview_image.width
            original_height = self.preview_image.height
        
        # Options de redimensionnement de l'image
        hint = ttk.Label(
            export_window,
            text="Le rendu est recalculé en pleine résolution depuis l'image source (peut prendre quelques secondes).",
            wraplength=380,
        )
        hint.pack(fill=tk.X, padx=10, pady=(8, 0))
        
        resize_frame = ttk.LabelFrame(export_window, text="Dimensions d'export", padding=10)
        resize_frame.pack(fill=tk.X, padx=10, pady=10)
        
        width_var = tk.StringVar(value=str(original_width))
        height_var = tk.StringVar(value=str(original_height))
        
        # Checkbox pour conserver l'échelle
        keep_ratio_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(resize_frame, text="Conserver l'échelle", 
                       variable=keep_ratio_var).pack(anchor=tk.W, pady=5)
        
        size_frame = ttk.Frame(resize_frame)
        size_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(size_frame, text="Largeur:").grid(row=0, column=0, sticky=tk.W, padx=5)
        width_entry = ttk.Entry(size_frame, textvariable=width_var, width=10)
        width_entry.grid(row=0, column=1, padx=5)
        
        ttk.Label(size_frame, text="Hauteur:").grid(row=0, column=2, sticky=tk.W, padx=5)
        height_entry = ttk.Entry(size_frame, textvariable=height_var, width=10)
        height_entry.grid(row=0, column=3, padx=5)
        
        # Fonction pour maintenir le ratio (avec flag pour éviter les boucles)
        updating_ratio = {'active': False}
        
        def update_height_from_width():
            if keep_ratio_var.get() and not updating_ratio['active']:
                try:
                    updating_ratio['active'] = True
                    new_width = int(width_var.get())
                    if new_width > 0:
                        ratio = original_height / original_width
                        new_height = int(new_width * ratio)
                        height_var.set(str(new_height))
                except ValueError:
                    pass
                finally:
                    updating_ratio['active'] = False
        
        def update_width_from_height():
            if keep_ratio_var.get() and not updating_ratio['active']:
                try:
                    updating_ratio['active'] = True
                    new_height = int(height_var.get())
                    if new_height > 0:
                        ratio = original_width / original_height
                        new_width = int(new_height * ratio)
                        width_var.set(str(new_width))
                except ValueError:
                    pass
                finally:
                    updating_ratio['active'] = False
        
        width_entry.bind('<KeyRelease>', lambda e: update_height_from_width())
        height_entry.bind('<KeyRelease>', lambda e: update_width_from_height())
        
        def do_export():
            """Effectue l'export avec les options choisies."""
            file_path = filedialog.asksaveasfilename(
                title="Exporter le résultat",
                defaultextension=".png",
                filetypes=[
                    ("PNG", "*.png"),
                    ("JPEG", "*.jpg"),
                    ("TIFF", "*.tiff"),
                    ("Tous les fichiers", "*.*")
                ]
            )
            
            if not file_path:
                export_window.destroy()
                return
            
            try:
                self.root.config(cursor="watch")
                self.root.update_idletasks()
                try:
                    export_image = self._compute_final_image(self.current_image.copy())
                finally:
                    self.root.config(cursor="")
                
                # S'assurer que l'image est dans un mode compatible (RGBA ou RGB)
                if export_image.mode not in ('RGB', 'RGBA', 'L', 'LA'):
                    export_image = export_image.convert('RGBA')
                
                # Obtenir les dimensions demandées
                try:
                    width = int(width_var.get()) if width_var.get() else original_width
                    height = int(height_var.get()) if height_var.get() else original_height
                except ValueError:
                    messagebox.showerror("Erreur", "Les dimensions doivent être des nombres entiers")
                    return
                
                # Déterminer si les dimensions ont changé
                dimensions_changed = (width != original_width) or (height != original_height)
                
                # Utiliser LANCZOS si les dimensions changent, NEAREST sinon
                if dimensions_changed:
                    resampling_method = Image.Resampling.LANCZOS
                else:
                    resampling_method = Image.Resampling.NEAREST
                
                # Redimensionner si nécessaire
                if dimensions_changed:
                    export_image = export_image.resize((width, height), resampling_method)
                
                # Sauvegarder avec les paramètres de qualité appropriés
                # Pour PNG, on peut spécifier la compression
                if file_path.lower().endswith('.png'):
                    export_image.save(file_path, optimize=False)
                elif file_path.lower().endswith(('.jpg', '.jpeg')):
                    # Pour JPEG, convertir en RGB si nécessaire et spécifier la qualité
                    if export_image.mode == 'RGBA':
                        # Créer un fond blanc pour les zones transparentes
                        background = Image.new('RGB', export_image.size, (255, 255, 255))
                        background.paste(export_image, mask=export_image.split()[3] if export_image.mode == 'RGBA' else None)
                        export_image = background
                    elif export_image.mode != 'RGB':
                        export_image = export_image.convert('RGB')
                    export_image.save(file_path, quality=95)
                else:
                    export_image.save(file_path)
                messagebox.showinfo("Succès", f"Image exportée vers:\n{file_path}")
                export_window.destroy()
            except Exception as e:
                messagebox.showerror("Erreur", f"Impossible d'exporter l'image:\n{e}")
        
        # Boutons
        button_frame = ttk.Frame(export_window)
        button_frame.pack(fill=tk.X, padx=10, pady=10)
        
        ttk.Button(button_frame, text="Annuler", command=export_window.destroy).pack(side=tk.RIGHT, padx=5)
        ttk.Button(button_frame, text="Exporter", command=do_export).pack(side=tk.RIGHT, padx=5)


def main():
    """Point d'entrée principal."""
    root = tk.Tk()
    app = TextureHaloGUI(root)
    root.mainloop()


if __name__ == '__main__':
    main()

