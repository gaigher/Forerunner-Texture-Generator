"""
Éditeur graphique pour dessiner manuellement des lignes de construction Forerunner.

Interface style Paint avec système d'étages intégré.
Permet de dessiner librement avec différents outils tout en respectant le système d'étages.
"""
import sys
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

from PIL import Image, ImageDraw, ImageTk
from typing import Optional, Tuple, FrozenSet, List
import math


# Outils avec enroulement (coord. brutes + continue_raw) — pas ligne / ellipses.
TOOLS_TILEABLE_GEOM: FrozenSet[str] = frozenset({
    "rectangle",
    "rectangle_filled",
})
TOOLS_TILEABLE_WRAP: FrozenSet[str] = frozenset({"pencil"}) | TOOLS_TILEABLE_GEOM

# Tous les outils « forme » (préview, bridage d’angles)
GEOM_TOOLS: FrozenSet[str] = frozenset({
    "line",
    "rectangle",
    "rectangle_filled",
    "ellipse",
    "ellipse_filled",
})

# Angles autorisés pour le bridage (degrés)
_ALLOWED_SNAP_ANGLES_DEG: Tuple[int, ...] = (0, 45, 90, 135, 180, 225, 270, 315)

# Système d'étages (0-5)
FLOOR_0 = 0      # Noir
FLOOR_1 = 51     # Gris foncé
FLOOR_2 = 102    # Gris moyen-foncé
FLOOR_3 = 153    # Gris moyen
FLOOR_4 = 204    # Gris clair
FLOOR_5 = 255    # Blanc (plus haut)

FLOOR_TO_GRAY = {
    0: FLOOR_0,
    1: FLOOR_1,
    2: FLOOR_2,
    3: FLOOR_3,
    4: FLOOR_4,
    5: FLOOR_5
}

FLOOR_NAMES = {
    0: "Noir (fond)",
    1: "Gris foncé",
    2: "Gris moyen-foncé",
    3: "Gris moyen",
    4: "Gris clair",
    5: "Blanc"
}

# Aligné sur effets.utilitaires (TM_*)
TM_FIT_CHOICES: List[Tuple[str, str]] = [
    ("anisotropic", "Étirement libre (cadre exact)"),
    ("uniform_cover", "Étirement proportionnel (remplit, rognage)"),
    ("tile", "Mosaïque (répétition sans déformation)"),
    ("tile_approx_stretch", "Mosaïque (tuiles au plus proche puis étirement)"),
]


class ConstructionEditor:
    """Éditeur graphique pour lignes de construction Forerunner."""
    
    def __init__(
        self,
        root: tk.Tk,
        modal: bool = False,
        initial_image: Optional[Image.Image] = None,
        reference_metal_size: Optional[Tuple[int, int]] = None,
        initial_tm_fit_mode: str = "anisotropic",
    ):
        self.root = root
        self.root.title("Éditeur de Lignes de Construction Forerunner")
        self.root.geometry("1200x800")
        self.modal = modal
        self.result_image: Optional[Image.Image] = None
        self.result_tm_fit_mode: Optional[str] = None
        self.reference_metal_size = reference_metal_size
        self._initial_tm_fit_mode = initial_tm_fit_mode if initial_tm_fit_mode in {
            k for k, _ in TM_FIT_CHOICES
        } else "anisotropic"
        
        # Dimensions : le dessin de construction (initial_image) prime toujours sur la taille du fichier TM ;
        # sans dessin, la toile reprend la taille native de la TM si elle est connue.
        self.width = 1024
        self.height = 512
        if initial_image is not None:
            self.width, self.height = initial_image.size
        elif reference_metal_size is not None:
            self.width, self.height = reference_metal_size
        self.current_floor = 0  # Étage actuel (0 = fond / plus bas, 5 = plus haut / blanc)
        self.current_tool = "pencil"  # Outil actuel
        self.brush_size = 5
        self.tileable = True  # Mode carrelable (enroulement des bords)
        self.angle_snap = False  # Mode bridage des angles (horizontal, vertical, diagonale)
        # Crayon : distance (px) parcourue depuis l’ancrage avant de pouvoir changer d’angle (0 = immédiat)
        self.angle_snap_lock_pixels = 8
        self._pencil_angle_lock_ax = 0.0
        self._pencil_angle_lock_ay = 0.0
        self._pencil_angle_lock_deg: Optional[int] = None
        self.erase_mode = False  # Mode gomme (tous les outils deviennent des gommes)
        self.image: Optional[Image.Image] = None
        self.draw: Optional[ImageDraw.Draw] = None
        self.photo: Optional[ImageTk.PhotoImage] = None
        
        # Variables pour le dessin
        self.start_x = 0
        self.start_y = 0
        self.last_x = 0
        self.last_y = 0
        self.drawing = False
        self.cursor_x = None  # Position actuelle du curseur (coordonnées du canevas)
        self.cursor_y = None
        self.state_saved_for_current_action = False  # Flag pour éviter de sauvegarder plusieurs fois la même action
        # Verrou curseur = pinceau : Maj détectée via event.state sur la souris (mode carrelable + crayon)
        self._is_warping_cursor = False
        
        # Zoom
        self.zoom_level = 1.0
        self.pan_x = 0
        self.pan_y = 0
        
        # Système annuler/refaire
        self.undo_stack = []  # Pile des états précédents
        self.redo_stack = []  # Pile des états à refaire
        self.max_undo_steps = 50  # Nombre maximum d'étapes à mémoriser
        
        self.setup_ui()
        if initial_image is not None:
            # Charger l'image initiale (dimensions déjà fixées ci-dessus)
            if initial_image.mode != 'L':
                self.image = initial_image.convert('L')
            else:
                self.image = initial_image.copy()
            self.draw = ImageDraw.Draw(self.image)
            self.undo_stack.clear()
            self.redo_stack.clear()
            self.update_undo_redo_menu_state()
            self._sync_dim_widgets()
            self._update_tm_fit_visibility()
            self.update_canvas()
            self.update_info()
            self.root.after_idle(self.fit_to_window)
        else:
            self.new_image()
        # Initialiser l'état des menus annuler/refaire
        if hasattr(self, 'edit_menu'):
            self.update_undo_redo_menu_state()
        
        # Si mode modal, modifier le comportement de fermeture
        if self.modal:
            self.root.protocol("WM_DELETE_WINDOW", self._on_modal_close)
    
    def setup_ui(self):
        """Configure l'interface utilisateur."""
        # Barre de menus
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)
        
        # Menu Fichier
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Fichier", menu=file_menu)
        file_menu.add_command(label="Nouveau...", command=self.new_image_dialog, accelerator="Ctrl+N")
        file_menu.add_command(label="Ouvrir...", command=self.open_image, accelerator="Ctrl+O")
        file_menu.add_command(label="Sauvegarder", command=self.save_image, accelerator="Ctrl+S")
        file_menu.add_command(label="Sauvegarder sous...", command=self.save_image_as, accelerator="Ctrl+Shift+S")
        file_menu.add_separator()
        if self.modal:
            file_menu.add_command(label="Valider et fermer", command=self._validate_and_close, accelerator="Ctrl+Enter")
            file_menu.add_separator()
        file_menu.add_command(label="Quitter", command=self.root.quit)
        
        # Menu Édition
        self.edit_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Édition", menu=self.edit_menu)
        self.edit_menu.add_command(label="Annuler", command=self.undo, accelerator="Ctrl+Z", state=tk.DISABLED)
        self.edit_menu.add_command(label="Refaire", command=self.redo, accelerator="Ctrl+Y", state=tk.DISABLED)
        self.edit_menu.add_separator()
        self.edit_menu.add_command(label="Effacer tout", command=self.clear_image)
        
        # Menu Affichage
        view_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Affichage", menu=view_menu)
        view_menu.add_command(label="Zoom +", command=lambda: self.set_zoom(self.zoom_level * 1.2), accelerator="Ctrl++")
        view_menu.add_command(label="Zoom -", command=lambda: self.set_zoom(self.zoom_level / 1.2), accelerator="Ctrl+-")
        view_menu.add_command(label="Zoom 100%", command=lambda: self.set_zoom(1.0), accelerator="Ctrl+0")
        view_menu.add_command(label="Ajuster à la fenêtre", command=self.fit_to_window)
        
        # Raccourcis clavier
        self.root.bind("<Control-n>", lambda e: self.new_image_dialog())
        self.root.bind("<Control-o>", lambda e: self.open_image())
        self.root.bind("<Control-s>", lambda e: self.save_image())
        self.root.bind("<Control-S>", lambda e: self.save_image_as())
        self.root.bind("<Control-plus>", lambda e: self.set_zoom(self.zoom_level * 1.2))
        self.root.bind("<Control-minus>", lambda e: self.set_zoom(self.zoom_level / 1.2))
        self.root.bind("<Control-0>", lambda e: self.set_zoom(1.0))
        self.root.bind("<Control-z>", lambda e: self.undo())
        self.root.bind("<Control-y>", lambda e: self.redo())
        # Raccourci alternatif pour refaire (Ctrl+Shift+Z sur certains systèmes)
        self.root.bind("<Control-Z>", lambda e: self.redo())
        # Raccourci pour valider en mode modal
        if self.modal:
            self.root.bind("<Control-Return>", lambda e: self._validate_and_close())
            self.root.bind("<Control-KP_Enter>", lambda e: self._validate_and_close())
        
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
        
        # Panneau gauche (outils et options)
        left_panel = ttk.Frame(main_panel, width=250)
        main_panel.add(left_panel, minsize=180, stretch="never")
        
        # Panneau droit (canevas)
        right_panel = ttk.Frame(main_panel)
        main_panel.add(right_panel, stretch="always")
        
        self.setup_tool_panel(left_panel)
        self.setup_canvas(right_panel)
    
    def setup_tool_panel(self, parent: ttk.Frame):
        """Configure le panneau d'outils."""
        # Titre (fixe en haut)
        title_label = ttk.Label(parent, text="Outils de Dessin", font=("Arial", 12, "bold"))
        title_label.pack(pady=10)
        
        scroll_outer = ttk.Frame(parent)
        scroll_outer.pack(fill=tk.BOTH, expand=True)
        tool_canvas = tk.Canvas(scroll_outer, highlightthickness=0)
        tool_scroll = ttk.Scrollbar(scroll_outer, orient=tk.VERTICAL, command=tool_canvas.yview)
        tool_canvas.configure(yscrollcommand=tool_scroll.set)
        tool_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        tool_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollable = ttk.Frame(tool_canvas)
        _tool_win = tool_canvas.create_window((0, 0), window=scrollable, anchor=tk.NW)

        def _tool_on_frame_configure(_event=None):
            tool_canvas.configure(scrollregion=tool_canvas.bbox("all"))

        def _tool_on_canvas_configure(event):
            tool_canvas.itemconfigure(_tool_win, width=event.width)

        scrollable.bind("<Configure>", _tool_on_frame_configure)
        tool_canvas.bind("<Configure>", _tool_on_canvas_configure)

        def _tool_on_mousewheel(event):
            if getattr(event, "delta", 0):
                tool_canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
            return "break"

        def _tool_on_linux_up(_event):
            tool_canvas.yview_scroll(-3, "units")
            return "break"

        def _tool_on_linux_down(_event):
            tool_canvas.yview_scroll(3, "units")
            return "break"

        def _bind_tool_wheel_rec(widget):
            widget.bind("<MouseWheel>", _tool_on_mousewheel)
            widget.bind("<Button-4>", _tool_on_linux_up)
            widget.bind("<Button-5>", _tool_on_linux_down)
            for ch in widget.winfo_children():
                _bind_tool_wheel_rec(ch)

        parent = scrollable  # le reste du panneau est dans la zone défilante
        
        # Dimensions de la toile (avant le sélecteur d'étage)
        dim_frame = ttk.LabelFrame(parent, text="Dimensions de la toile", padding=10)
        dim_frame.pack(fill=tk.X, padx=5, pady=5)
        dim_row = ttk.Frame(dim_frame)
        dim_row.pack(fill=tk.X)
        ttk.Label(dim_row, text="Largeur:").pack(side=tk.LEFT)
        self.dim_w_var = tk.IntVar(value=max(1, self.width))
        self.dim_h_var = tk.IntVar(value=max(1, self.height))
        self.dim_w_spin = ttk.Spinbox(dim_row, from_=1, to=16384, width=7, textvariable=self.dim_w_var)
        self.dim_w_spin.pack(side=tk.LEFT, padx=(4, 12))
        ttk.Label(dim_row, text="Hauteur:").pack(side=tk.LEFT)
        self.dim_h_spin = ttk.Spinbox(dim_row, from_=1, to=16384, width=7, textvariable=self.dim_h_var)
        self.dim_h_spin.pack(side=tk.LEFT, padx=4)
        ttk.Button(dim_row, text="Appliquer", command=self.apply_canvas_dimensions).pack(side=tk.LEFT, padx=(8, 0))
        
        self.tm_fit_mode_var = tk.StringVar(value=self._initial_tm_fit_mode)
        self.tm_fit_frame: Optional[ttk.LabelFrame] = None
        self.tm_fit_combo: Optional[ttk.Combobox] = None
        self.tm_fit_hint: Optional[ttk.Label] = None
        if self.reference_metal_size is not None:
            self.tm_fit_frame = ttk.LabelFrame(parent, text="Texture métal sur le pipeline", padding=10)
            mw, mh = self.reference_metal_size
            self.tm_fit_hint = ttk.Label(
                self.tm_fit_frame,
                text=f"Taille du fichier TM : {mw}×{mh}px — utilisée si la toile ≠ TM.",
                font=("TkDefaultFont", 8),
                wraplength=220,
            )
            self.tm_fit_hint.pack(anchor=tk.W, pady=(0, 4))
            labels = [lab for _k, lab in TM_FIT_CHOICES]
            keys = [k for k, _lab in TM_FIT_CHOICES]
            self._tm_fit_labels_to_key = {lab: k for lab, k in zip(labels, keys)}
            self._tm_fit_key_to_label = {k: lab for k, lab in TM_FIT_CHOICES}
            self.tm_fit_combo = ttk.Combobox(
                self.tm_fit_frame,
                values=labels,
                state="readonly",
                width=42,
            )
            cur_label = self._tm_fit_key_to_label.get(self.tm_fit_mode_var.get(), TM_FIT_CHOICES[0][1])
            self.tm_fit_combo.set(cur_label)
            self.tm_fit_combo.pack(fill=tk.X, pady=2)
            self.tm_fit_combo.bind("<<ComboboxSelected>>", self._on_tm_fit_combo)
        
        # Sélecteur d'étage
        floor_frame = ttk.LabelFrame(parent, text="Niveau d'Étage", padding=10)
        self._floor_frame_widget = floor_frame
        floor_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.floor_var = tk.IntVar(value=self.current_floor)
        # Affichage du plus haut (5) au plus bas (0), comme les étages d'un bâtiment
        for floor in range(5, -1, -1):
            floor_btn = ttk.Radiobutton(
                floor_frame,
                text=f"Étage {floor}: {FLOOR_NAMES[floor]} ({FLOOR_TO_GRAY[floor]})",
                variable=self.floor_var,
                value=floor,
                command=self.on_floor_changed
            )
            floor_btn.pack(anchor=tk.W, pady=2)
        
        # Aperçu de la couleur
        self.color_preview = tk.Canvas(floor_frame, width=200, height=30, bg="white", relief=tk.SUNKEN)
        self.color_preview.pack(pady=5)
        self.update_color_preview()
        
        # Outils de dessin
        tool_frame = ttk.LabelFrame(parent, text="Outils", padding=10)
        tool_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.tool_var = tk.StringVar(value="pencil")
        tools = [
            ("Crayon", "pencil"),
            ("Ligne", "line"),
            ("Rectangle", "rectangle"),
            ("Rectangle plein", "rectangle_filled"),
            ("Ellipse", "ellipse"),
            ("Ellipse pleine", "ellipse_filled"),
            ("Remplissage", "fill")
        ]
        
        for text, value in tools:
            ttk.Radiobutton(
                tool_frame,
                text=text,
                variable=self.tool_var,
                value=value,
                command=self.on_tool_changed
            ).pack(anchor=tk.W, pady=2)
        
        # Taille du pinceau
        brush_frame = ttk.LabelFrame(parent, text="Taille du Pinceau", padding=10)
        brush_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.brush_size_var = tk.IntVar(value=self.brush_size)
        brush_scale = ttk.Scale(
            brush_frame,
            from_=1,
            to=50,
            variable=self.brush_size_var,
            orient=tk.HORIZONTAL,
            command=self.on_brush_size_changed
        )
        brush_scale.pack(fill=tk.X, pady=5)
        
        self.brush_size_label = ttk.Label(brush_frame, text=f"Taille: {self.brush_size}px")
        self.brush_size_label.pack()
        
        # Mode carrelable
        tileable_frame = ttk.LabelFrame(parent, text="Options", padding=10)
        tileable_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.tileable_var = tk.BooleanVar(value=self.tileable)
        self.tileable_check = ttk.Checkbutton(
            tileable_frame,
            text="Mode carrelable (enroulement des bords)",
            variable=self.tileable_var,
            command=self.on_tileable_changed
        )
        self.tileable_check.pack(anchor=tk.W, pady=2)
        
        ttk.Label(
            tileable_frame,
            text="Maintenir Maj : curseur lié au pinceau (crayon, mode carrelable)",
            font=("TkDefaultFont", 8),
            foreground="gray"
        ).pack(anchor=tk.W, pady=(0, 4))
        
        self.angle_snap_var = tk.BooleanVar(value=self.angle_snap)
        angle_snap_check = ttk.Checkbutton(
            tileable_frame,
            text="Bridage des angles (H/V/Diagonale)",
            variable=self.angle_snap_var,
            command=self.on_angle_snap_changed
        )
        angle_snap_check.pack(anchor=tk.W, pady=2)
        
        self.angle_snap_lock_row = ttk.Frame(tileable_frame)
        self.angle_snap_lock_row.pack(anchor=tk.W, fill=tk.X, pady=(0, 2))
        ttk.Label(
            self.angle_snap_lock_row,
            text="  Seuil avant changement d’angle (px, 0 = immédiat):",
            font=("TkDefaultFont", 8),
        ).pack(side=tk.LEFT)
        self.angle_snap_lock_var = tk.IntVar(value=self.angle_snap_lock_pixels)
        self.angle_snap_lock_spin = ttk.Spinbox(
            self.angle_snap_lock_row,
            from_=0,
            to=200,
            width=5,
            textvariable=self.angle_snap_lock_var,
            command=self.on_angle_snap_lock_pixels_changed,
        )
        self.angle_snap_lock_spin.pack(side=tk.LEFT, padx=4)
        self.angle_snap_lock_spin.bind("<FocusOut>", lambda _e: self.on_angle_snap_lock_pixels_changed())
        self.angle_snap_lock_spin.bind("<Return>", lambda _e: self.on_angle_snap_lock_pixels_changed())
        self._sync_angle_snap_lock_widgets()
        
        self.erase_mode_var = tk.BooleanVar(value=self.erase_mode)
        erase_mode_check = ttk.Checkbutton(
            tileable_frame,
            text="Mode gomme",
            variable=self.erase_mode_var,
            command=self.on_erase_mode_changed
        )
        erase_mode_check.pack(anchor=tk.W, pady=2)
        
        # Boutons de nettoyage
        cleanup_frame = ttk.LabelFrame(parent, text="Nettoyage", padding=10)
        cleanup_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Button(
            cleanup_frame,
            text="Nettoyer tous les étages",
            command=self.clear_all_floors
        ).pack(fill=tk.X, pady=2)
        
        # Informations
        info_frame = ttk.LabelFrame(parent, text="Informations", padding=10)
        info_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.info_label = ttk.Label(info_frame, text="", justify=tk.LEFT)
        self.info_label.pack(anchor=tk.W)
        self.update_info()

        tool_canvas.bind("<MouseWheel>", _tool_on_mousewheel)
        tool_canvas.bind("<Button-4>", _tool_on_linux_up)
        tool_canvas.bind("<Button-5>", _tool_on_linux_down)
        _bind_tool_wheel_rec(scrollable)
    
    def setup_canvas(self, parent: ttk.Frame):
        """Configure le canevas de dessin."""
        # Barre d'outils du canevas
        self.canvas_toolbar = ttk.Frame(parent)
        self.canvas_toolbar.pack(fill=tk.X, pady=5)
        canvas_toolbar = self.canvas_toolbar
        
        # Boutons Annuler/Refaire avec icônes
        self.undo_btn = ttk.Button(
            canvas_toolbar, 
            text="◀", 
            command=self.undo,
            width=3,
            state=tk.DISABLED
        )
        self.undo_btn.pack(side=tk.LEFT, padx=2)
        
        self.redo_btn = ttk.Button(
            canvas_toolbar, 
            text="▶", 
            command=self.redo,
            width=3,
            state=tk.DISABLED
        )
        self.redo_btn.pack(side=tk.LEFT, padx=2)
        
        # Séparateur
        ttk.Separator(canvas_toolbar, orient=tk.VERTICAL).pack(side=tk.LEFT, padx=5, fill=tk.Y)
        
        ttk.Label(canvas_toolbar, text="Zoom:").pack(side=tk.LEFT, padx=5)
        self.zoom_label = ttk.Label(canvas_toolbar, text="100%")
        self.zoom_label.pack(side=tk.LEFT, padx=5)
        
        ttk.Button(canvas_toolbar, text="Zoom +", command=lambda: self.set_zoom(self.zoom_level * 1.2)).pack(side=tk.LEFT, padx=2)
        ttk.Button(canvas_toolbar, text="Zoom -", command=lambda: self.set_zoom(self.zoom_level / 1.2)).pack(side=tk.LEFT, padx=2)
        ttk.Button(canvas_toolbar, text="100%", command=lambda: self.set_zoom(1.0)).pack(side=tk.LEFT, padx=2)
        ttk.Button(canvas_toolbar, text="Ajuster", command=self.fit_to_window).pack(side=tk.LEFT, padx=2)
        
        # Ajouter les boutons modaux si nécessaire
        if self.modal:
            ttk.Separator(canvas_toolbar, orient=tk.VERTICAL).pack(side=tk.LEFT, padx=5, fill=tk.Y)
            validate_btn = ttk.Button(canvas_toolbar, text="Valider", command=self._validate_and_close)
            validate_btn.pack(side=tk.LEFT, padx=2)
            cancel_btn = ttk.Button(canvas_toolbar, text="Annuler", command=self._cancel_modal)
            cancel_btn.pack(side=tk.LEFT, padx=2)
        
        # Canevas avec barres de défilement
        canvas_frame = ttk.Frame(parent)
        canvas_frame.pack(fill=tk.BOTH, expand=True)
        
        self.canvas = tk.Canvas(
            canvas_frame,
            bg="gray",
            cursor="crosshair"
        )
        
        v_scrollbar = ttk.Scrollbar(canvas_frame, orient=tk.VERTICAL, command=self.canvas.yview)
        h_scrollbar = ttk.Scrollbar(canvas_frame, orient=tk.HORIZONTAL, command=self.canvas.xview)
        
        self.canvas.configure(yscrollcommand=v_scrollbar.set, xscrollcommand=h_scrollbar.set)
        
        self.canvas.grid(row=0, column=0, sticky="nsew")
        v_scrollbar.grid(row=0, column=1, sticky="ns")
        h_scrollbar.grid(row=1, column=0, sticky="ew")
        
        canvas_frame.grid_rowconfigure(0, weight=1)
        canvas_frame.grid_columnconfigure(0, weight=1)
        
        # Lier les événements au canevas
        self.canvas.bind("<Button-1>", self.on_mouse_down)
        self.canvas.bind("<B1-Motion>", self.on_mouse_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_mouse_up)
        self.canvas.bind("<Motion>", self.on_mouse_move)
        self.canvas.bind("<Leave>", self.on_mouse_leave)  # Quand la souris quitte le canevas
        self.canvas.bind("<MouseWheel>", self.on_mouse_wheel)
        self.canvas.bind("<Button-4>", self.on_mouse_wheel)  # Linux
        self.canvas.bind("<Button-5>", self.on_mouse_wheel)  # Linux
    
    def new_image(self, width: int = None, height: int = None):
        """Crée une nouvelle image."""
        if width is None:
            width = self.width
        if height is None:
            height = self.height
        
        self.width = width
        self.height = height
        self.image = Image.new('L', (width, height), FLOOR_5)  # Toile initiale en blanc (étage 5)
        self.draw = ImageDraw.Draw(self.image)
        # Réinitialiser les piles annuler/refaire
        self.undo_stack.clear()
        self.redo_stack.clear()
        self.update_undo_redo_menu_state()
        self._sync_dim_widgets()
        self._update_tm_fit_visibility()
        self.update_canvas()
        self.update_info()
        # Ajuster automatiquement à la fenêtre après que la fenêtre soit affichée
        self.root.after_idle(self.fit_to_window)
    
    def _sync_dim_widgets(self) -> None:
        """Synchronise les champs largeur/hauteur avec self.width / self.height."""
        if not hasattr(self, "dim_w_var"):
            return
        self.dim_w_var.set(max(1, int(self.width)))
        self.dim_h_var.set(max(1, int(self.height)))
    
    def _on_tm_fit_combo(self, _event=None) -> None:
        if self.tm_fit_combo is None:
            return
        label = self.tm_fit_combo.get()
        key = getattr(self, "_tm_fit_labels_to_key", {}).get(label)
        if key:
            self.tm_fit_mode_var.set(key)
    
    def _update_tm_fit_visibility(self) -> None:
        if self.tm_fit_frame is None or self.reference_metal_size is None:
            return
        mw, mh = self.reference_metal_size
        need = self.width != mw or self.height != mh
        if need:
            self.tm_fit_frame.pack(fill=tk.X, padx=5, pady=5, before=self._floor_frame_widget)
        else:
            self.tm_fit_frame.pack_forget()
    
    def apply_canvas_dimensions(self) -> None:
        """Redimensionne la toile (rééchantillonnage du dessin en niveaux de gris)."""
        try:
            w = int(self.dim_w_var.get())
            h = int(self.dim_h_var.get())
        except (tk.TclError, ValueError):
            messagebox.showerror("Erreur", "Largeur et hauteur doivent être des entiers valides.")
            return
        if w < 1 or h < 1 or w > 16384 or h > 16384:
            messagebox.showerror("Erreur", "Dimensions hors plage (1–16384 px).")
            return
        if w == self.width and h == self.height:
            self._update_tm_fit_visibility()
            return
        if self.image is None:
            self.new_image(w, h)
            return
        self.save_state()
        self.width, self.height = w, h
        self.image = self.image.resize((w, h), Image.Resampling.LANCZOS)
        self.draw = ImageDraw.Draw(self.image)
        self._sync_dim_widgets()
        self._update_tm_fit_visibility()
        self.update_canvas()
        self.update_info()
        self.root.after_idle(self.fit_to_window)
    
    def new_image_dialog(self):
        """Ouvre une boîte de dialogue pour créer une nouvelle image."""
        dialog = tk.Toplevel(self.root)
        dialog.title("Nouvelle Image")
        dialog.geometry("300x150")
        dialog.transient(self.root)
        dialog.grab_set()
        
        ttk.Label(dialog, text="Largeur:").grid(row=0, column=0, padx=10, pady=10, sticky=tk.W)
        width_var = tk.StringVar(value=str(self.width))
        width_entry = ttk.Entry(dialog, textvariable=width_var, width=10)
        width_entry.grid(row=0, column=1, padx=10, pady=10)
        
        ttk.Label(dialog, text="Hauteur:").grid(row=1, column=0, padx=10, pady=10, sticky=tk.W)
        height_var = tk.StringVar(value=str(self.height))
        height_entry = ttk.Entry(dialog, textvariable=height_var, width=10)
        height_entry.grid(row=1, column=1, padx=10, pady=10)
        
        def create():
            try:
                w = int(width_var.get())
                h = int(height_var.get())
                if w > 0 and h > 0:
                    self.new_image(w, h)
                    dialog.destroy()
                else:
                    messagebox.showerror("Erreur", "Les dimensions doivent être positives.")
            except ValueError:
                messagebox.showerror("Erreur", "Veuillez entrer des nombres valides.")
        
        ttk.Button(dialog, text="Créer", command=create).grid(row=2, column=0, columnspan=2, pady=20)
        width_entry.focus()
        width_entry.select_range(0, tk.END)
    
    def open_image(self):
        """Ouvre une image existante."""
        filename = filedialog.askopenfilename(
            title="Ouvrir une image",
            filetypes=[
                ("Images", "*.png *.jpg *.jpeg *.bmp *.tiff *.tif"),
                ("PNG", "*.png"),
                ("Tous les fichiers", "*.*")
            ]
        )
        
        if filename:
            try:
                img = Image.open(filename)
                # Convertir en niveaux de gris si nécessaire
                if img.mode != 'L':
                    img = img.convert('L')
                
                self.width, self.height = img.size
                self.image = img
                self.draw = ImageDraw.Draw(self.image)
                # Réinitialiser les piles annuler/refaire
                self.undo_stack.clear()
                self.redo_stack.clear()
                self.update_undo_redo_menu_state()
                self._sync_dim_widgets()
                self._update_tm_fit_visibility()
                self.update_canvas()
                self.update_info()
                # Ajuster automatiquement à la fenêtre après que la fenêtre soit affichée
                self.root.after_idle(self.fit_to_window)
            except Exception as e:
                messagebox.showerror("Erreur", f"Impossible d'ouvrir l'image:\n{e}")
    
    def save_image(self):
        """Sauvegarde l'image."""
        if not hasattr(self, 'current_file') or self.current_file is None:
            self.save_image_as()
        else:
            try:
                self.image.convert('RGB').save(self.current_file)
                messagebox.showinfo("Succès", f"Image sauvegardée: {self.current_file}")
            except Exception as e:
                messagebox.showerror("Erreur", f"Impossible de sauvegarder:\n{e}")
    
    def save_image_as(self):
        """Sauvegarde l'image sous un nouveau nom."""
        filename = filedialog.asksaveasfilename(
            title="Sauvegarder l'image",
            defaultextension=".png",
            filetypes=[
                ("PNG", "*.png"),
                ("JPEG", "*.jpg *.jpeg"),
                ("BMP", "*.bmp"),
                ("TIFF", "*.tiff *.tif"),
                ("Tous les fichiers", "*.*")
            ]
        )
        
        if filename:
            try:
                self.current_file = filename
                self.image.convert('RGB').save(filename)
                messagebox.showinfo("Succès", f"Image sauvegardée: {filename}")
            except Exception as e:
                messagebox.showerror("Erreur", f"Impossible de sauvegarder:\n{e}")
    
    def _validate_and_close(self):
        """Valide l'image et ferme la fenêtre en mode modal."""
        if self.image is not None:
            self.result_image = self.image.copy()
        if hasattr(self, "tm_fit_mode_var"):
            self.result_tm_fit_mode = self.tm_fit_mode_var.get()
        self.root.destroy()
    
    def _cancel_modal(self):
        """Annule et ferme la fenêtre en mode modal."""
        self.result_image = None
        self.result_tm_fit_mode = None
        self.root.destroy()
    
    def _on_modal_close(self):
        """Gère la fermeture de la fenêtre en mode modal."""
        # En mode modal, demander confirmation ou utiliser l'image actuelle
        if self.image is not None:
            self.result_image = self.image.copy()
        if hasattr(self, "tm_fit_mode_var"):
            self.result_tm_fit_mode = self.tm_fit_mode_var.get()
        self.root.destroy()
    
    def clear_image(self):
        """Efface toute l'image (remet tout au blanc, étage 5)."""
        if messagebox.askyesno("Confirmer", "Voulez-vous vraiment effacer toute l'image ?"):
            self.save_state()  # Sauvegarder l'état avant d'effacer
            self.image = Image.new('L', (self.width, self.height), FLOOR_5)
            self.draw = ImageDraw.Draw(self.image)
            self.update_canvas()
    
    def save_state(self):
        """Sauvegarde l'état actuel de l'image pour annuler/refaire."""
        if self.image is None:
            return
        
        # Sauvegarder une copie de l'image
        state = self.image.copy()
        self.undo_stack.append(state)
        
        # Limiter la taille de la pile
        if len(self.undo_stack) > self.max_undo_steps:
            self.undo_stack.pop(0)
        
        # Vider la pile « refaire » quand on fait une nouvelle action
        self.redo_stack.clear()
        
        # Mettre à jour l'état des menus
        self.update_undo_redo_menu_state()
    
    def undo(self):
        """Annule la dernière action."""
        if not self.undo_stack:
            return
        
        # Sauvegarder l'état actuel dans « refaire » avant d'annuler
        if self.image is not None:
            self.redo_stack.append(self.image.copy())
            # Limiter la taille de la pile « refaire »
            if len(self.redo_stack) > self.max_undo_steps:
                self.redo_stack.pop(0)
        
        # Restaurer l'état précédent
        previous_state = self.undo_stack.pop()
        self.image = previous_state.copy()
        self.draw = ImageDraw.Draw(self.image)
        self.update_canvas()
        
        # Mettre à jour l'état des menus
        self.update_undo_redo_menu_state()
    
    def redo(self):
        """Refait la dernière action annulée."""
        if not self.redo_stack:
            return
        
        # Sauvegarder l'état actuel dans « annuler » avant de refaire
        if self.image is not None:
            self.undo_stack.append(self.image.copy())
            # Limiter la taille de la pile « annuler »
            if len(self.undo_stack) > self.max_undo_steps:
                self.undo_stack.pop(0)
        
        # Restaurer l'état suivant
        next_state = self.redo_stack.pop()
        self.image = next_state.copy()
        self.draw = ImageDraw.Draw(self.image)
        self.update_canvas()
        
        # Mettre à jour l'état des menus
        self.update_undo_redo_menu_state()
    
    def update_undo_redo_menu_state(self):
        """Met à jour l'état (activé/désactivé) des menus Annuler et Refaire."""
        if not hasattr(self, 'edit_menu'):
            return
        
        # Mettre à jour l'état du menu Annuler (index 0)
        if len(self.undo_stack) > 0:
            self.edit_menu.entryconfig(0, state=tk.NORMAL)
        else:
            self.edit_menu.entryconfig(0, state=tk.DISABLED)
        
        # Mettre à jour l'état du menu Refaire (index 1)
        if len(self.redo_stack) > 0:
            self.edit_menu.entryconfig(1, state=tk.NORMAL)
        else:
            self.edit_menu.entryconfig(1, state=tk.DISABLED)
        
        # Mettre à jour l'état des boutons dans la barre d'outils
        if hasattr(self, 'undo_btn'):
            if len(self.undo_stack) > 0:
                self.undo_btn.config(state=tk.NORMAL)
            else:
                self.undo_btn.config(state=tk.DISABLED)
        
        if hasattr(self, 'redo_btn'):
            if len(self.redo_stack) > 0:
                self.redo_btn.config(state=tk.NORMAL)
            else:
                self.redo_btn.config(state=tk.DISABLED)
    
    def on_floor_changed(self):
        """Appelé quand l'étage change."""
        self.current_floor = self.floor_var.get()
        self.update_color_preview()
    
    def on_tool_changed(self):
        """Appelé quand l'outil change."""
        self.current_tool = self.tool_var.get()
        
        # Changer le curseur selon l'outil (sauf si mode gomme)
        if self.erase_mode:
            self.canvas.config(cursor="circle")
        elif self.current_tool == "fill":
            self.canvas.config(cursor="spraycan")
        else:
            self.canvas.config(cursor="crosshair")
    
    def on_brush_size_changed(self, value):
        """Appelé quand la taille du pinceau change."""
        self.brush_size = int(float(value))
        self.brush_size_label.config(text=f"Taille: {self.brush_size}px")
    
    def on_tileable_changed(self):
        """Appelé quand le mode carrelable change."""
        self.tileable = self.tileable_var.get()
        self.update_info()
    
    def on_angle_snap_changed(self):
        """Appelé quand le mode bridage des angles change."""
        self.angle_snap = self.angle_snap_var.get()
        self._sync_angle_snap_lock_widgets()
        self.update_info()
    
    def on_angle_snap_lock_pixels_changed(self, *_args) -> None:
        """Met à jour le seuil (px) avant autorisation de changement d’angle (crayon)."""
        try:
            v = int(self.angle_snap_lock_var.get())
        except (tk.TclError, ValueError):
            return
        v = max(0, min(200, v))
        self.angle_snap_lock_pixels = v
        if self.angle_snap_lock_var.get() != v:
            self.angle_snap_lock_var.set(v)
        self.update_info()
    
    def _sync_angle_snap_lock_widgets(self) -> None:
        state = tk.NORMAL if self.angle_snap else tk.DISABLED
        try:
            self.angle_snap_lock_spin.configure(state=state)
        except tk.TclError:
            pass
    
    def on_erase_mode_changed(self):
        """Appelé quand le mode gomme change."""
        self.erase_mode = self.erase_mode_var.get()
        # Changer le curseur selon le mode
        if self.erase_mode:
            self.canvas.config(cursor="circle")
        else:
            # Restaurer le curseur selon l'outil
            if self.current_tool == "fill":
                self.canvas.config(cursor="spraycan")
            else:
                self.canvas.config(cursor="crosshair")
        self.update_info()
    
    def update_color_preview(self):
        """Met à jour l'aperçu de couleur."""
        gray_value = FLOOR_TO_GRAY[self.current_floor]
        color = f"#{gray_value:02x}{gray_value:02x}{gray_value:02x}"
        self.color_preview.delete("all")
        self.color_preview.create_rectangle(0, 0, 200, 30, fill=color, outline="black")
        self.color_preview.create_text(100, 15, text=f"Étage {self.current_floor}: {gray_value}", fill="white" if gray_value < 128 else "black")
    
    def update_info(self):
        """Met à jour les informations affichées."""
        info = f"Dimensions: {self.width}×{self.height}px\n"
        tool_text = f"{self.current_tool}"
        if self.erase_mode:
            tool_text += " (gomme)"
        info += f"Outil: {tool_text}\n"
        info += f"Étage: {self.current_floor} ({FLOOR_NAMES[self.current_floor]})\n"
        info += f"Taille pinceau: {self.brush_size}px\n"
        info += f"Carrelable: {'Oui' if self.tileable else 'Non'}\n"
        ba = "Oui" if self.angle_snap else "Non"
        if self.angle_snap:
            ba += f" (seuil crayon: {self.angle_snap_lock_pixels}px)"
        info += f"Bridage angles: {ba}\n"
        info += f"Mode gomme: {'Oui' if self.erase_mode else 'Non'}"
        self.info_label.config(text=info)
    
    def canvas_to_image_coords(self, x: int, y: int, allow_wrapping: bool = None) -> Tuple[int, int]:
        """
        Convertit les coordonnées du canevas en coordonnées d'image.
        
        Paramètres :
            x, y: Coordonnées du canevas
            allow_wrapping: Si True, permet l'enroulement (mode carrelable). Si None, utilise self.tileable.
                           Si False, borner aux bords (mode normal).
        """
        # Prendre en compte le zoom et le pan
        img_x = int((x - self.pan_x) / self.zoom_level)
        img_y = int((y - self.pan_y) / self.zoom_level)
        
        # Déterminer si on doit enrouler
        should_wrap = allow_wrapping if allow_wrapping is not None else self.tileable
        
        if should_wrap:
            # Mode carrelable : utiliser modulo pour enrouler
            img_x = img_x % self.width
            img_y = img_y % self.height
            # Gérer les valeurs négatives
            if img_x < 0:
                img_x += self.width
            if img_y < 0:
                img_y += self.height
        else:
            # Mode normal : borner aux bords
            img_x = max(0, min(self.width - 1, img_x))
            img_y = max(0, min(self.height - 1, img_y))
        
        return img_x, img_y
    
    def canvas_to_image_coords_raw(self, cx: float, cy: float) -> Tuple[float, float]:
        """Coordonnées image sans enroulement (plan infini), pour le mode carrelable."""
        return (cx - self.pan_x) / self.zoom_level, (cy - self.pan_y) / self.zoom_level
    
    def _event_canvas_xy_raw_image(self, event: tk.Event) -> Tuple[float, float]:
        """
        Coordonnées image brutes à partir du signal souris, avant verrou Maj / repositionnement du curseur.
        Nécessaire pour que le crayon en mode carrelable suive une trajectoire continue quand le curseur
        est ramené sur la tuile.
        """
        px = self.canvas.canvasx(event.x)
        py = self.canvas.canvasy(event.y)
        return self.canvas_to_image_coords_raw(px, py)
    
    def _continue_raw_axis(self, last_u: float, raw_measured: float, size: int) -> float:
        """
        Ramène raw_measured sur la même « feuille » que last_u (last_u + k·size le plus proche).
        Indispensable avec Maj : après repositionnement du curseur, la mesure saute (ex. 1023→12) alors que le tracé
        doit suivre 1024, 1025… — sinon Bresenham parcourt toute la ligne horizontale.
        """
        if size <= 1:
            return raw_measured
        k = round((last_u - raw_measured) / size)
        return raw_measured + k * size
    
    def _tileable_wrap_coords_enabled(self) -> bool:
        return self.tileable and self.current_tool in TOOLS_TILEABLE_WRAP
    
    def snap_to_angle_raw(
        self, start_x: float, start_y: float, target_x: float, target_y: float
    ) -> Tuple[int, int]:
        """Bridage d'angles en coordonnées non enroulées (mode carrelable)."""
        if not self.angle_snap:
            return int(round(target_x)), int(round(target_y))
        dx = target_x - start_x
        dy = target_y - start_y
        if abs(dx) < 1e-6 and abs(dy) < 1e-6:
            return int(round(target_x)), int(round(target_y))
        angle_rad = math.atan2(dy, dx)
        angle_deg = math.degrees(angle_rad)
        if angle_deg < 0:
            angle_deg += 360
        allowed_angles = [0, 45, 90, 135, 180, 225, 270, 315]
        closest_angle = min(
            allowed_angles,
            key=lambda a: min(
                abs(angle_deg - a), abs(angle_deg - a + 360), abs(angle_deg - a - 360)
            ),
        )
        distance = math.sqrt(dx * dx + dy * dy)
        angle_rad_snapped = math.radians(closest_angle)
        new_x = int(round(start_x + distance * math.cos(angle_rad_snapped)))
        new_y = int(round(start_y + distance * math.sin(angle_rad_snapped)))
        return new_x, new_y
    
    def _brush_cursor_lock_from_event(self, event: tk.Event) -> bool:
        """Maj maintenue pendant le mouvement souris (masque Tk standard)."""
        return (
            self.tileable
            and self.current_tool == "pencil"
            and bool(event.state & 0x0001)
        )
    
    def _warp_pointer_to_canvas(self, tcx: float, tcy: float) -> None:
        """Repositionne le curseur système sur (tcx, tcy) en coordonnées du widget canevas."""
        ix, iy = int(round(tcx)), int(round(tcy))
        if sys.platform == "win32":
            try:
                import ctypes
                sx = self.canvas.winfo_rootx() + ix
                sy = self.canvas.winfo_rooty() + iy
                ctypes.windll.user32.SetCursorPos(int(sx), int(sy))
                return
            except (AttributeError, OSError, ValueError):
                pass
        self.canvas.event_generate("<Motion>", x=ix, y=iy, warp=True)
    
    def _apply_brush_cursor_lock(self, event: tk.Event) -> Tuple[float, float]:
        """
        Retourne les coordonnées du canevas à utiliser pour le pinceau.
        En mode verrou (Maj + mode carrelable + crayon), attire le curseur sur la position enroulée.
        """
        cx = self.canvas.canvasx(event.x)
        cy = self.canvas.canvasy(event.y)
        if not self._brush_cursor_lock_from_event(event):
            return cx, cy
        if self._is_warping_cursor:
            return cx, cy
        pix_x, pix_y = self.canvas_to_image_coords(cx, cy, allow_wrapping=True)
        tcx = pix_x * self.zoom_level + self.pan_x
        tcy = pix_y * self.zoom_level + self.pan_y
        if abs(cx - tcx) < 1.0 and abs(cy - tcy) < 1.0:
            return cx, cy
        self._is_warping_cursor = True
        try:
            self._warp_pointer_to_canvas(tcx, tcy)
        finally:
            self.root.after(1, lambda: setattr(self, "_is_warping_cursor", False))
        return tcx, tcy
    
    def clear_all_floors(self):
        """Nettoie complètement tous les étages (remet toute l'image à FLOOR_5)."""
        if messagebox.askyesno("Confirmer", "Voulez-vous vraiment nettoyer complètement tous les étages ?"):
            self.save_state()
            self.image = Image.new('L', (self.width, self.height), FLOOR_5)
            self.draw = ImageDraw.Draw(self.image)
            self.update_canvas()
    
    def wrap_coord(self, coord: int, size: int) -> int:
        """Enroule une coordonnée selon la taille (pour mode carrelable)."""
        if self.tileable:
            coord = coord % size
            if coord < 0:
                coord += size
            return coord
        else:
            return max(0, min(size - 1, coord))
    
    def snap_to_angle(self, start_x: int, start_y: int, target_x: int, target_y: int) -> Tuple[int, int]:
        """
        Ajuste les coordonnées cibles pour respecter les angles autorisés.
        Angles autorisés : 0° (horizontal), 90° (vertical), 45°, 135°.
        
        Paramètres :
            start_x, start_y: Point de départ
            target_x, target_y: Point cible
            
        Renvoie :
            Tuple (x, y) ajusté selon l'angle le plus proche
        """
        if not self.angle_snap:
            return target_x, target_y
        
        dx = target_x - start_x
        dy = target_y - start_y
        
        # Si le point est très proche du point de départ, ne rien faire
        if abs(dx) < 2 and abs(dy) < 2:
            return target_x, target_y
        
        # Calculer l'angle actuel (en degrés)
        angle_rad = math.atan2(dy, dx)
        angle_deg = math.degrees(angle_rad)
        if angle_deg < 0:
            angle_deg += 360
        
        # Angles autorisés : 0°, 45°, 90°, 135°, 180°, 225°, 270°, 315°
        # (0° et 180° = horizontal, 90° et 270° = vertical, 45°/135°/225°/315° = diagonales)
        allowed_angles = [0, 45, 90, 135, 180, 225, 270, 315]
        
        # Trouver l'angle autorisé le plus proche
        closest_angle = min(allowed_angles, key=lambda a: min(abs(angle_deg - a), abs(angle_deg - a + 360), abs(angle_deg - a - 360)))
        
        # Calculer la distance
        distance = math.sqrt(dx**2 + dy**2)
        
        # Calculer les nouvelles coordonnées selon l'angle choisi
        angle_rad_snapped = math.radians(closest_angle)
        new_x = int(start_x + distance * math.cos(angle_rad_snapped))
        new_y = int(start_y + distance * math.sin(angle_rad_snapped))
        
        return new_x, new_y
    
    def _closest_snap_angle_deg_from_delta(self, dx: float, dy: float) -> int:
        if abs(dx) < 1e-12 and abs(dy) < 1e-12:
            return 0
        angle_deg = math.degrees(math.atan2(dy, dx))
        if angle_deg < 0:
            angle_deg += 360
        return min(
            _ALLOWED_SNAP_ANGLES_DEG,
            key=lambda a: min(
                abs(angle_deg - a),
                abs(angle_deg - a + 360),
                abs(angle_deg - a - 360),
            ),
        )
    
    def _reset_pencil_angle_hysteresis(self) -> None:
        self._pencil_angle_lock_deg = None
    
    def snap_pencil_angle_segment(
        self, lx: float, ly: float, tx: float, ty: float, as_int: bool
    ) -> Tuple[float, float]:
        """
        Extrémité bridée pour le crayon : maintient l’angle courant tant que le curseur
        n’a pas parcouru au moins ``angle_snap_lock_pixels`` depuis le dernier ancrage.
        """
        if not self.angle_snap:
            return (tx, ty)
        thresh = max(0, int(self.angle_snap_lock_pixels))
        if thresh == 0:
            if as_int:
                sx, sy, ex, ey = (
                    int(round(lx)),
                    int(round(ly)),
                    int(round(tx)),
                    int(round(ty)),
                )
                rx, ry = self.snap_to_angle(sx, sy, ex, ey)
                return (float(rx), float(ry))
            nrx, nry = self.snap_to_angle_raw(lx, ly, tx, ty)
            return (float(nrx), float(nry))
        
        dx = tx - lx
        dy = ty - ly
        dist_lc = math.hypot(dx, dy)
        if dist_lc < 1e-9:
            return (tx, ty)
        
        if self._pencil_angle_lock_deg is None:
            self._pencil_angle_lock_deg = self._closest_snap_angle_deg_from_delta(dx, dy)
            self._pencil_angle_lock_ax = float(lx)
            self._pencil_angle_lock_ay = float(ly)
        else:
            dax = tx - self._pencil_angle_lock_ax
            day = ty - self._pencil_angle_lock_ay
            if math.hypot(dax, day) >= float(thresh):
                self._pencil_angle_lock_deg = self._closest_snap_angle_deg_from_delta(dax, day)
                self._pencil_angle_lock_ax = float(tx)
                self._pencil_angle_lock_ay = float(ty)
        
        # Projection signée sur la direction verrouillée (évite les « cordes » en mode carrelable :
        # avec hypot() toujours positif, un retour en arrière le long du même axe était
        # dessiné comme un long segment en avant sur le tore.)
        rad = math.radians(self._pencil_angle_lock_deg)
        ux = math.cos(rad)
        uy = math.sin(rad)
        s = dx * ux + dy * uy
        nx = lx + s * ux
        ny = ly + s * uy
        if as_int:
            return (float(int(round(nx))), float(int(round(ny))))
        return (nx, ny)
    
    def draw_pencil_line_with_interpolation(self, x1: int, y1: int, x2: int, y2: int, fill: int, use_square: bool = False):
        """
        Dessine une ligne avec interpolation pour éviter les espaces lors de mouvements rapides.
        Utilise l'algorithme de ligne de Bresenham pour dessiner pixel par pixel avec des cercles ou des rectangles.
        
        Paramètres :
            x1, y1: Point de départ
            x2, y2: Point d'arrivée
            fill: Couleur de remplissage
            use_square: Si True, dessine des rectangles au lieu de cercles (pour coins nets en mode bridage)
        """
        # Calculer la distance entre les deux points
        dx = abs(x2 - x1)
        dy = abs(y2 - y1)
        
        # Si la distance est très petite, dessiner juste un point
        if dx == 0 and dy == 0:
            x1_w = self.wrap_coord(x1, self.width)
            y1_w = self.wrap_coord(y1, self.height)
            x1_norm, x2_norm = x1_w - self.brush_size//2, x1_w + self.brush_size//2
            y1_norm, y2_norm = y1_w - self.brush_size//2, y1_w + self.brush_size//2
            x1_final, x2_final = min(x1_norm, x2_norm), max(x1_norm, x2_norm)
            y1_final, y2_final = min(y1_norm, y2_norm), max(y1_norm, y2_norm)
            if use_square:
                self.draw.rectangle([(x1_final, y1_final), (x2_final, y2_final)], fill=fill)
            else:
                self.draw.ellipse([(x1_final, y1_final), (x2_final, y2_final)], fill=fill)
            return
        
        # Algorithme de ligne de Bresenham pour parcourir tous les pixels de la ligne
        sx = 1 if x1 < x2 else -1
        sy = 1 if y1 < y2 else -1
        err = dx - dy
        
        x, y = x1, y1
        last_valid_x = x1
        last_valid_y = y1
        
        # Dessiner un cercle ou rectangle à chaque pixel de la ligne
        while True:
            # Vérifier si on est encore dans l'image (en mode carrelable, on peut sortir)
            x_in_bounds = 0 <= x < self.width
            y_in_bounds = 0 <= y < self.height
            
            # En mode carrelable, si on sort de l'image, on a atteint le bord — arrêter après avoir dessiné
            if self.tileable and (not x_in_bounds or not y_in_bounds):
                # On a atteint le bord, dessiner le dernier point valide et s'arrêter
                # Le dernier point valide est celui qui était encore dans l'image
                if last_valid_x != x or last_valid_y != y:
                    px_w = self.wrap_coord(last_valid_x, self.width)
                    py_w = self.wrap_coord(last_valid_y, self.height)
                    x1_shape, x2_shape = px_w - self.brush_size//2, px_w + self.brush_size//2
                    y1_shape, y2_shape = py_w - self.brush_size//2, py_w + self.brush_size//2
                    x1_norm, x2_norm = min(x1_shape, x2_shape), max(x1_shape, x2_shape)
                    y1_norm, y2_norm = min(y1_shape, y2_shape), max(y1_shape, y2_shape)
                    if use_square:
                        self.draw.rectangle([(x1_norm, y1_norm), (x2_norm, y2_norm)], fill=fill)
                    else:
                        self.draw.ellipse([(x1_norm, y1_norm), (x2_norm, y2_norm)], fill=fill)
                break
            
            # Si on est dans l'image, dessiner normalement
            if x_in_bounds and y_in_bounds:
                # Mémoriser ce point comme dernier point valide
                last_valid_x = x
                last_valid_y = y
                
                # Enrouler les coordonnées si nécessaire
                px_w = self.wrap_coord(x, self.width)
                py_w = self.wrap_coord(y, self.height)
                
                # Dessiner un cercle ou rectangle à cette position
                x1_shape, x2_shape = px_w - self.brush_size//2, px_w + self.brush_size//2
                y1_shape, y2_shape = py_w - self.brush_size//2, py_w + self.brush_size//2
                x1_norm, x2_norm = min(x1_shape, x2_shape), max(x1_shape, x2_shape)
                y1_norm, y2_norm = min(y1_shape, y2_shape), max(y1_shape, y2_shape)
                
                if use_square:
                    # Dessiner un rectangle pour des coins nets
                    self.draw.rectangle([(x1_norm, y1_norm), (x2_norm, y2_norm)], fill=fill)
                else:
                    # Dessiner un cercle
                    self.draw.ellipse([(x1_norm, y1_norm), (x2_norm, y2_norm)], fill=fill)
            
            # Arrêter si on a atteint le point final
            if x == x2 and y == y2:
                break
            
            # Calculer le prochain point avec l'algorithme de Bresenham
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x += sx
            if e2 < dx:
                err += dx
                y += sy
    
    def draw_pencil_line_with_interpolation_raw(
        self,
        rx1: float,
        ry1: float,
        rx2: float,
        ry2: float,
        fill: int,
        use_square: bool = False,
    ) -> None:
        """
        Interpolation Bresenham en coordonnées image non enroulées ; chaque pas est ramené sur la tuile.
        Évite la corde parasite quand le curseur franchit un bord en mode carrelable.
        """
        x1i = int(round(rx1))
        y1i = int(round(ry1))
        x2i = int(round(rx2))
        y2i = int(round(ry2))
        dx = abs(x2i - x1i)
        dy = abs(y2i - y1i)
        if dx == 0 and dy == 0:
            px_w = self.wrap_coord(x1i, self.width)
            py_w = self.wrap_coord(y1i, self.height)
            x1_shape, x2_shape = px_w - self.brush_size // 2, px_w + self.brush_size // 2
            y1_shape, y2_shape = py_w - self.brush_size // 2, py_w + self.brush_size // 2
            x1_norm, x2_norm = min(x1_shape, x2_shape), max(x1_shape, x2_shape)
            y1_norm, y2_norm = min(y1_shape, y2_shape), max(y1_shape, y2_shape)
            if use_square:
                self.draw.rectangle([(x1_norm, y1_norm), (x2_norm, y2_norm)], fill=fill)
            else:
                self.draw.ellipse([(x1_norm, y1_norm), (x2_norm, y2_norm)], fill=fill)
            return
        sx = 1 if x1i < x2i else -1
        sy = 1 if y1i < y2i else -1
        err = dx - dy
        x, y = x1i, y1i
        while True:
            px_w = self.wrap_coord(x, self.width)
            py_w = self.wrap_coord(y, self.height)
            x1_shape, x2_shape = px_w - self.brush_size // 2, px_w + self.brush_size // 2
            y1_shape, y2_shape = py_w - self.brush_size // 2, py_w + self.brush_size // 2
            x1_norm, x2_norm = min(x1_shape, x2_shape), max(x1_shape, x2_shape)
            y1_norm, y2_norm = min(y1_shape, y2_shape), max(y1_shape, y2_shape)
            if use_square:
                self.draw.rectangle([(x1_norm, y1_norm), (x2_norm, y2_norm)], fill=fill)
            else:
                self.draw.ellipse([(x1_norm, y1_norm), (x2_norm, y2_norm)], fill=fill)
            if x == x2i and y == y2i:
                break
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x += sx
            if e2 < dx:
                err += dx
                y += sy
    
    def draw_tileable_line(self, x1: int, y1: int, x2: int, y2: int, fill: int, width: int = 1):
        """Dessine une ligne en mode carrelable (gère le passage des bords et des coins)."""
        if not self.tileable:
            # Mode normal : dessiner directement
            self.draw.line([(x1, y1), (x2, y2)], fill=fill, width=width)
            return
        
        # Normaliser les coordonnées
        x1_w = self.wrap_coord(x1, self.width)
        y1_w = self.wrap_coord(y1, self.height)
        x2_w = self.wrap_coord(x2, self.width)
        y2_w = self.wrap_coord(y2, self.height)
        
        # Calculer les écarts entre coins après enroulement sur la tuile
        dx_w = x2_w - x1_w
        dy_w = y2_w - y1_w
        
        # Calculer les écarts bruts (sans enroulement) pour déterminer la direction
        dx_real = x2 - x1
        dy_real = y2 - y1
        
        # Vérifier si on traverse les bords (écart enroulé > la moitié de la dimension)
        crosses_x = abs(dx_w) > self.width // 2
        crosses_y = abs(dy_w) > self.height // 2
        
        if crosses_x and crosses_y:
            # Traversée de coin : la ligne traverse à la fois X et Y
            # On doit dessiner 3 segments : jusqu'au premier bord, entre les bords, et après le second bord
            
            # Déterminer l'ordre de traversée des bords
            # Calculer les intersections avec les bords
            
            # Intersection avec bord horizontal (gauche ou droite)
            if dx_real != 0:
                if dx_real > 0:
                    # Sort par la droite
                    t_x = (self.width - x1_w) / dx_real
                else:
                    # Sort par la gauche
                    t_x = -x1_w / dx_real
            else:
                t_x = float('inf')
            
            # Intersection avec bord vertical (haut ou bas)
            if dy_real != 0:
                if dy_real > 0:
                    # Sort par le bas
                    t_y = (self.height - y1_w) / dy_real
                else:
                    # Sort par le haut
                    t_y = -y1_w / dy_real
            else:
                t_y = float('inf')
            
            # Déterminer quel bord est traversé en premier
            if t_x < t_y:
                # Traverse d'abord le bord horizontal
                if dx_real > 0:
                    # Sort par la droite
                    exit_x = self.width - 1
                    exit_y = int(y1_w + t_x * dy_real)
                    exit_y = self.wrap_coord(exit_y, self.height)
                    # Segment 1 : du point de départ au bord droit
                    self.draw.line([(x1_w, y1_w), (exit_x, exit_y)], fill=fill, width=width)
                    # Segment 2 : du bord gauche au bord vertical
                    entry_x = 0
                    entry_y = exit_y
                    if dy_real != 0:
                        remaining_t = t_y - t_x
                        if dy_real > 0:
                            # Sort par le bas
                            final_y = self.height - 1
                            final_x = int(entry_x + remaining_t * dx_real)
                            final_x = self.wrap_coord(final_x, self.width)
                            self.draw.line([(entry_x, entry_y), (final_x, final_y)], fill=fill, width=width)
                            # Segment 3 : du bord haut au point d'arrivée
                            final_entry_x = final_x
                            final_entry_y = 0
                            self.draw.line([(final_entry_x, final_entry_y), (x2_w, y2_w)], fill=fill, width=width)
                        else:
                            # Sort par le haut
                            final_y = 0
                            final_x = int(entry_x + remaining_t * dx_real)
                            final_x = self.wrap_coord(final_x, self.width)
                            self.draw.line([(entry_x, entry_y), (final_x, final_y)], fill=fill, width=width)
                            # Segment 3 : du bord bas au point d'arrivée
                            final_entry_x = final_x
                            final_entry_y = self.height - 1
                            self.draw.line([(final_entry_x, final_entry_y), (x2_w, y2_w)], fill=fill, width=width)
                else:
                    # Sort par la gauche
                    exit_x = 0
                    exit_y = int(y1_w + t_x * dy_real)
                    exit_y = self.wrap_coord(exit_y, self.height)
                    # Segment 1 : du point de départ au bord gauche
                    self.draw.line([(x1_w, y1_w), (exit_x, exit_y)], fill=fill, width=width)
                    # Segment 2 : du bord droit au bord vertical
                    entry_x = self.width - 1
                    entry_y = exit_y
                    if dy_real != 0:
                        remaining_t = t_y - t_x
                        if dy_real > 0:
                            # Sort par le bas
                            final_y = self.height - 1
                            final_x = int(entry_x + remaining_t * dx_real)
                            final_x = self.wrap_coord(final_x, self.width)
                            self.draw.line([(entry_x, entry_y), (final_x, final_y)], fill=fill, width=width)
                            # Segment 3 : du bord haut au point d'arrivée
                            final_entry_x = final_x
                            final_entry_y = 0
                            self.draw.line([(final_entry_x, final_entry_y), (x2_w, y2_w)], fill=fill, width=width)
                        else:
                            # Sort par le haut
                            final_y = 0
                            final_x = int(entry_x + remaining_t * dx_real)
                            final_x = self.wrap_coord(final_x, self.width)
                            self.draw.line([(entry_x, entry_y), (final_x, final_y)], fill=fill, width=width)
                            # Segment 3 : du bord bas au point d'arrivée
                            final_entry_x = final_x
                            final_entry_y = self.height - 1
                            self.draw.line([(final_entry_x, final_entry_y), (x2_w, y2_w)], fill=fill, width=width)
            else:
                # Traverse d'abord le bord vertical
                if dy_real > 0:
                    # Sort par le bas
                    exit_x = int(x1_w + t_y * dx_real)
                    exit_x = self.wrap_coord(exit_x, self.width)
                    exit_y = self.height - 1
                    # Segment 1 : du point de départ au bord bas
                    self.draw.line([(x1_w, y1_w), (exit_x, exit_y)], fill=fill, width=width)
                    # Segment 2 : du bord haut au bord horizontal
                    entry_x = exit_x
                    entry_y = 0
                    if dx_real != 0:
                        remaining_t = t_x - t_y
                        if dx_real > 0:
                            # Sort par la droite
                            final_x = self.width - 1
                            final_y = int(entry_y + remaining_t * dy_real)
                            final_y = self.wrap_coord(final_y, self.height)
                            self.draw.line([(entry_x, entry_y), (final_x, final_y)], fill=fill, width=width)
                            # Segment 3 : du bord gauche au point d'arrivée
                            final_entry_x = 0
                            final_entry_y = final_y
                            self.draw.line([(final_entry_x, final_entry_y), (x2_w, y2_w)], fill=fill, width=width)
                        else:
                            # Sort par la gauche
                            final_x = 0
                            final_y = int(entry_y + remaining_t * dy_real)
                            final_y = self.wrap_coord(final_y, self.height)
                            self.draw.line([(entry_x, entry_y), (final_x, final_y)], fill=fill, width=width)
                            # Segment 3 : du bord droit au point d'arrivée
                            final_entry_x = self.width - 1
                            final_entry_y = final_y
                            self.draw.line([(final_entry_x, final_entry_y), (x2_w, y2_w)], fill=fill, width=width)
                else:
                    # Sort par le haut
                    exit_x = int(x1_w + t_y * dx_real)
                    exit_x = self.wrap_coord(exit_x, self.width)
                    exit_y = 0
                    # Segment 1 : du point de départ au bord haut
                    self.draw.line([(x1_w, y1_w), (exit_x, exit_y)], fill=fill, width=width)
                    # Segment 2 : du bord bas au bord horizontal
                    entry_x = exit_x
                    entry_y = self.height - 1
                    if dx_real != 0:
                        remaining_t = t_x - t_y
                        if dx_real > 0:
                            # Sort par la droite
                            final_x = self.width - 1
                            final_y = int(entry_y + remaining_t * dy_real)
                            final_y = self.wrap_coord(final_y, self.height)
                            self.draw.line([(entry_x, entry_y), (final_x, final_y)], fill=fill, width=width)
                            # Segment 3 : du bord gauche au point d'arrivée
                            final_entry_x = 0
                            final_entry_y = final_y
                            self.draw.line([(final_entry_x, final_entry_y), (x2_w, y2_w)], fill=fill, width=width)
                        else:
                            # Sort par la gauche
                            final_x = 0
                            final_y = int(entry_y + remaining_t * dy_real)
                            final_y = self.wrap_coord(final_y, self.height)
                            self.draw.line([(entry_x, entry_y), (final_x, final_y)], fill=fill, width=width)
                            # Segment 3 : du bord droit au point d'arrivée
                            final_entry_x = self.width - 1
                            final_entry_y = final_y
                            self.draw.line([(final_entry_x, final_entry_y), (x2_w, y2_w)], fill=fill, width=width)
        elif crosses_x:
            # La ligne traverse uniquement le bord horizontal
            if dx_w > 0:
                # Passer par la gauche
                if dx_w != 0:
                    t = -x1_w / (x2_w - x1_w + self.width)
                    mid_y = int(y1_w + t * (y2_w - y1_w))
                    mid_y = self.wrap_coord(mid_y, self.height)
                    self.draw.line([(x1_w, y1_w), (0, mid_y)], fill=fill, width=width)
                    self.draw.line([(self.width - 1, mid_y), (x2_w, y2_w)], fill=fill, width=width)
                else:
                    self.draw.line([(x1_w, y1_w), (x2_w, y2_w)], fill=fill, width=width)
            else:
                # Passer par la droite
                if dx_w != 0:
                    t = (self.width - x1_w) / (x2_w - x1_w - self.width)
                    mid_y = int(y1_w + t * (y2_w - y1_w))
                    mid_y = self.wrap_coord(mid_y, self.height)
                    self.draw.line([(x1_w, y1_w), (self.width - 1, mid_y)], fill=fill, width=width)
                    self.draw.line([(0, mid_y), (x2_w, y2_w)], fill=fill, width=width)
                else:
                    self.draw.line([(x1_w, y1_w), (x2_w, y2_w)], fill=fill, width=width)
        elif crosses_y:
            # La ligne traverse uniquement le bord vertical
            if dy_w > 0:
                # Passer par le haut
                if dy_w != 0:
                    t = -y1_w / (y2_w - y1_w + self.height)
                    mid_x = int(x1_w + t * (x2_w - x1_w))
                    mid_x = self.wrap_coord(mid_x, self.width)
                    self.draw.line([(x1_w, y1_w), (mid_x, 0)], fill=fill, width=width)
                    self.draw.line([(mid_x, self.height - 1), (x2_w, y2_w)], fill=fill, width=width)
                else:
                    self.draw.line([(x1_w, y1_w), (x2_w, y2_w)], fill=fill, width=width)
            else:
                # Passer par le bas
                if dy_w != 0:
                    t = (self.height - y1_w) / (y2_w - y1_w - self.height)
                    mid_x = int(x1_w + t * (x2_w - x1_w))
                    mid_x = self.wrap_coord(mid_x, self.width)
                    self.draw.line([(x1_w, y1_w), (mid_x, self.height - 1)], fill=fill, width=width)
                    self.draw.line([(mid_x, 0), (x2_w, y2_w)], fill=fill, width=width)
                else:
                    self.draw.line([(x1_w, y1_w), (x2_w, y2_w)], fill=fill, width=width)
        else:
            # Pas de traversée, dessiner normalement
            self.draw.line([(x1_w, y1_w), (x2_w, y2_w)], fill=fill, width=width)
    
    def draw_tileable_rectangle(self, x1: int, y1: int, x2: int, y2: int, fill: Optional[int] = None, 
                                 outline: Optional[int] = None, width: int = 1):
        """Dessine un rectangle en mode carrelable (gère correctement les parties enroulées)."""
        # Normaliser les coordonnées pour garantir x1 <= x2 et y1 <= y2
        x1_norm, x2_norm = min(x1, x2), max(x1, x2)
        y1_norm, y2_norm = min(y1, y2), max(y1, y2)
        
        if not self.tileable:
            # Mode normal
            if fill is not None:
                self.draw.rectangle([(x1_norm, y1_norm), (x2_norm, y2_norm)], fill=fill, outline=outline)
            else:
                self.draw.rectangle([(x1_norm, y1_norm), (x2_norm, y2_norm)], outline=outline, width=width)
            return
        
        # Mode carrelable : rectangle aligné aux axes dans la feuille de revêtement.
        # Important : si dx < largeur mais les coins enroulés croisent un bord (x1_w > x2_w),
        # il faut DEUX morceaux — pas un seul min/max des coins.
        dx = x2_norm - x1_norm
        dy = y2_norm - y1_norm
        
        x1_w = self.wrap_coord(x1_norm, self.width)
        y1_w = self.wrap_coord(y1_norm, self.height)
        x2_w = self.wrap_coord(x2_norm, self.width)
        y2_w = self.wrap_coord(y2_norm, self.height)
        
        if abs(dx) >= self.width:
            # Le rectangle couvre toute la largeur (ou plus)
            if fill is not None:
                if abs(dy) >= self.height:
                    self.draw.rectangle([(0, 0), (self.width - 1, self.height - 1)], fill=fill)
                else:
                    y_start = y1_w
                    y_end = y2_w
                    if y_start <= y_end:
                        self.draw.rectangle([(0, y_start), (self.width - 1, y_end)], fill=fill, outline=outline)
                    else:
                        self.draw.rectangle([(0, y_start), (self.width - 1, self.height - 1)], fill=fill, outline=outline)
                        self.draw.rectangle([(0, 0), (self.width - 1, y_end)], fill=fill, outline=outline)
            else:
                if abs(dy) >= self.height:
                    self.draw.line([(0, 0), (self.width - 1, 0)], fill=outline, width=width)
                    self.draw.line([(0, self.height - 1), (self.width - 1, self.height - 1)], fill=outline, width=width)
                else:
                    y_start = y1_w
                    y_end = y2_w
                    if y_start <= y_end:
                        self.draw.line([(0, y_start), (self.width - 1, y_start)], fill=outline, width=width)
                        self.draw.line([(0, y_end), (self.width - 1, y_end)], fill=outline, width=width)
                    else:
                        self.draw.line([(0, y_start), (self.width - 1, y_start)], fill=outline, width=width)
                        self.draw.line([(0, self.height - 1), (self.width - 1, self.height - 1)], fill=outline, width=width)
                        self.draw.line([(0, 0), (self.width - 1, 0)], fill=outline, width=width)
                        self.draw.line([(0, y_end), (self.width - 1, y_end)], fill=outline, width=width)
        elif abs(dy) >= self.height:
            if fill is not None:
                x_start = x1_w
                x_end = x2_w
                if x_start <= x_end:
                    self.draw.rectangle([(x_start, 0), (x_end, self.height - 1)], fill=fill, outline=outline)
                else:
                    self.draw.rectangle([(x_start, 0), (self.width - 1, self.height - 1)], fill=fill, outline=outline)
                    self.draw.rectangle([(0, 0), (x_end, self.height - 1)], fill=fill, outline=outline)
            else:
                x_start = x1_w
                x_end = x2_w
                if x_start <= x_end:
                    self.draw.line([(x_start, 0), (x_start, self.height - 1)], fill=outline, width=width)
                    self.draw.line([(x_end, 0), (x_end, self.height - 1)], fill=outline, width=width)
                else:
                    self.draw.line([(x_start, 0), (x_start, self.height - 1)], fill=outline, width=width)
                    self.draw.line([(self.width - 1, 0), (self.width - 1, self.height - 1)], fill=outline, width=width)
                    self.draw.line([(0, 0), (0, self.height - 1)], fill=outline, width=width)
                    self.draw.line([(x_end, 0), (x_end, self.height - 1)], fill=outline, width=width)
        else:
            # Fenêtre contenue dans une tuile « logique » : 1 bloc ou 2 (couture H ou V)
            x_start, x_end = x1_w, x2_w
            y_start, y_end = y1_w, y2_w
            
            if x_start > x_end:
                if fill is not None:
                    self.draw.rectangle([(x_start, y_start), (self.width - 1, y_end)], fill=fill, outline=outline)
                    self.draw.rectangle([(0, y_start), (x_end, y_end)], fill=fill, outline=outline)
                else:
                    self.draw.line([(x_start, y_start), (self.width - 1, y_start)], fill=outline, width=width)
                    self.draw.line([(0, y_start), (x_end, y_start)], fill=outline, width=width)
                    self.draw.line([(x_start, y_end), (self.width - 1, y_end)], fill=outline, width=width)
                    self.draw.line([(0, y_end), (x_end, y_end)], fill=outline, width=width)
                    self.draw.line([(x_start, y_start), (x_start, y_end)], fill=outline, width=width)
                    self.draw.line([(x_end, y_start), (x_end, y_end)], fill=outline, width=width)
            elif y_start > y_end:
                if fill is not None:
                    self.draw.rectangle([(x_start, y_start), (x_end, self.height - 1)], fill=fill, outline=outline)
                    self.draw.rectangle([(x_start, 0), (x_end, y_end)], fill=fill, outline=outline)
                else:
                    self.draw.line([(x_start, y_start), (x_start, self.height - 1)], fill=outline, width=width)
                    self.draw.line([(x_start, 0), (x_start, y_end)], fill=outline, width=width)
                    self.draw.line([(x_end, y_start), (x_end, self.height - 1)], fill=outline, width=width)
                    self.draw.line([(x_end, 0), (x_end, y_end)], fill=outline, width=width)
                    self.draw.line([(x_start, y_start), (x_end, y_start)], fill=outline, width=width)
                    self.draw.line([(x_start, y_end), (x_end, y_end)], fill=outline, width=width)
            else:
                if fill is not None:
                    self.draw.rectangle([(x_start, y_start), (x_end, y_end)], fill=fill, outline=outline)
                else:
                    self.draw.rectangle([(x_start, y_start), (x_end, y_end)], outline=outline, width=width)
    
    def on_mouse_down(self, event):
        """Gère le clic de souris."""
        self.canvas.focus_set()
        self.drawing = True
        self.state_saved_for_current_action = False  # Réinitialiser le flag pour la nouvelle action
        allow_wrap = self._tileable_wrap_coords_enabled()
        raw_x, raw_y = self._event_canvas_xy_raw_image(event)
        cx, cy = self._apply_brush_cursor_lock(event)
        if self._tileable_wrap_coords_enabled():
            self._stroke_raw_x0 = raw_x
            self._stroke_raw_y0 = raw_y
            if self.current_tool == "pencil":
                self._pencil_last_raw_x = raw_x
                self._pencil_last_raw_y = raw_y
            elif self.current_tool in TOOLS_TILEABLE_GEOM:
                self._geom_last_raw_x = raw_x
                self._geom_last_raw_y = raw_y
        x, y = self.canvas_to_image_coords(cx, cy, allow_wrapping=allow_wrap)
        self.cursor_x, self.cursor_y = cx, cy
        self.start_x = x
        self.start_y = y
        self.last_x = x
        self.last_y = y
        if self.current_tool == "pencil":
            self._reset_pencil_angle_hysteresis()
        
        gray_value = FLOOR_TO_GRAY[self.current_floor]
        
        if self.current_tool == "pencil":
            # Sauvegarder l'état avant de dessiner
            if not self.state_saved_for_current_action:
                self.save_state()
                self.state_saved_for_current_action = True
            # Dessiner un point ou gommer selon le mode
            x1, x2 = x - self.brush_size//2, x + self.brush_size//2
            y1, y2 = y - self.brush_size//2, y + self.brush_size//2
            # Normaliser les coordonnées pour garantir x1 <= x2 et y1 <= y2
            x1_norm, x2_norm = min(x1, x2), max(x1, x2)
            y1_norm, y2_norm = min(y1, y2), max(y1, y2)
            if self.erase_mode:
                # Mode gomme : remettre à FLOOR_5
                self.draw.ellipse(
                    [(x1_norm, y1_norm), (x2_norm, y2_norm)],
                    fill=FLOOR_5
                )
            else:
                # Mode dessin : dessiner l'ellipse
                self.draw.ellipse(
                    [(x1_norm, y1_norm), (x2_norm, y2_norm)],
                    fill=gray_value
                )
            self.update_canvas()
        elif self.current_tool == "fill":
            # Sauvegarder l'état avant de remplir
            if not self.state_saved_for_current_action:
                self.save_state()
                self.state_saved_for_current_action = True
            if self.erase_mode:
                # Mode gomme : remettre à FLOOR_5 (petit cercle)
                x1, x2 = x - self.brush_size//2, x + self.brush_size//2
                y1, y2 = y - self.brush_size//2, y + self.brush_size//2
                x1_norm, x2_norm = min(x1, x2), max(x1, x2)
                y1_norm, y2_norm = min(y1, y2), max(y1, y2)
                self.draw.ellipse(
                    [(x1_norm, y1_norm), (x2_norm, y2_norm)],
                    fill=FLOOR_5
                )
            else:
                # Remplissage normal
                self.flood_fill(x, y, gray_value)
            self.update_canvas()
    
    def on_mouse_drag(self, event):
        """Gère le glissement de la souris."""
        if not self.drawing:
            return
        
        # Sauvegarder l'état au début du drag (une seule fois)
        if not self.state_saved_for_current_action:
            self.save_state()
            self.state_saved_for_current_action = True
        
        raw_x_m, raw_y_m = self._event_canvas_xy_raw_image(event)
        self.cursor_x, self.cursor_y = self._apply_brush_cursor_lock(event)
        
        allow_wrap = self._tileable_wrap_coords_enabled()
        x, y = self.canvas_to_image_coords(self.cursor_x, self.cursor_y, allow_wrapping=allow_wrap)
        
        if self.tileable:
            if self.current_tool == "pencil":
                raw_x = self._continue_raw_axis(
                    self._pencil_last_raw_x, raw_x_m, self.width
                )
                raw_y = self._continue_raw_axis(
                    self._pencil_last_raw_y, raw_y_m, self.height
                )
            elif self.current_tool in TOOLS_TILEABLE_GEOM:
                raw_x = self._continue_raw_axis(
                    self._geom_last_raw_x, raw_x_m, self.width
                )
                raw_y = self._continue_raw_axis(
                    self._geom_last_raw_y, raw_y_m, self.height
                )
            else:
                raw_x, raw_y = raw_x_m, raw_y_m
        else:
            raw_x, raw_y = raw_x_m, raw_y_m
        
        if self.angle_snap and self.current_tool == "pencil":
            if self.tileable:
                nrx, nry = self.snap_pencil_angle_segment(
                    self._pencil_last_raw_x,
                    self._pencil_last_raw_y,
                    raw_x,
                    raw_y,
                    as_int=False,
                )
                raw_x, raw_y = float(nrx), float(nry)
                x = self.wrap_coord(int(round(nrx)), self.width)
                y = self.wrap_coord(int(round(nry)), self.height)
            else:
                sx, sy = self.snap_pencil_angle_segment(
                    float(self.last_x),
                    float(self.last_y),
                    float(x),
                    float(y),
                    as_int=True,
                )
                x, y = int(sx), int(sy)
        elif self.angle_snap and self.current_tool in GEOM_TOOLS:
            if self.tileable and self.current_tool in TOOLS_TILEABLE_GEOM:
                nrx, nry = self.snap_to_angle_raw(
                    self._stroke_raw_x0, self._stroke_raw_y0, raw_x, raw_y
                )
                raw_x, raw_y = float(nrx), float(nry)
                x = self.wrap_coord(nrx, self.width)
                y = self.wrap_coord(nry, self.height)
            else:
                x, y = self.snap_to_angle(self.start_x, self.start_y, x, y)
        
        gray_value = FLOOR_TO_GRAY[self.current_floor]
        
        if self.current_tool == "pencil":
            # Ligne entre le dernier point et le point actuel
            if self.tileable:
                if self.erase_mode:
                    self.draw_pencil_line_with_interpolation_raw(
                        self._pencil_last_raw_x,
                        self._pencil_last_raw_y,
                        raw_x,
                        raw_y,
                        FLOOR_5,
                        use_square=self.angle_snap,
                    )
                else:
                    self.draw_pencil_line_with_interpolation_raw(
                        self._pencil_last_raw_x,
                        self._pencil_last_raw_y,
                        raw_x,
                        raw_y,
                        gray_value,
                        use_square=self.angle_snap,
                    )
                self._pencil_last_raw_x = raw_x
                self._pencil_last_raw_y = raw_y
                self.last_x = x
                self.last_y = y
            elif self.erase_mode:
                # Mode gomme : remettre à FLOOR_5
                if self.angle_snap:
                    snapped_x, snapped_y = x, y
                    snapped_x = max(0, min(self.width - 1, snapped_x))
                    snapped_y = max(0, min(self.height - 1, snapped_y))
                    last_x_clamped = max(0, min(self.width - 1, self.last_x))
                    last_y_clamped = max(0, min(self.height - 1, self.last_y))
                    self.draw.line(
                        [(last_x_clamped, last_y_clamped), (snapped_x, snapped_y)],
                        fill=FLOOR_5,
                        width=self.brush_size,
                    )
                    self.last_x = snapped_x
                    self.last_y = snapped_y
                else:
                    x_clamped = max(0, min(self.width - 1, x))
                    y_clamped = max(0, min(self.height - 1, y))
                    last_x_clamped = max(0, min(self.width - 1, self.last_x))
                    last_y_clamped = max(0, min(self.height - 1, self.last_y))
                    self.draw.line(
                        [(last_x_clamped, last_y_clamped), (x_clamped, y_clamped)],
                        fill=FLOOR_5,
                        width=self.brush_size,
                    )
                    self.last_x = x
                    self.last_y = y
            else:
                if self.angle_snap:
                    snapped_x, snapped_y = x, y
                    self.draw_pencil_line_with_interpolation(
                        self.last_x, self.last_y, snapped_x, snapped_y, gray_value, use_square=True
                    )
                    self.last_x = snapped_x
                    self.last_y = snapped_y
                else:
                    self.draw_pencil_line_with_interpolation(
                        self.last_x, self.last_y, x, y, gray_value, use_square=False
                    )
                    self.last_x = x
                    self.last_y = y
            self.update_canvas()
        elif self.current_tool in GEOM_TOOLS:
            # Mettre à jour la dernière position connue
            self.last_x = x
            self.last_y = y
            if self.tileable and self.current_tool in TOOLS_TILEABLE_GEOM:
                self._geom_last_raw_x = raw_x
                self._geom_last_raw_y = raw_y
            
            preview_raw = (
                (raw_x, raw_y)
                if self.tileable and self.current_tool in TOOLS_TILEABLE_GEOM
                else None
            )
            self.update_canvas_preview(x, y, preview_raw)
    
    def on_mouse_up(self, event):
        """Gère le relâchement de la souris."""
        if not self.drawing:
            return
        
        self.drawing = False
        allow_wrap = self._tileable_wrap_coords_enabled()
        raw_x_m, raw_y_m = self._event_canvas_xy_raw_image(event)
        cx, cy = self._apply_brush_cursor_lock(event)
        x, y = self.canvas_to_image_coords(cx, cy, allow_wrapping=allow_wrap)
        self.cursor_x, self.cursor_y = cx, cy
        
        if self.tileable and self.current_tool in TOOLS_TILEABLE_GEOM:
            raw_x = self._continue_raw_axis(self._geom_last_raw_x, raw_x_m, self.width)
            raw_y = self._continue_raw_axis(self._geom_last_raw_y, raw_y_m, self.height)
        elif self.tileable and self.current_tool == "pencil":
            raw_x, raw_y = raw_x_m, raw_y_m
        else:
            raw_x, raw_y = raw_x_m, raw_y_m
        
        if self.angle_snap and self.current_tool in GEOM_TOOLS:
            if self.tileable and self.current_tool in TOOLS_TILEABLE_GEOM:
                nrx, nry = self.snap_to_angle_raw(
                    self._stroke_raw_x0, self._stroke_raw_y0, raw_x, raw_y
                )
                raw_x, raw_y = float(nrx), float(nry)
                x = self.wrap_coord(nrx, self.width)
                y = self.wrap_coord(nry, self.height)
            else:
                x, y = self.snap_to_angle(self.start_x, self.start_y, x, y)
        
        # Sauvegarder l'état avant de dessiner la forme finale (si pas déjà fait)
        if not self.state_saved_for_current_action:
            self.save_state()
            self.state_saved_for_current_action = True
        
        gray_value = FLOOR_TO_GRAY[self.current_floor]
        
        if self.current_tool == "line":
            # Ligne : toujours segment dans l’image (pas d’enroulement), même si « carrelable » est coché.
            start_x_clamped = max(0, min(self.width - 1, self.start_x))
            start_y_clamped = max(0, min(self.height - 1, self.start_y))
            x_clamped = max(0, min(self.width - 1, x))
            y_clamped = max(0, min(self.height - 1, y))
            if self.erase_mode:
                self.draw.line(
                    [(start_x_clamped, start_y_clamped), (x_clamped, y_clamped)],
                    fill=FLOOR_5,
                    width=self.brush_size,
                )
            else:
                self.draw.line(
                    [(start_x_clamped, start_y_clamped), (x_clamped, y_clamped)],
                    fill=gray_value,
                    width=self.brush_size,
                )
            self.update_canvas()
        elif self.current_tool == "rectangle":
            if self.tileable:
                rx0 = int(round(self._stroke_raw_x0))
                ry0 = int(round(self._stroke_raw_y0))
                rx1 = int(round(raw_x))
                ry1 = int(round(raw_y))
                if self.erase_mode:
                    self.draw_tileable_rectangle(
                        rx0, ry0, rx1, ry1,
                        fill=None,
                        outline=FLOOR_5,
                        width=self.brush_size,
                    )
                else:
                    self.draw_tileable_rectangle(
                        rx0, ry0, rx1, ry1,
                        fill=None,
                        outline=gray_value,
                        width=self.brush_size,
                    )
            else:
                x1_norm, x2_norm = min(self.start_x, x), max(self.start_x, x)
                y1_norm, y2_norm = min(self.start_y, y), max(self.start_y, y)
                x1_clamped = max(0, min(self.width - 1, x1_norm))
                y1_clamped = max(0, min(self.height - 1, y1_norm))
                x2_clamped = max(0, min(self.width - 1, x2_norm))
                y2_clamped = max(0, min(self.height - 1, y2_norm))
                if self.erase_mode:
                    self.draw.rectangle(
                        [(x1_clamped, y1_clamped), (x2_clamped, y2_clamped)],
                        outline=FLOOR_5,
                        width=self.brush_size,
                    )
                else:
                    self.draw.rectangle(
                        [(x1_clamped, y1_clamped), (x2_clamped, y2_clamped)],
                        outline=gray_value,
                        width=self.brush_size,
                    )
            self.update_canvas()
        elif self.current_tool == "rectangle_filled":
            if self.tileable:
                rx0 = int(round(self._stroke_raw_x0))
                ry0 = int(round(self._stroke_raw_y0))
                rx1 = int(round(raw_x))
                ry1 = int(round(raw_y))
                if self.erase_mode:
                    self.draw_tileable_rectangle(
                        rx0, ry0, rx1, ry1,
                        fill=FLOOR_5,
                        outline=FLOOR_5,
                    )
                else:
                    self.draw_tileable_rectangle(
                        rx0, ry0, rx1, ry1,
                        fill=gray_value,
                        outline=gray_value,
                    )
            else:
                x1_norm, x2_norm = min(self.start_x, x), max(self.start_x, x)
                y1_norm, y2_norm = min(self.start_y, y), max(self.start_y, y)
                x1_clamped = max(0, min(self.width - 1, x1_norm))
                y1_clamped = max(0, min(self.height - 1, y1_norm))
                x2_clamped = max(0, min(self.width - 1, x2_norm))
                y2_clamped = max(0, min(self.height - 1, y2_norm))
                if self.erase_mode:
                    self.draw.rectangle(
                        [(x1_clamped, y1_clamped), (x2_clamped, y2_clamped)],
                        fill=FLOOR_5,
                        outline=FLOOR_5,
                    )
                else:
                    self.draw.rectangle(
                        [(x1_clamped, y1_clamped), (x2_clamped, y2_clamped)],
                        fill=gray_value,
                        outline=gray_value,
                    )
            self.update_canvas()
        elif self.current_tool == "ellipse":
            # Ellipse : pas d’enroulement carrelable (rectangle de sélection borné dans l’image)
            x1_norm, x2_norm = min(self.start_x, x), max(self.start_x, x)
            y1_norm, y2_norm = min(self.start_y, y), max(self.start_y, y)
            x1_clamped = max(0, min(self.width - 1, x1_norm))
            y1_clamped = max(0, min(self.height - 1, y1_norm))
            x2_clamped = max(0, min(self.width - 1, x2_norm))
            y2_clamped = max(0, min(self.height - 1, y2_norm))
            if self.erase_mode:
                self.draw.ellipse(
                    [(x1_clamped, y1_clamped), (x2_clamped, y2_clamped)],
                    outline=FLOOR_5,
                    width=self.brush_size,
                )
            else:
                self.draw.ellipse(
                    [(x1_clamped, y1_clamped), (x2_clamped, y2_clamped)],
                    outline=gray_value,
                    width=self.brush_size,
                )
            self.update_canvas()
        elif self.current_tool == "ellipse_filled":
            x1_norm, x2_norm = min(self.start_x, x), max(self.start_x, x)
            y1_norm, y2_norm = min(self.start_y, y), max(self.start_y, y)
            x1_clamped = max(0, min(self.width - 1, x1_norm))
            y1_clamped = max(0, min(self.height - 1, y1_norm))
            x2_clamped = max(0, min(self.width - 1, x2_norm))
            y2_clamped = max(0, min(self.height - 1, y2_norm))
            if self.erase_mode:
                self.draw.ellipse(
                    [(x1_clamped, y1_clamped), (x2_clamped, y2_clamped)],
                    fill=FLOOR_5,
                    outline=FLOOR_5,
                )
            else:
                self.draw.ellipse(
                    [(x1_clamped, y1_clamped), (x2_clamped, y2_clamped)],
                    fill=gray_value,
                    outline=gray_value,
                )
            self.update_canvas()
        
        # Réinitialiser le flag pour la prochaine action
        self.state_saved_for_current_action = False
    
    def on_mouse_move(self, event):
        """Gère le mouvement de la souris (pour afficher les coordonnées)."""
        # Stocker la position du curseur en coordonnées du canevas
        self.cursor_x, self.cursor_y = self._apply_brush_cursor_lock(event)
        
        allow_wrap = self._tileable_wrap_coords_enabled()
        x, y = self.canvas_to_image_coords(self.cursor_x, self.cursor_y, allow_wrapping=allow_wrap)
        # Optionnel: afficher les coordonnées dans la barre de statut
        
        # Mettre à jour l'affichage pour montrer les curseurs enroulés en mode carrelable
        if self.tileable and not self.drawing:
            self.update_cursor_indicators()
    
    def on_mouse_leave(self, event):
        """Gère la sortie de la souris du canevas."""
        if self.drawing and self._brush_cursor_lock_from_event(event):
            return
        self.cursor_x = None
        self.cursor_y = None
        # Supprimer les indicateurs de curseur
        self.canvas.delete("cursor_indicator")
    
    def on_mouse_wheel(self, event):
        """Gère la molette de la souris pour le zoom."""
        if event.delta > 0 or event.num == 4:
            self.set_zoom(self.zoom_level * 1.1)
        else:
            self.set_zoom(self.zoom_level / 1.1)
    
    def flood_fill(self, x: int, y: int, fill_value: int):
        """Remplissage par propagation (inondation) — mode normal, pas carrelable."""
        # Borner les coordonnées aux bords de l'image
        x = max(0, min(self.width - 1, x))
        y = max(0, min(self.height - 1, y))
        
        # Obtenir la valeur actuelle du pixel
        current_value = self.image.getpixel((x, y))
        
        # Si c'est déjà la même valeur, ne rien faire
        if current_value == fill_value:
            return
        
        # Algorithme de remplissage simple (peut être lent sur de grandes zones)
        stack = [(x, y)]
        visited = set()
        
        while stack:
            cx, cy = stack.pop()
            
            if (cx, cy) in visited:
                continue
            
            # Mode normal : vérifier les bords
            if cx < 0 or cx >= self.width or cy < 0 or cy >= self.height:
                continue
            
            if self.image.getpixel((cx, cy)) != current_value:
                continue
            
            visited.add((cx, cy))
            self.image.putpixel((cx, cy), fill_value)
            
            # Ajouter les voisins
            stack.append((cx + 1, cy))
            stack.append((cx - 1, cy))
            stack.append((cx, cy + 1))
            stack.append((cx, cy - 1))
    
    def center_image(self):
        """Centre l'image dans le canevas."""
        # Obtenir les dimensions du canevas
        self.canvas.update_idletasks()  # S'assurer que les dimensions sont à jour
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        
        # Calculer les dimensions de l'image zoomée
        img_width = int(self.width * self.zoom_level)
        img_height = int(self.height * self.zoom_level)
        
        # Centrer l'image si elle est plus petite que le canevas
        if canvas_width > 1 and canvas_height > 1:
            if img_width < canvas_width:
                self.pan_x = (canvas_width - img_width) // 2
            else:
                self.pan_x = 0
            
            if img_height < canvas_height:
                self.pan_y = (canvas_height - img_height) // 2
            else:
                self.pan_y = 0
    
    def update_canvas(self):
        """Met à jour l'affichage du canevas."""
        if self.image is None:
            return
        
        # Centrer l'image si nécessaire
        self.center_image()
        
        # Convertir l'image en RGB pour l'affichage
        img_rgb = self.image.convert('RGB')
        
        # Appliquer le zoom
        if self.zoom_level != 1.0:
            new_width = int(self.width * self.zoom_level)
            new_height = int(self.height * self.zoom_level)
            img_rgb = img_rgb.resize((new_width, new_height), Image.NEAREST)
        
        self.photo = ImageTk.PhotoImage(img_rgb)
        
        # Mettre à jour le canevas
        self.canvas.delete("all")
        self.canvas.create_image(
            self.pan_x,
            self.pan_y,
            anchor=tk.NW,
            image=self.photo
        )
        
        # Afficher les indicateurs de curseur en mode carrelable
        if self.tileable and not self.drawing and self.cursor_x is not None and self.cursor_y is not None:
            self.draw_cursor_indicators()
        
        # Ajuster la zone de défilement
        self.canvas.config(scrollregion=self.canvas.bbox("all"))
        self.zoom_label.config(text=f"{int(self.zoom_level * 100)}%")
    
    def update_cursor_indicators(self):
        """Met à jour uniquement les indicateurs de curseur (sans redessiner toute l'image)."""
        if self.image is None or not self.tileable or self.cursor_x is None or self.cursor_y is None:
            return
        
        # Supprimer les indicateurs déjà affichés
        self.canvas.delete("cursor_indicator")
        
        # Dessiner les nouveaux indicateurs
        self.draw_cursor_indicators()
    
    def draw_cursor_indicators(self):
        """Dessine les indicateurs de curseur aux positions enroulées."""
        if not self.tileable or self.cursor_x is None or self.cursor_y is None:
            return
        
        # Convertir les coordonnées du canevas en coordonnées image (sans enroulement)
        img_x_raw = (self.cursor_x - self.pan_x) / self.zoom_level
        img_y_raw = (self.cursor_y - self.pan_y) / self.zoom_level
        
        # Calculer toutes les positions enroulées possibles
        # On veut montrer le curseur aux positions où il apparaîtrait si on sortait de l'image
        cursor_positions = []
        
        # Toujours calculer la position enroulée principale (celle qui est dans l'image)
        # Utiliser le modulo pour obtenir la position dans l'image
        img_x_wrapped = img_x_raw % self.width
        img_y_wrapped = img_y_raw % self.height
        if img_x_wrapped < 0:
            img_x_wrapped += self.width
        if img_y_wrapped < 0:
            img_y_wrapped += self.height
        
        # Position principale (toujours dans l'image grâce à l'enroulement)
        cursor_positions.append((img_x_wrapped, img_y_wrapped))
        
        # Si le curseur est en dehors de l'image visible, on montre aussi les positions enroulées
        # Horizontal
        if img_x_raw < 0:
            # Sortie à gauche, apparaît à droite
            cursor_positions.append((img_x_raw + self.width, img_y_wrapped))
        elif img_x_raw >= self.width:
            # Sortie à droite, apparaît à gauche
            cursor_positions.append((img_x_raw - self.width, img_y_wrapped))
        
        # Vertical
        if img_y_raw < 0:
            # Sortie en haut, apparaît en bas
            cursor_positions.append((img_x_wrapped, img_y_raw + self.height))
        elif img_y_raw >= self.height:
            # Sortie en bas, apparaît en haut
            cursor_positions.append((img_x_wrapped, img_y_raw - self.height))
        
        # Diagonales (coins) — positions enroulées complètes
        if img_x_raw < 0 and img_y_raw < 0:
            cursor_positions.append((img_x_raw + self.width, img_y_raw + self.height))
        elif img_x_raw >= self.width and img_y_raw < 0:
            cursor_positions.append((img_x_raw - self.width, img_y_raw + self.height))
        elif img_x_raw < 0 and img_y_raw >= self.height:
            cursor_positions.append((img_x_raw + self.width, img_y_raw - self.height))
        elif img_x_raw >= self.width and img_y_raw >= self.height:
            cursor_positions.append((img_x_raw - self.width, img_y_raw - self.height))
        
        # Dessiner les indicateurs pour chaque position
        for pos_x, pos_y in cursor_positions:
            # Convertir en coordonnées du canevas
            canvas_x = pos_x * self.zoom_level + self.pan_x
            canvas_y = pos_y * self.zoom_level + self.pan_y
            
            # Vérifier que la position est dans les limites du canevas (avec une marge)
            # On accepte les positions même légèrement en dehors pour voir le curseur aux bords
            canvas_width = self.canvas.winfo_width()
            canvas_height = self.canvas.winfo_height()
            
            # Afficher l'indicateur si la position est proche du canevas (dans une marge de 50px)
            margin = 50
            if -margin <= canvas_x <= canvas_width + margin and -margin <= canvas_y <= canvas_height + margin:
                # Dessiner un petit cercle pour indiquer la position
                size = max(3, int(5 * self.zoom_level))
                self.canvas.create_oval(
                    canvas_x - size, canvas_y - size,
                    canvas_x + size, canvas_y + size,
                    outline="red",
                    width=2,
                    tags="cursor_indicator"
                )
                # Dessiner une croix
                self.canvas.create_line(
                    canvas_x - size - 2, canvas_y,
                    canvas_x + size + 2, canvas_y,
                    fill="red",
                    width=1,
                    tags="cursor_indicator"
                )
                self.canvas.create_line(
                    canvas_x, canvas_y - size - 2,
                    canvas_x, canvas_y + size + 2,
                    fill="red",
                    width=1,
                    tags="cursor_indicator"
                )
    
    def update_canvas_preview(
        self,
        end_x: int,
        end_y: int,
        end_raw: Optional[Tuple[float, float]] = None,
    ):
        """Met à jour le canevas avec un aperçu de la forme en cours de dessin."""
        if self.image is None:
            return
        
        if self.angle_snap and self.current_tool in GEOM_TOOLS and end_raw is None:
            end_x, end_y = self.snap_to_angle(self.start_x, self.start_y, end_x, end_y)
        
        # Créer une copie temporaire pour l'aperçu
        temp_image = self.image.copy()
        
        # Sauvegarder le contexte de dessin actuel et en créer un temporaire pour l'aperçu
        original_draw = self.draw
        temp_draw = ImageDraw.Draw(temp_image)
        self.draw = temp_draw
        
        preview_color = FLOOR_5 if self.erase_mode else FLOOR_TO_GRAY[self.current_floor]
        
        try:
            if self.tileable and end_raw is not None:
                rx0 = int(round(self._stroke_raw_x0))
                ry0 = int(round(self._stroke_raw_y0))
                rx1 = int(round(end_raw[0]))
                ry1 = int(round(end_raw[1]))
                if self.current_tool == "rectangle":
                    self.draw_tileable_rectangle(
                        rx0, ry0, rx1, ry1,
                        fill=None,
                        outline=preview_color,
                        width=self.brush_size,
                    )
                elif self.current_tool == "rectangle_filled":
                    self.draw_tileable_rectangle(
                        rx0, ry0, rx1, ry1,
                        fill=preview_color,
                        outline=preview_color,
                    )
            else:
                if self.current_tool == "line":
                    start_x_clamped = max(0, min(self.width - 1, self.start_x))
                    start_y_clamped = max(0, min(self.height - 1, self.start_y))
                    end_x_clamped = max(0, min(self.width - 1, end_x))
                    end_y_clamped = max(0, min(self.height - 1, end_y))
                    temp_draw.line(
                        [(start_x_clamped, start_y_clamped), (end_x_clamped, end_y_clamped)],
                        fill=preview_color,
                        width=self.brush_size,
                    )
                elif self.current_tool == "rectangle":
                    x1_norm, x2_norm = min(self.start_x, end_x), max(self.start_x, end_x)
                    y1_norm, y2_norm = min(self.start_y, end_y), max(self.start_y, end_y)
                    x1_clamped = max(0, min(self.width - 1, x1_norm))
                    y1_clamped = max(0, min(self.height - 1, y1_norm))
                    x2_clamped = max(0, min(self.width - 1, x2_norm))
                    y2_clamped = max(0, min(self.height - 1, y2_norm))
                    temp_draw.rectangle(
                        [(x1_clamped, y1_clamped), (x2_clamped, y2_clamped)],
                        outline=preview_color,
                        width=self.brush_size,
                    )
                elif self.current_tool == "rectangle_filled":
                    x1_norm, x2_norm = min(self.start_x, end_x), max(self.start_x, end_x)
                    y1_norm, y2_norm = min(self.start_y, end_y), max(self.start_y, end_y)
                    x1_clamped = max(0, min(self.width - 1, x1_norm))
                    y1_clamped = max(0, min(self.height - 1, y1_norm))
                    x2_clamped = max(0, min(self.width - 1, x2_norm))
                    y2_clamped = max(0, min(self.height - 1, y2_norm))
                    temp_draw.rectangle(
                        [(x1_clamped, y1_clamped), (x2_clamped, y2_clamped)],
                        fill=preview_color,
                        outline=preview_color,
                    )
                elif self.current_tool == "ellipse":
                    x1_norm, x2_norm = min(self.start_x, end_x), max(self.start_x, end_x)
                    y1_norm, y2_norm = min(self.start_y, end_y), max(self.start_y, end_y)
                    x1_clamped = max(0, min(self.width - 1, x1_norm))
                    y1_clamped = max(0, min(self.height - 1, y1_norm))
                    x2_clamped = max(0, min(self.width - 1, x2_norm))
                    y2_clamped = max(0, min(self.height - 1, y2_norm))
                    temp_draw.ellipse(
                        [(x1_clamped, y1_clamped), (x2_clamped, y2_clamped)],
                        outline=preview_color,
                        width=self.brush_size,
                    )
                elif self.current_tool == "ellipse_filled":
                    x1_norm, x2_norm = min(self.start_x, end_x), max(self.start_x, end_x)
                    y1_norm, y2_norm = min(self.start_y, end_y), max(self.start_y, end_y)
                    x1_clamped = max(0, min(self.width - 1, x1_norm))
                    y1_clamped = max(0, min(self.height - 1, y1_norm))
                    x2_clamped = max(0, min(self.width - 1, x2_norm))
                    y2_clamped = max(0, min(self.height - 1, y2_norm))
                    temp_draw.ellipse(
                        [(x1_clamped, y1_clamped), (x2_clamped, y2_clamped)],
                        fill=preview_color,
                        outline=preview_color,
                    )
        finally:
            self.draw = original_draw
        
        # Centrer l'image si nécessaire
        self.center_image()
        
        # Afficher l'aperçu
        img_rgb = temp_image.convert('RGB')
        if self.zoom_level != 1.0:
            new_width = int(self.width * self.zoom_level)
            new_height = int(self.height * self.zoom_level)
            img_rgb = img_rgb.resize((new_width, new_height), Image.NEAREST)
        
        self.photo = ImageTk.PhotoImage(img_rgb)
        self.canvas.delete("all")
        self.canvas.create_image(
            self.pan_x,
            self.pan_y,
            anchor=tk.NW,
            image=self.photo
        )
        # Réafficher les indicateurs de curseur si nécessaire
        if self.tileable and not self.drawing and self.cursor_x is not None and self.cursor_y is not None:
            self.draw_cursor_indicators()
        self.canvas.config(scrollregion=self.canvas.bbox("all"))
    
    def set_zoom(self, zoom: float):
        """Définit le niveau de zoom."""
        self.zoom_level = max(0.1, min(10.0, zoom))
        self.update_canvas()
    
    def fit_to_window(self):
        """Ajuste l'image à la fenêtre."""
        # Forcer la mise à jour des dimensions du canevas
        self.root.update_idletasks()
        # Calculer le zoom pour que l'image rentre dans le canevas
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        
        if canvas_width > 1 and canvas_height > 1:
            zoom_x = canvas_width / self.width
            zoom_y = canvas_height / self.height
            self.zoom_level = min(zoom_x, zoom_y) * 0.9  # 90% pour laisser de la marge
            self.update_canvas()
        else:
            # Si les dimensions ne sont pas encore disponibles, réessayer après un court délai
            self.root.after(100, self.fit_to_window)


def main():
    """Point d'entrée principal."""
    root = tk.Tk()
    app = ConstructionEditor(root)
    root.mainloop()


if __name__ == "__main__":
    main()

