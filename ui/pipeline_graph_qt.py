# -*- coding: utf-8 -*-
"""
Visionneuse du graphe pipeline avec NodeGraphQt (Qt) — optionnelle.

Lanceur : python -m ui.pipeline_graph_qt <fichier.pkl> (depuis la racine du projet)
Le pickle contient un dict {"payload": ..., "opts": ...} : même snapshot que le moteur (**image de base à paliers**
+ **opts**) pour un rendu donné. Toute la logique affichée part de cette paire image/options.

Référence : https://github.com/jchanvfx/NodeGraphQt
"""
from __future__ import annotations

import os
import pickle
import sys
import time
from io import BytesIO
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from PIL import Image

# Avant tout import Qt : sur Windows, Qt peut journaliser qt.qpa.screen (échec WMI / DISPLAY1,
# erreur 0xe0000225) sans empêcher l’affichage — on réduit ce bruit.
os.environ.setdefault("QT_LOGGING_RULES", "qt.qpa.screen=false")


def _install_qt_graphics_noise_filter() -> None:
    """
    Filtre un avertissement Qt fréquent et bénin : ``QGraphicsItem::ungrabMouse: not a mouse grabber``.
    Il apparaît avec NodeGraphQt / QGraphicsView (fin de drag, pan, focus) sans indiquer un bug applicatif.
    """
    try:
        from Qt import QtCore

        if not hasattr(QtCore, "qInstallMessageHandler"):
            return
        state: Dict[str, Any] = {"prev": None}

        def _handler(msg_type: Any, context: Any, message: Any) -> None:
            if isinstance(message, (bytes, bytearray, memoryview)):
                text = bytes(message).decode("utf-8", errors="replace")
            else:
                text = str(message)
            if "ungrabMouse" in text and "not a mouse grabber" in text:
                return
            prev = state["prev"]
            if prev is not None:
                prev(msg_type, context, message)
            else:
                print(text, file=sys.stderr, end="")

        state["prev"] = QtCore.qInstallMessageHandler(_handler)
    except Exception:
        pass


def _composite_rgba_on_checkerboard(img: "Image.Image", cell: int = 8) -> "Image.Image":
    """
    Zones transparentes (alpha faible) : fond en damier puis fusion, pour lecture comme en interface graphique.
    Les masques debug RVB restent inchangés ; les calques RGBA (ex. après blanc/noir→α) montrent le damier.
    """
    from PIL import Image
    import numpy as np

    if img.mode != "RGBA":
        return img
    w, h = img.size
    fg = np.asarray(img, dtype=np.uint8)
    x = np.arange(w, dtype=np.int32)
    y = np.arange(h, dtype=np.int32)
    xv, yv = np.meshgrid(x, y)
    chk = ((xv // cell) + (yv // cell)) % 2
    bg_rgb = np.where(chk[..., None], np.array([240, 240, 240], dtype=np.uint8), np.array([195, 195, 195], dtype=np.uint8))
    a = fg[..., 3:4].astype(np.float32) / 255.0
    rgb = fg[..., :3].astype(np.float32)
    out = (rgb * a + bg_rgb.astype(np.float32) * (1.0 - a)).clip(0, 255).astype(np.uint8)
    return Image.fromarray(out, mode="RGB")


def _pil_to_qpixmap(pil_img: "Image.Image", max_edge: Optional[int] = 200) -> Any:
    """Convertit une image Pillow en QPixmap. Si ``max_edge`` est ``None``, aucune réduction."""
    from PIL import Image
    from Qt import QtGui

    img = pil_img.copy()
    if img.mode not in ("RGB", "RGBA"):
        img = img.convert("RGBA")
    if img.mode == "RGBA":
        img = _composite_rgba_on_checkerboard(img)
    w, h = img.size
    if max_edge is not None and max(w, h) > max_edge:
        r = max_edge / float(max(w, h))
        img = img.resize((max(1, int(w * r)), max(1, int(h * r))), Image.Resampling.LANCZOS)
        w, h = img.size
    buf = BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    pm = QtGui.QPixmap()
    if not pm.loadFromData(buf.read()):
        edge = max_edge if max_edge is not None else min(w, h, 512)
        pm = QtGui.QPixmap(max(1, min(w, edge)), max(1, min(h, edge)))
        pm.fill(QtGui.QColor(60, 60, 60))
    return pm


def _show_full_image_dialog(pil_image: "Image.Image", title: str) -> None:
    """Fenêtre modale avec défilement, pixmap à résolution native (pas de réduction).

    Parent ``None`` : un parent pris dans la scène NodeGraphQt peut provoquer des fermetures
    intempestives de la fenêtre principale à la fermeture du dialogue (hiérarchie Qt / WA_DeleteOnClose).
    """
    from Qt import QtCore, QtGui, QtWidgets

    try:
        dlg = QtWidgets.QDialog(None)
        dlg.setWindowTitle(title[:200])
        dlg.setModal(True)
        dlg.setWindowModality(QtCore.Qt.ApplicationModal)
        dlg.setAttribute(QtCore.Qt.WA_DeleteOnClose, True)

        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(False)
        scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
        scroll.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)

        inner = QtWidgets.QLabel()
        inner.setAlignment(QtCore.Qt.AlignCenter)
        pm = _pil_to_qpixmap(pil_image, max_edge=None)
        inner.setPixmap(pm)
        inner.setFixedSize(pm.size())
        scroll.setWidget(inner)

        btn = QtWidgets.QPushButton("Fermer")
        btn.clicked.connect(dlg.accept)

        outer = QtWidgets.QVBoxLayout(dlg)
        outer.addWidget(scroll)
        outer.addWidget(btn)

        screen = QtWidgets.QApplication.primaryScreen()
        if screen is not None:
            avail = screen.availableGeometry()
            margin = 80
            dlg_w = min(pm.width() + margin, avail.width() - 40)
            dlg_h = min(pm.height() + 120, avail.height() - 40)
            dlg.resize(max(400, dlg_w), max(300, dlg_h))
        else:
            dlg.resize(min(1400, pm.width() + 80), min(900, pm.height() + 120))

        dlg.exec()
    except Exception:
        import traceback

        traceback.print_exc(file=sys.stderr)


def run_viewer(payload: dict, opts: Optional[dict]) -> None:
    from NodeGraphQt import BaseNode, NodeGraph
    from NodeGraphQt.constants import NodePropWidgetEnum
    from NodeGraphQt.widgets.node_widgets import NodeBaseWidget
    from PIL import Image
    from Qt import QtCore, QtGui, QtWidgets

    class PreviewLabel(QtWidgets.QLabel):
        """
        Dans un QGraphicsProxyWidget (NodeGraphQt), ``mouseDoubleClickEvent`` est souvent absent.
        On détecte deux appuis gauche rapprochés (~double-clic) et on émet ``openRequested``.
        """

        openRequested = QtCore.Signal()

        def __init__(self, parent: Any = None) -> None:
            super().__init__(parent)
            self.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
            self._prev_press_t: Optional[float] = None

        def mousePressEvent(self, event: Any) -> None:
            if event.button() == QtCore.Qt.LeftButton:
                now = time.monotonic()
                if self._prev_press_t is not None and (now - self._prev_press_t) <= 0.5:
                    self.openRequested.emit()
                    self._prev_press_t = None
                    event.accept()
                    return
                self._prev_press_t = now
            else:
                self._prev_press_t = None
            super().mousePressEvent(event)

        def mouseDoubleClickEvent(self, event: Any) -> None:
            if event.button() == QtCore.Qt.LeftButton:
                self.openRequested.emit()
                self._prev_press_t = None
                event.accept()
                return
            super().mouseDoubleClickEvent(event)

    class NodeThumbnailWidget(NodeBaseWidget):
        """Vignette dans le nœud ; bouton + double-clic / deux clics pour l’aperçu pleine taille."""

        def __init__(
            self,
            parent: Any = None,
            name: str = "preview",
            label: str = "",
            pil_image: Optional[Image.Image] = None,
            max_edge: int = 200,
            dialog_title: str = "",
        ) -> None:
            super().__init__(parent, name, label)
            container = QtWidgets.QWidget()
            v = QtWidgets.QVBoxLayout(container)
            v.setContentsMargins(0, 0, 0, 4)
            v.setSpacing(4)

            if pil_image is not None:
                full = pil_image.copy()
                ttl = (dialog_title or "Image").strip() or "Aperçu"

                def _open() -> None:
                    _show_full_image_dialog(full, ttl)

                lab = PreviewLabel()
                lab.setAlignment(QtCore.Qt.AlignCenter)
                pm = _pil_to_qpixmap(pil_image, max_edge=max_edge)
                lab.setPixmap(pm)
                lab.setFixedSize(pm.size())
                lab.setToolTip(
                    "Deux clics gauche rapides sur l’image, ou double-clic si pris en charge ; "
                    "sinon utiliser le bouton ci-dessous."
                )
                lab.openRequested.connect(_open)

                btn = QtWidgets.QPushButton("Aperçu pleine taille")
                btn.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
                btn.setStyleSheet(
                    "QPushButton { font-size: 9pt; padding: 4px 8px; "
                    "background-color: #3c3c3c; color: #e0e0e0; border: 1px solid #555; border-radius: 3px; }"
                    "QPushButton:hover { background-color: #4a4a4a; }"
                )
                btn.setToolTip("Ouvre l’image sans réduction (fiabilité maximale dans NodeGraphQt).")
                btn.clicked.connect(_open)

                v.addWidget(lab)
                v.addWidget(btn)
            else:
                lab = QtWidgets.QLabel("(pas d’image)")
                lab.setMinimumSize(120, 72)
                v.addWidget(lab)

            self.set_custom_widget(container)

        def get_value(self) -> str:
            return ""

        def set_value(self, text: str = "") -> None:
            pass

    def _graph_node_body_text(n: Dict[str, Any]) -> str:
        parts: List[str] = []
        u = (n.get("user_text") or "").strip()
        if u:
            parts.append(u)
        on = (n.get("order_note") or "").strip()
        if on:
            parts.append(f"Note — {on}")
        return "\n\n".join(parts)

    class NodeExplanationWidget(NodeBaseWidget):
        """Bloc texte (nœuds sans vignette : options, sections, paramètres, cul-de-sac)."""

        def __init__(
            self,
            parent: Any = None,
            name: str = "explanation",
            text: str = "",
            max_h: int = 200,
        ) -> None:
            super().__init__(parent, name, "")
            te = QtWidgets.QTextEdit()
            te.setReadOnly(True)
            te.setPlainText(text.strip() or "—")
            te.setMaximumHeight(max_h)
            te.setMaximumWidth(340)
            te.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
            te.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
            te.setStyleSheet(
                "QTextEdit { background-color:#252526; color:#d4d4d4; "
                "font-size:9pt; border:1px solid #3c3c3c; border-radius:2px; padding:4px; }"
            )
            self.set_custom_widget(te)

        def get_value(self) -> str:
            return ""

        def set_value(self, text: str = "") -> None:
            pass

    class PipelineStepNode(BaseNode):
        __identifier__ = "halo.forerunner"
        NODE_NAME = "Étape"

        def __init__(self) -> None:
            super().__init__()
            # NodeGraphQt : sans multi_input=True, un seul fil par port d’entrée (hub n_opts, fusions, etc.).
            self.add_input("in", multi_input=True)
            self.add_output("out")

    class PipelineInputNode(BaseNode):
        """Nœud visuel pour canaux d’entrée (image de base, texture métal, paramètres étage)."""

        __identifier__ = "halo.forerunner"
        NODE_NAME = "Entrée"

        def __init__(self) -> None:
            super().__init__()
            self.add_input("in", multi_input=True)
            self.add_output("out")

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    _install_qt_graphics_noise_filter()

    from .pipeline_graph import graph_edges_from_nodes, layout_pipeline_graph, list_graph_nodes

    nodes_spec = list_graph_nodes(payload, opts or {})
    # NodeGraphQt ne fournit pas d’auto-layout : tout est dans ``layout_pipeline_graph`` (pipeline_graph.py).
    # Ajuster au besoin : linear_chain_gap, fan_section_gap, branch_gap_from_parent, mask_row_dy, etc. (défaut 500 px).
    positions, _sizes = layout_pipeline_graph(nodes_spec)
    edges = graph_edges_from_nodes(nodes_spec)

    graph = NodeGraph()
    graph.register_node(PipelineStepNode)
    graph.register_node(PipelineInputNode)

    type_step = f"{PipelineStepNode.__identifier__}.{PipelineStepNode.__name__}"
    type_input = f"{PipelineInputNode.__identifier__}.{PipelineInputNode.__name__}"
    # Vert Forerunner : entrées logiciel (image source, TM, réglages par étage)
    _input_rgb = (46, 125, 50)

    qt_nodes: Dict[str, Any] = {}
    spec_by_id: Dict[str, Dict[str, Any]] = {n["id"]: n for n in nodes_spec}

    for n in nodes_spec:
        nid = n["id"]
        title = (n.get("title") or nid)[:80]
        is_input = bool(n.get("input_channel"))
        qn = graph.create_node(type_input if is_input else type_step, name=title)
        if is_input:
            qn.set_color(*_input_rgb)
        pos = positions.get(nid, (0.0, 0.0))
        qn.set_pos(pos[0], pos[1])
        qt_nodes[nid] = qn
        im = n.get("image") if n.get("type") == "image" else None
        if im is not None:
            tw = NodeThumbnailWidget(
                qn.view,
                f"thumb_{nid}",
                "",
                im,
                max_edge=200,
                dialog_title=(n.get("title") or nid),
            )
            qn.add_custom_widget(tw, widget_type=NodePropWidgetEnum.HIDDEN.value)
        else:
            body = _graph_node_body_text(n)
            if body:
                typ = (n.get("type") or "").lower()
                if typ == "code":
                    max_h = 280 if nid == "n_opts" else 260
                elif typ == "section":
                    max_h = 150
                elif typ == "dead":
                    max_h = 175
                else:
                    max_h = 200
                ew = NodeExplanationWidget(
                    qn.view,
                    f"txt_{nid}",
                    body,
                    max_h=max_h,
                )
                qn.add_custom_widget(ew, widget_type=NodePropWidgetEnum.HIDDEN.value)

    for src_id, dst_id, _kind in edges:
        if src_id not in qt_nodes or dst_id not in qt_nodes:
            continue
        try:
            qt_nodes[src_id].output(0).connect_to(qt_nodes[dst_id].input(0))
        except Exception:
            pass

    w = graph.widget
    def _nodes_positions_as_text() -> str:
        rows: List[str] = []
        rows.append("id\tname\tx\ty")
        for nid in sorted(qt_nodes.keys()):
            qn = qt_nodes[nid]
            name = (spec_by_id.get(nid, {}).get("title") or nid).replace("\t", " ").replace("\n", " ").strip()
            try:
                x = float(qn.x_pos())
                y = float(qn.y_pos())
            except Exception:
                px, py = positions.get(nid, (0.0, 0.0))
                x, y = float(px), float(py)
            rows.append(f"{nid}\t{name}\t{int(round(x))}\t{int(round(y))}")
        return "\n".join(rows)

    def _copy_nodes_positions_to_clipboard() -> None:
        txt = _nodes_positions_as_text()
        cb = QtWidgets.QApplication.clipboard()
        cb.setText(txt)
        QtWidgets.QMessageBox.information(
            w,
            "Positions copiées",
            "La liste des nœuds (id, nom, x, y) a été copiée dans le presse-papiers.",
        )

    copy_btn = QtWidgets.QPushButton("Copier positions des nœuds", w)
    copy_btn.setToolTip("Copie id, nom, x, y de tous les nœuds.")
    copy_btn.clicked.connect(_copy_nodes_positions_to_clipboard)

    shortcut_copy = QtGui.QShortcut(QtGui.QKeySequence("Ctrl+Shift+C"), w)
    shortcut_copy.activated.connect(_copy_nodes_positions_to_clipboard)

    def _place_copy_button() -> None:
        bw = copy_btn.sizeHint().width() + 16
        bh = copy_btn.sizeHint().height() + 8
        copy_btn.resize(bw, bh)
        x = max(8, w.width() - bw - 12)
        y = 8
        copy_btn.move(x, y)
        copy_btn.raise_()

    _orig_resize = w.resizeEvent
    def _resize_event(ev: Any) -> None:
        _orig_resize(ev)
        _place_copy_button()
    w.resizeEvent = _resize_event

    w.setWindowTitle(f"Pipeline Forerunner — NodeGraphQt ({len(qt_nodes)} nœuds)")
    w.resize(1200, 820)
    _place_copy_button()
    w.show()
    for qn in qt_nodes.values():
        qn.set_selected(True)
    did_fit = False
    try:
        graph.fit_to_selection()
        did_fit = True
    except Exception:
        did_fit = False
    if not did_fit:
        try:
            view = graph.viewer()
            if view is not None and view.scene() is not None:
                rect = view.scene().itemsBoundingRect()
                if rect.isValid():
                    view.fitInView(rect.adjusted(-80, -80, 80, 80), QtCore.Qt.KeepAspectRatio)
        except Exception:
            pass
    for qn in qt_nodes.values():
        qn.set_selected(False)

    app.exec()


def main(argv: Optional[List[str]] = None) -> int:
    argv = argv if argv is not None else sys.argv
    if len(argv) < 2:
        print("Usage: python -m ui.pipeline_graph_qt <chemin.pkl>", file=sys.stderr)
        return 1
    path = argv[1]
    try:
        with open(path, "rb") as f:
            bundle = pickle.load(f)
        payload = bundle["payload"]
        opts = bundle.get("opts")
        run_viewer(payload, opts)
    except Exception:
        import traceback

        traceback.print_exc(file=sys.stderr)
        return 1
    finally:
        try:
            os.remove(path)
        except OSError:
            pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
