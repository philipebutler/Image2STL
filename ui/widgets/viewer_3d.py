"""
3D model viewer widget - interactive wireframe viewer with rotate/zoom.
"""
from __future__ import annotations

import math
from pathlib import Path

from PySide6.QtCore import Qt, QPoint
from PySide6.QtGui import QPainter, QPen, QColor, QVector3D
from PySide6.QtWidgets import QWidget


# Unit-cube vertices (8 corners)
_CUBE_VERTICES = [
    QVector3D(-1, -1, -1),
    QVector3D(1, -1, -1),
    QVector3D(1, 1, -1),
    QVector3D(-1, 1, -1),
    QVector3D(-1, -1, 1),
    QVector3D(1, -1, 1),
    QVector3D(1, 1, 1),
    QVector3D(-1, 1, 1),
]

# 12 edges of the cube (index pairs)
_CUBE_EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 0),
    (4, 5), (5, 6), (6, 7), (7, 4),
    (0, 4), (1, 5), (2, 6), (3, 7),
]


def _rotate_y(v: QVector3D, angle: float) -> QVector3D:
    c, s = math.cos(angle), math.sin(angle)
    return QVector3D(c * v.x() + s * v.z(), v.y(), -s * v.x() + c * v.z())


def _rotate_x(v: QVector3D, angle: float) -> QVector3D:
    c, s = math.cos(angle), math.sin(angle)
    return QVector3D(v.x(), c * v.y() - s * v.z(), s * v.y() + c * v.z())


class Viewer3D(QWidget):
    """Interactive 3D wireframe viewer.

    Shows a rotating wireframe cube placeholder.  When an actual mesh file is
    loaded in a future iteration the cube will be replaced by the model geometry.

    Controls:
    - Left-click drag: rotate
    - Scroll wheel: zoom
    """

    def __init__(self, config=None, parent=None):
        super().__init__(parent)
        self._config = config
        self._yaw = 0.5
        self._pitch = 0.3
        self._zoom = 1.0
        self._dragging = False
        self._last_pos: QPoint = QPoint()
        self._vertices: list[QVector3D] = list(_CUBE_VERTICES)
        self._edges: list[tuple[int, int]] = list(_CUBE_EDGES)
        self.setMinimumSize(300, 300)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

    def reset_placeholder(self):
        self._vertices = list(_CUBE_VERTICES)
        self._edges = list(_CUBE_EDGES)
        self._yaw = 0.5
        self._pitch = 0.3
        self._zoom = 1.0
        self.update()

    def load_model(self, model_path: str | Path) -> bool:
        try:
            import trimesh  # type: ignore

            loaded = trimesh.load(str(model_path), force="mesh")
            if isinstance(loaded, trimesh.Scene):
                mesh = loaded.dump(concatenate=True)
            elif isinstance(loaded, (list, tuple)):
                mesh = loaded[0] if loaded else None
            else:
                mesh = loaded

            if mesh is None:
                self.reset_placeholder()
                return False

            raw_vertices = getattr(mesh, "vertices", None)
            raw_faces = getattr(mesh, "faces", None)
            if raw_vertices is None or raw_faces is None:
                self.reset_placeholder()
                return False

            verts = [tuple(float(c) for c in row[:3]) for row in raw_vertices]
            if not verts:
                self.reset_placeholder()
                return False

            min_x = min(v[0] for v in verts)
            max_x = max(v[0] for v in verts)
            min_y = min(v[1] for v in verts)
            max_y = max(v[1] for v in verts)
            min_z = min(v[2] for v in verts)
            max_z = max(v[2] for v in verts)

            center_x = (min_x + max_x) / 2.0
            center_y = (min_y + max_y) / 2.0
            center_z = (min_z + max_z) / 2.0
            span = max(max_x - min_x, max_y - min_y, max_z - min_z, 1e-9)
            norm = 2.0 / span

            self._vertices = [
                QVector3D(
                    (vx - center_x) * norm,
                    (vy - center_y) * norm,
                    (vz - center_z) * norm,
                )
                for vx, vy, vz in verts
            ]

            edge_set: set[tuple[int, int]] = set()
            for face in raw_faces:
                face_indices = [int(idx) for idx in face]
                if len(face_indices) < 2:
                    continue
                for i in range(len(face_indices)):
                    a = face_indices[i]
                    b = face_indices[(i + 1) % len(face_indices)]
                    if a == b:
                        continue
                    edge_set.add((a, b) if a < b else (b, a))

            self._edges = sorted(edge_set)
            if not self._edges:
                self.reset_placeholder()
                return False

            if len(self._edges) > 50000:
                self._edges = self._edges[:50000]

            self._yaw = 0.5
            self._pitch = 0.3
            self._zoom = 1.0
            self.update()
            return True
        except Exception:
            self.reset_placeholder()
            return False

    # ------------------------------------------------------------------
    # Painting
    # ------------------------------------------------------------------

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), QColor("#171717"))

        w, h = self.width(), self.height()
        if w < 2 or h < 2:
            return

        cx, cy = w / 2, h / 2
        scale = min(w, h) * 0.33 * self._zoom

        projected = []
        for v in self._vertices:
            v = _rotate_y(v, self._yaw)
            v = _rotate_x(v, self._pitch)
            z = v.z() + 4.0
            s = scale / z if z != 0 else scale
            projected.append((cx + v.x() * s, cy - v.y() * s))

        pen = QPen(QColor("#00bfff"), 2)
        painter.setPen(pen)
        for a, b in self._edges:
            if a < 0 or b < 0 or a >= len(projected) or b >= len(projected):
                continue
            painter.drawLine(
                int(projected[a][0]), int(projected[a][1]),
                int(projected[b][0]), int(projected[b][1]),
            )

    # ------------------------------------------------------------------
    # Mouse interaction
    # ------------------------------------------------------------------

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self._dragging = True
            self._last_pos = event.position().toPoint()
            event.accept()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self._dragging = False
            event.accept()

    def mouseMoveEvent(self, event):
        if not self._dragging:
            return
        pos = event.position().toPoint()
        delta = pos - self._last_pos
        self._last_pos = pos
        self._yaw += delta.x() * 0.01
        self._pitch += delta.y() * 0.01
        self.update()
        event.accept()

    def wheelEvent(self, event):
        delta = event.angleDelta().y() / 120.0  # steps
        self._zoom = max(0.4, min(4.0, self._zoom + delta * 0.1))
        self.update()
        event.accept()
