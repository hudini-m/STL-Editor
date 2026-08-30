import numpy as np
from PySide6.QtWidgets import QLabel, QVBoxLayout, QWidget

try:
    import pyvista as pv
    from pyvistaqt import QtInteractor
except Exception:  # pragma: no cover
    pv = None
    QtInteractor = None


class Viewer3DWidget(QWidget):
    """Read-only inspection view. It deliberately has no mesh edit handlers."""

    def __init__(self):
        super().__init__()
        self.plotter = None
        self.actor = None
        layout = QVBoxLayout(self)
        if pv is not None and QtInteractor is not None:
            self.plotter = QtInteractor(self)
            self.plotter.set_background("#eef3f7")
            layout.addWidget(self.plotter)
        else:
            layout.addWidget(QLabel("PyVistaQt is not available in this environment."))

    def set_mesh(self, vertices, faces):
        if self.plotter is None:
            return
        if self.actor is not None:
            try:
                self.plotter.remove_actor(self.actor)
            except Exception:
                pass
        faces = np.asarray(faces, dtype=int)
        cells = np.hstack([np.full((len(faces), 1), faces.shape[1], dtype=int), faces]).ravel()
        mesh = pv.PolyData(np.asarray(vertices, dtype=float), cells)
        self.actor = self.plotter.add_mesh(mesh, color="#73a9c2", smooth_shading=True)
        self.plotter.view_isometric()
        self.plotter.reset_camera()
        self.plotter.render()
