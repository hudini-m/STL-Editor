import sys
from pathlib import Path

import numpy as np
from PySide6.QtCore import QEvent, Qt, Signal
from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QSlider,
    QLabel,
    QFrame,
    QMessageBox,
    QComboBox,
)

sys.path.insert(0, str(Path(__file__).parent))

try:
    import pyvista as pv
except Exception:  # pragma: no cover
    pv = None

try:
    from pyvistaqt import QtInteractor
except Exception:  # pragma: no cover
    QtInteractor = None

try:
    from vtkmodules.vtkRenderingAnnotation import vtkLegendScaleActor
except Exception:  # pragma: no cover
    vtkLegendScaleActor = None

import trimesh

from geometry.mesh_model import MeshModel
from geometry.deformation import DeformationEngine, apply_plane_displacement_field
from geometry.mesh_validation import format_mesh_validation_report
from geometry.projection import lift_points, plane_normal_axis, project_points
from interaction.control_points import ControlPointManager
from interaction.picking import PickingManager


class ViewportWidget(QWidget):
    mesh_loaded = Signal(dict)
    mesh_changed = Signal(object, object)

    def __init__(self):
        super().__init__()
        self.setAcceptDrops(True)

        layout = QVBoxLayout(self)
        self.plotter = None
        self.mesh_actor = None
        self.contour_actor = None
        self.ruler_actor = None
        self.grid_actor = None
        self.grid_enabled = True
        self.control_point_actors = {}
        self.control_point_meshes = {}
        self.mesh_polydata = None
        self.mesh_model = None
        self.picking_manager = None
        self.deformation_engine = None
        self.control_point_manager = None
        self.mesh = None
        self.original_mesh = None
        self.history = []
        self.history_index = -1
        self.is_wireframe = False
        self._mouse_drag_index = None
        self._mouse_drag_changed = False
        self.movement_plane_locked = False
        self.local_neighbor_count = 4
        self.edit_mode = "Smooth contour"
        self.influence_radius_percent = 20
        self.edit_strength_percent = 50
        self.falloff_type = "Smoothstep"

        if pv is not None and QtInteractor is not None:
            self.plotter = QtInteractor(self)
            layout.addWidget(self.plotter)
            self.plotter.installEventFilter(self)
            self.plotter.set_background("white")
            try:
                self.plotter.enable_anti_aliasing()
            except Exception:
                pass
        else:
            self.plotter = None
            layout.addWidget(QLabel("PyVistaQt is not available in this environment."))

        control_frame = QFrame()
        control_frame.setFrameStyle(QFrame.StyledPanel)
        control_layout = QHBoxLayout(control_frame)
        layout.addWidget(control_frame)
        self.add_controls(control_layout)

    def add_controls(self, layout):
        for text, func in [
            ("Fit 3D", self.fit_to_view),
            ("XY", lambda: self.set_movement_plane("XY")),
            ("YZ", lambda: self.set_movement_plane("YZ")),
            ("XZ", lambda: self.set_movement_plane("XZ")),
        ]:
            btn = QPushButton(text)
            btn.clicked.connect(func)
            layout.addWidget(btn)

        self.grid_btn = QPushButton("Grid")
        self.grid_btn.setCheckable(True)
        self.grid_btn.setChecked(True)
        self.grid_btn.toggled.connect(self.toggle_grid)
        layout.addWidget(self.grid_btn)

        layout.addWidget(QLabel("Transparency"))
        self.transparency_slider = QSlider(Qt.Horizontal)
        self.transparency_slider.setRange(0, 100)
        self.transparency_slider.setValue(70)
        self.transparency_slider.valueChanged.connect(self.update_transparency)
        layout.addWidget(self.transparency_slider)

        self.wireframe_btn = QPushButton("Wireframe")
        self.wireframe_btn.setCheckable(True)
        self.wireframe_btn.clicked.connect(self.toggle_wireframe)
        layout.addWidget(self.wireframe_btn)

        validate_btn = QPushButton("Validate Mesh")
        validate_btn.clicked.connect(self.validate_mesh)
        layout.addWidget(validate_btn)

        layout.addWidget(QLabel("Selection Mode:"))
        self.selection_combo = QComboBox()
        self.selection_combo.addItems(["Vertex", "Face", "Region"])
        layout.addWidget(self.selection_combo)

        layout.addWidget(QLabel("Movement Plane:"))
        self.movement_plane_combo = QComboBox()
        self.movement_plane_combo.addItems(["XY", "YZ", "XZ"])
        self.movement_plane_combo.currentTextChanged.connect(self.set_movement_plane)
        layout.addWidget(self.movement_plane_combo)

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event):
        for url in event.mimeData().urls():
            file_path = url.toLocalFile()
            if file_path.lower().endswith(".stl"):
                self.load_mesh(file_path)

    def _load_trimesh(self, file_path):
        loaded = trimesh.load(file_path, force="mesh")
        if isinstance(loaded, trimesh.Scene):
            loaded = trimesh.util.concatenate(tuple(loaded.dump()))
        if not isinstance(loaded, trimesh.Trimesh):
            raise ValueError("Unsupported mesh file")
        return loaded

    def load_sample_mesh(self):
        """Load a generated cube so the editing workflow is available immediately."""
        try:
            sample = trimesh.creation.box(extents=(40.0, 28.0, 18.0))
            self._load_mesh_data(sample, "sample_cube.stl")
        except Exception as exc:
            QMessageBox.critical(self, "Error", f"Failed to create sample mesh:\n{exc}")

    def _load_mesh_data(self, loaded, display_name):
        vertices = np.asarray(loaded.vertices, dtype=float)
        faces = np.asarray(loaded.faces, dtype=int)
        self.mesh_model = MeshModel(vertices, faces)
        self.mesh_model.load_from_trimesh(loaded)
        self.picking_manager = PickingManager(self.mesh_model)
        self.deformation_engine = DeformationEngine(self.mesh_model)
        self.control_point_manager = ControlPointManager(self.mesh_model)

        if pv is not None:
            self.mesh_polydata = pv.PolyData(vertices, self._faces_to_pyvista(faces))
            self.original_mesh = self.mesh_polydata.copy()
            self.mesh = self.mesh_polydata
            self._apply_mesh_to_plotter()

        self._push_history()
        self.mesh_loaded.emit({"name": display_name, "vertices": len(vertices), "faces": len(faces)})
        self.mesh_changed.emit(self.mesh_model.vertices.copy(), self.mesh_model.faces.copy())

    def _faces_to_pyvista(self, faces):
        if len(faces) == 0:
            return np.empty((0,), dtype=int)
        faces = np.asarray(faces, dtype=int)
        counts = np.full((faces.shape[0], 1), faces.shape[1], dtype=int)
        return np.hstack([counts, faces]).ravel()

    def _apply_mesh_to_plotter(self):
        if self.plotter is None or self.mesh_polydata is None:
            return
        if self.mesh_actor is not None:
            try:
                self.plotter.remove_actor(self.mesh_actor)
            except Exception:
                pass
        self.mesh_actor = self.plotter.add_mesh(
            self.mesh_polydata,
            color="lightblue",
            smooth_shading=True,
            opacity=self.transparency_slider.value() / 100.0,
        )
        if self.is_wireframe and self.mesh_actor is not None:
            try:
                self.mesh_actor.GetProperty().SetRepresentationToWireframe()
            except Exception:
                pass
        self.fit_to_view()
        self.plotter.render()

    def _enable_control_point_picking(self):
        return

    def _on_point_picked(self, picked_point):
        if self.control_point_manager is None:
            return
        if picked_point is None or self.mesh_model is None:
            return
        points = self.control_point_manager.get_control_points()
        if not points:
            return
        picked_point = np.asarray(picked_point, dtype=float)
        distances = np.linalg.norm(np.asarray(points) - picked_point, axis=1)
        nearest = int(np.argmin(distances))
        bounds = self.mesh_model.get_bounds()
        scale = max(float(np.linalg.norm(bounds[1] - bounds[0])), 1.0)
        selection_radius = scale * 0.12
        selected = self.control_point_manager.get_selected_point()
        if selected is not None and distances[selected] > selection_radius:
            self._move_selected_control_point(picked_point)
        elif distances[nearest] <= selection_radius:
            self.control_point_manager.select_point(nearest)
            self._refresh_control_point_markers()
            self.status_message(f"Selected control point {nearest + 1}")

    def _nearest_control_point(self, picked_point):
        points = self.control_point_manager.get_control_points()
        if not points:
            return None, float("inf")
        distances = np.linalg.norm(np.asarray(points) - np.asarray(picked_point, dtype=float), axis=1)
        return int(np.argmin(distances)), float(np.min(distances))

    def _screen_to_plane_point(self, event):
        """Map a Qt mouse coordinate to the orthographic plane at camera focal depth."""
        try:
            renderer = self.plotter.renderer
            focal = self.plotter.camera.GetFocalPoint()
            renderer.SetWorldPoint(*focal, 1.0)
            renderer.WorldToDisplay()
            depth = renderer.GetDisplayPoint()[2]
            position = event.position()
            renderer.SetDisplayPoint(position.x(), self.plotter.height() - position.y(), depth)
            renderer.DisplayToWorld()
            world = np.asarray(renderer.GetWorldPoint(), dtype=float)
            return world[:3] / world[3] if abs(world[3]) > 1e-12 else world[:3]
        except Exception:
            return self._pick_mouse_position()

    def _nearest_control_point_at_screen(self, event, radius=14.0):
        if self.control_point_manager is None or not self.control_point_manager.control_points:
            return None
        try:
            renderer = self.plotter.renderer
            plane = self.movement_plane_combo.currentText()
            points = project_points(self.control_point_manager.get_control_points(), plane)
            points_3d = lift_points(points, self._overlay_depth(), plane)
            mouse = event.position()
            best_index = None
            best_distance = float("inf")
            for index, point in enumerate(points_3d):
                renderer.SetWorldPoint(*point, 1.0)
                renderer.WorldToDisplay()
                display = renderer.GetDisplayPoint()
                distance = float(np.hypot(display[0] - mouse.x(), self.plotter.height() - display[1] - mouse.y()))
                if distance < best_distance:
                    best_index, best_distance = index, distance
            return best_index if best_distance <= radius else None
        except Exception:
            return None

    def eventFilter(self, watched, event):
        """Provide reliable marker dragging through Qt when VTK picking is unavailable."""
        if watched is self.plotter and self.plotter is not None and self.movement_plane_locked:
            if event.type() == QEvent.MouseButtonPress and event.button() == Qt.LeftButton:
                index = self._nearest_control_point_at_screen(event)
                if index is not None:
                    self.control_point_manager.select_point(index)
                    if event.modifiers() & Qt.ShiftModifier:
                        self.control_point_manager.toggle_anchor(index)
                        self._push_history()
                    elif not self.control_point_manager.is_point_pinned(index):
                        self._mouse_drag_index = index
                    self._refresh_control_point_markers()
                    return True
                if self.movement_plane_locked:
                    return True

            elif event.type() == QEvent.MouseMove and self._mouse_drag_index is not None:
                picked = self._screen_to_plane_point(event)
                if picked is not None:
                    changed = self._move_selected_control_point(picked, record_history=False)
                    self._mouse_drag_changed = self._mouse_drag_changed or changed
                    return False

            elif event.type() == QEvent.MouseMove and self.movement_plane_locked:
                # Keep 2D navigation available: VTK uses middle drag for pan
                # and right drag for zoom. Only rotation is blocked.
                if event.buttons() & (Qt.MiddleButton | Qt.RightButton):
                    return False
                return True

            elif event.type() == QEvent.MouseButtonRelease and event.button() == Qt.LeftButton:
                if self._mouse_drag_changed:
                    self._push_history()
                self._mouse_drag_index = None
                self._mouse_drag_changed = False

            elif event.type() == QEvent.MouseButtonPress and self.movement_plane_locked:
                if event.button() in (Qt.MiddleButton, Qt.RightButton):
                    return False
                return True

        return super().eventFilter(watched, event)

    def _pick_mouse_position(self):
        try:
            picked = self.plotter.pick_mouse_position()
            if picked is None:
                return None
            picked = np.asarray(picked, dtype=float)
            return picked if np.all(np.isfinite(picked)) else None
        except Exception:
            return None

    def _constrain_to_view_plane(self, old_position, new_position):
        """Keep movement in the selected 2D plane by preserving its normal axis."""
        constrained = np.asarray(new_position, dtype=float).copy()
        plane = self.movement_plane_combo.currentText() if hasattr(self, "movement_plane_combo") else "XY"
        depth_axis = {"XY": 2, "YZ": 0, "XZ": 1}.get(plane, 2)
        constrained[depth_axis] = old_position[depth_axis]
        return constrained

    def _move_selected_control_point(self, new_position, record_history=True):
        index = self.control_point_manager.get_selected_point()
        if index is None or self.control_point_manager.is_point_pinned(index):
            return False

        old_position = np.asarray(self.control_point_manager.get_control_points()[index], dtype=float)
        new_position = self._constrain_to_view_plane(old_position, new_position)
        displacement = new_position - old_position
        if np.linalg.norm(displacement) < 1e-8:
            return False

        vertices = np.asarray(self.mesh_model.vertices, dtype=float)
        source_points = np.asarray(self.control_point_manager.get_control_points(), dtype=float)
        move_method = (
            self.control_point_manager.move_point_exact
            if self.edit_mode == "Point-wise"
            else self.control_point_manager.move_point_locally
        )
        moved = (
            move_method(index, new_position)
            if self.edit_mode == "Point-wise"
            else move_method(index, new_position, neighbor_count=self.local_neighbor_count)
        )
        if not moved:
            self.status_message("Drag rejected: contour would become invalid")
            return False
        target_points = np.asarray(self.control_point_manager.get_control_points(), dtype=float)
        bounds = self.mesh_model.get_bounds()
        plane = self.movement_plane_combo.currentText()
        plane_axes = {"XY": (0, 1), "XZ": (0, 2), "YZ": (1, 2)}[plane]
        plane_span = float(np.linalg.norm((bounds[1] - bounds[0])[list(plane_axes)]))
        support_radius = max(plane_span * self.influence_radius_percent / 100.0, 0.001)
        if self.edit_mode == "Point-wise":
            projected_vertices = project_points(vertices, plane)
            nearest_vertex = int(np.argmin(np.linalg.norm(projected_vertices - project_points(old_position, plane), axis=1)))
            first_axis, second_axis, _ = (0, 1, 2) if plane == "XY" else (0, 2, 1) if plane == "XZ" else (1, 2, 0)
            vertices[nearest_vertex, first_axis] += displacement[first_axis] * self.edit_strength_percent / 100.0
            vertices[nearest_vertex, second_axis] += displacement[second_axis] * self.edit_strength_percent / 100.0
        else:
            target_points = source_points + (target_points - source_points) * self.edit_strength_percent / 100.0
            vertices = apply_plane_displacement_field(vertices, source_points, target_points, plane, support_radius)

        self.mesh_model.vertices = vertices
        if self.mesh_polydata is not None:
            self.mesh_polydata.points = vertices
        self.mesh_changed.emit(self.mesh_model.vertices.copy(), self.mesh_model.faces.copy())
        if record_history:
            self._push_history()
            self._refresh_grid_actor()
            self._refresh_contour_actor()
            self._refresh_control_point_markers()
            self.status_message(f"Moved control point {index + 1} and deformed mesh")
        else:
            self._refresh_grid_actor()
            self._refresh_contour_actor()
            self._refresh_control_point_markers()
        return True

    def _remove_control_point_actors(self):
        if self.plotter is None:
            return
        for actor in self.control_point_actors.values():
            try:
                self.plotter.remove_actor(actor)
            except Exception:
                pass
        self.control_point_actors = {}
        self.control_point_meshes = {}

    def _overlay_depth(self):
        bounds = self.mesh_model.get_bounds()
        axis = plane_normal_axis(self.movement_plane_combo.currentText())
        extent = max(float(np.linalg.norm(bounds[1] - bounds[0])), 1.0)
        return bounds[1, axis] + extent * 0.015

    def _refresh_contour_actor(self):
        if self.plotter is None or self.control_point_manager is None or pv is None:
            return
        if self.contour_actor is not None:
            try:
                self.plotter.remove_actor(self.contour_actor)
            except Exception:
                pass
            self.contour_actor = None
        contour = self.control_point_manager.contour_2d
        if not self.movement_plane_locked or len(contour) < 2:
            return
        plane = self.movement_plane_combo.currentText()
        points_3d = lift_points(contour, self._overlay_depth(), plane)
        points_3d = np.vstack([points_3d, points_3d[0]])
        polyline = pv.PolyData(points_3d)
        polyline.lines = np.hstack([[len(points_3d)], np.arange(len(points_3d), dtype=int)])
        self.contour_actor = self.plotter.add_mesh(
            polyline,
            name="outer_silhouette",
            color="#0f766e",
            line_width=3,
            render_lines_as_tubes=False,
            pickable=False,
        )

    def _remove_grid_actor(self):
        if self.grid_actor is None or self.plotter is None:
            return
        try:
            self.plotter.remove_actor(self.grid_actor)
        except Exception:
            pass
        self.grid_actor = None

    @staticmethod
    def _grid_step(span):
        """Choose a readable 1/2/5 grid step with roughly ten divisions."""
        target = max(float(span) / 10.0, 1e-9)
        magnitude = 10.0 ** np.floor(np.log10(target))
        normalized = target / magnitude
        multiplier = 1.0 if normalized <= 1.0 else 2.0 if normalized <= 2.0 else 5.0 if normalized <= 5.0 else 10.0
        return multiplier * magnitude

    def _refresh_grid_actor(self):
        """Draw a non-pickable model-space grid behind the active 2D plane."""
        self._remove_grid_actor()
        if (
            self.plotter is None
            or pv is None
            or self.mesh_model is None
            or not self.movement_plane_locked
            or not self.grid_enabled
        ):
            return

        try:
            plane = self.movement_plane_combo.currentText()
            projected = project_points(self.mesh_model.vertices, plane)
            minimum = np.min(projected, axis=0)
            maximum = np.max(projected, axis=0)
            span = maximum - minimum
            padding = max(float(np.max(span)) * 0.08, 1e-6)
            lower = minimum - padding
            upper = maximum + padding
            step = self._grid_step(max(float(np.max(span)), 1e-6))

            first_x = np.floor(lower[0] / step) * step
            last_x = np.ceil(upper[0] / step) * step
            first_y = np.floor(lower[1] / step) * step
            last_y = np.ceil(upper[1] / step) * step
            x_values = np.arange(first_x, last_x + step * 0.5, step)
            y_values = np.arange(first_y, last_y + step * 0.5, step)

            segments = []
            for x_value in x_values:
                segments.append([[x_value, first_y], [x_value, last_y]])
            for y_value in y_values:
                segments.append([[first_x, y_value], [last_x, y_value]])
            if not segments:
                return

            bounds = self.mesh_model.get_bounds()
            normal_axis = plane_normal_axis(plane)
            depth = bounds[0, normal_axis] - max(float(np.linalg.norm(bounds[1] - bounds[0])) * 0.01, 1e-6)
            points_2d = np.asarray(segments, dtype=float).reshape(-1, 2)
            grid = pv.PolyData(lift_points(points_2d, depth, plane))
            line_count = len(segments)
            grid.lines = np.column_stack(
                [np.full(line_count, 2, dtype=int), np.arange(line_count * 2, dtype=int).reshape(line_count, 2)]
            ).ravel()
            self.grid_actor = self.plotter.add_mesh(
                grid,
                name="editing_grid",
                color="#cbd5e1",
                opacity=0.8,
                line_width=1,
                render_lines_as_tubes=False,
                pickable=False,
            )
        except Exception:
            self.grid_actor = None

    def toggle_grid(self, checked=False):
        self.grid_enabled = bool(checked)
        self._refresh_grid_actor()
        if self.movement_plane_locked:
            self.status_message("2D grid shown" if self.grid_enabled else "2D grid hidden")

    def _refresh_control_point_markers(self):
        if self.plotter is None or self.control_point_manager is None or pv is None:
            return

        self._remove_control_point_actors()
        if not self.movement_plane_locked:
            self.plotter.render()
            return
        points = self.control_point_manager.control_points
        if not points:
            self.plotter.render()
            return

        selected = self.control_point_manager.get_selected_point()
        active = set(self.control_point_manager.get_active_region(selected, self.local_neighbor_count)) if selected is not None else set()
        groups = {"selected": [], "pinned": [], "active": [], "default": []}
        plane = self.movement_plane_combo.currentText()
        overlay_depth = self._overlay_depth()
        projected = project_points([position for position, _ in points], plane)
        for index, (_, pinned) in enumerate(points):
            position = lift_points(projected[index], overlay_depth, plane)
            if index == selected:
                groups["selected"].append(position)
            elif pinned:
                groups["pinned"].append(position)
            elif index in active:
                groups["active"].append(position)
            else:
                groups["default"].append(position)

        point_size = 9.0
        colors = {"default": "#f59e0b", "selected": "#ef4444", "pinned": "#2563eb", "active": "#8b5cf6"}
        for name, positions in groups.items():
            if not positions:
                continue
            marker_mesh = pv.PolyData(np.asarray(positions, dtype=float))
            self.control_point_meshes[name] = marker_mesh
            self.control_point_actors[name] = self.plotter.add_mesh(
                marker_mesh,
                name=f"control_points_{name}",
                color=colors[name],
                point_size=point_size,
                render_points_as_spheres=True,
                pickable=True,
            )
        self.plotter.render()

    def load_mesh(self, file_path):
        try:
            loaded = self._load_trimesh(file_path)
            self._load_mesh_data(loaded, Path(file_path).name)
        except Exception as exc:
            QMessageBox.critical(self, "Error", f"Failed to load mesh:\n{exc}")

    def save_mesh(self, file_path):
        if self.mesh_model is None:
            QMessageBox.warning(self, "Warning", "No mesh loaded to save")
            return
        try:
            if self.mesh_polydata is not None:
                vertices = np.asarray(self.mesh_polydata.points, dtype=float)
                faces = self._pyvista_faces_to_triangles(np.asarray(self.mesh_polydata.faces, dtype=int))
            else:
                vertices = self.mesh_model.vertices
                faces = self.mesh_model.faces
            out_mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
            out_mesh.export(file_path)
            QMessageBox.information(self, "Success", f"Mesh saved successfully to {file_path}")
        except Exception as exc:
            QMessageBox.critical(self, "Error", f"Failed to save mesh:\n{exc}")

    def fit_to_view(self):
        if self.plotter is None or self.mesh_model is None:
            return
        try:
            self.movement_plane_locked = False
            self._remove_control_point_actors()
            self._refresh_grid_actor()
            self._refresh_contour_actor()
            self._set_ruler_visible(False)
            self.set_camera_mode("perspective")
            self.plotter.view_isometric()
            self.plotter.reset_camera()
            self.status_message("3D view fitted")
        except Exception:
            pass

    def set_camera_view(self, view):
        if self.plotter is None:
            return
        try:
            presets = {
                "front": ((0, 0, 1), (0, 1, 0)),
                "back": ((0, 0, -1), (0, 1, 0)),
                "left": ((-1, 0, 0), (0, 0, 1)),
                "right": ((1, 0, 0), (0, 0, 1)),
                "top": ((0, 1, 0), (0, 0, 1)),
                "bottom": ((0, -1, 0), (0, 0, 1)),
            }
            if view in presets:
                direction, view_up = presets[view]
                if self.mesh_model is not None and self.mesh_model.get_bounds() is not None:
                    bounds = self.mesh_model.get_bounds()
                    center = (bounds[0] + bounds[1]) / 2.0
                    radius = max(float(np.linalg.norm(bounds[1] - bounds[0])) * 1.5, 1.0)
                else:
                    center = np.zeros(3)
                    radius = 10.0
                position = center + np.asarray(direction, dtype=float) * radius
                self.plotter.camera_position = [position, center, view_up]
                self.plotter.render()
        except Exception:
            pass

    def set_camera_mode(self, mode):
        if self.plotter is None:
            return
        try:
            if mode == "perspective":
                self.plotter.camera.parallel_projection = False
            else:
                self.plotter.camera.parallel_projection = True
            self.plotter.render()
        except Exception:
            pass

    def set_movement_plane(self, plane):
        """Select a fixed 2D editing plane and orient the camera to it."""
        camera_views = {"XY": "front", "YZ": "right", "XZ": "top"}
        self.movement_plane_locked = True
        if hasattr(self, "movement_plane_combo") and self.movement_plane_combo.currentText() != plane:
            self.movement_plane_combo.blockSignals(True)
            self.movement_plane_combo.setCurrentText(plane)
            self.movement_plane_combo.blockSignals(False)
        self.set_camera_mode("orthographic")
        self.set_camera_view(camera_views.get(plane, "XY"))
        self._set_ruler_visible(True)
        self._refresh_grid_actor()
        if self.control_point_manager is not None:
            self.auto_generate_contour_points(60, plane, announce=False)
        self._refresh_contour_actor()
        self._refresh_control_point_markers()
        self.status_message(f"Movement plane: {plane} (2D only)")

    def _set_ruler_visible(self, visible):
        """Show a VTK distance ruler in orthographic editing views only."""
        if self.plotter is None or vtkLegendScaleActor is None:
            return
        try:
            if self.ruler_actor is None:
                self.ruler_actor = vtkLegendScaleActor()
                self.ruler_actor.SetLabelModeToDistance()
                self.ruler_actor.SetTopAxisVisibility(False)
                self.ruler_actor.SetLeftAxisVisibility(True)
                self.ruler_actor.SetRightAxisVisibility(False)
                self.ruler_actor.SetBottomAxisVisibility(True)
                self.plotter.renderer.AddActor2D(self.ruler_actor)
            self.ruler_actor.SetVisibility(bool(visible))
            self.plotter.render()
        except Exception:
            pass

    def set_local_neighbor_count(self, count):
        self.local_neighbor_count = int(count)
        self._refresh_control_point_markers()
        self.status_message(f"Local neighbors: {self.local_neighbor_count}")

    def set_edit_mode(self, mode):
        self.edit_mode = str(mode)
        self.status_message(f"Edit mode: {self.edit_mode}")

    def set_influence_radius(self, value):
        self.influence_radius_percent = int(value)

    def set_edit_strength(self, value):
        self.edit_strength_percent = int(value)

    def set_falloff(self, value):
        self.falloff_type = str(value)

    def toggle_wireframe(self, checked=False):
        self.is_wireframe = bool(checked)
        if self.mesh_actor is None:
            return
        try:
            if self.is_wireframe:
                self.mesh_actor.GetProperty().SetRepresentationToWireframe()
            else:
                self.mesh_actor.GetProperty().SetRepresentationToSurface()
            if self.plotter is not None:
                self.plotter.render()
        except Exception:
            pass

    def update_transparency(self, value):
        if self.mesh_actor is None:
            return
        try:
            self.mesh_actor.GetProperty().SetOpacity(value / 100.0)
            if self.plotter is not None:
                self.plotter.render()
        except Exception:
            pass

    def validate_mesh(self):
        if self.mesh_model is None:
            QMessageBox.warning(self, "Warning", "No mesh loaded to validate")
            return
        try:
            mesh = self.mesh_model.to_trimesh()
            msg = format_mesh_validation_report(mesh)
            QMessageBox.information(self, "Mesh Validation", msg)
        except Exception as exc:
            QMessageBox.critical(self, "Error", f"Failed to validate mesh:\n{exc}")

    def _push_history(self):
        if self.mesh_polydata is None or self.control_point_manager is None:
            return
        snapshot = self._make_history_snapshot()
        self.history = self.history[: self.history_index + 1]
        self.history.append(snapshot)
        self.history_index = len(self.history) - 1

    def _make_history_snapshot(self):
        return {
            "vertices": np.asarray(self.mesh_polydata.points, dtype=float).copy(),
            "points": [(position.copy(), pinned) for position, pinned in self.control_point_manager.control_points],
            "selected": self.control_point_manager.selected_point,
            "plane": self.control_point_manager.active_plane,
            "contour": self.control_point_manager.contour_2d.copy(),
        }

    def _replace_current_history(self):
        if self.mesh_polydata is None or self.control_point_manager is None:
            return
        snapshot = self._make_history_snapshot()
        if self.history_index >= 0:
            self.history[self.history_index] = snapshot
        else:
            self.history = [snapshot]
            self.history_index = 0

    def _restore_history_state(self, snapshot):
        self.mesh_polydata.points = snapshot["vertices"].copy()
        self.mesh_model.vertices = snapshot["vertices"].copy()
        self.control_point_manager.control_points = [(position.copy(), pinned) for position, pinned in snapshot["points"]]
        self.control_point_manager.selected_point = snapshot["selected"]
        self.control_point_manager.active_plane = snapshot["plane"]
        self.control_point_manager.contour_2d = snapshot["contour"].copy()
        self._refresh_contour_actor()
        self._refresh_control_point_markers()
        self.mesh_changed.emit(self.mesh_model.vertices.copy(), self.mesh_model.faces.copy())

    def undo(self):
        if self.mesh_polydata is None or self.history_index <= 0:
            return
        self.history_index -= 1
        self._restore_history_state(self.history[self.history_index])

    def redo(self):
        if self.mesh_polydata is None or self.history_index >= len(self.history) - 1:
            return
        self.history_index += 1
        self._restore_history_state(self.history[self.history_index])

    def reset_view(self):
        if self.mesh_polydata is None:
            return
        self.mesh_polydata.points = self.mesh_model.original_vertices.copy()
        self.mesh_model.vertices = self.mesh_model.original_vertices.copy()
        self._push_history()
        self._apply_mesh_to_plotter()
        self.mesh_changed.emit(self.mesh_model.vertices.copy(), self.mesh_model.faces.copy())

    def add_control_point_at_center(self):
        if not self.movement_plane_locked:
            self.status_message("Select XY, YZ, or XZ before editing")
            return
        if self.control_point_manager is None or self.mesh_model is None:
            return
        bounds = self.mesh_model.get_bounds()
        center = self.mesh_model.get_center()
        # Lift the starter marker slightly above the top surface so an opaque mesh
        # cannot hide it from the user.
        extent = max(float(np.linalg.norm(bounds[1] - bounds[0])), 1.0)
        center[2] = bounds[1][2] + extent * 0.015
        self.control_point_manager.add_control_point(center)
        self._push_history()
        self._refresh_control_point_markers()
        self.status_message("Added visible center control point")

    def clear_control_points(self):
        if self.control_point_manager is None:
            return
        self.control_point_manager.clear_all_points()
        self._push_history()
        self._refresh_control_point_markers()
        self.status_message("Control points cleared")

    def remove_selected_control_point(self):
        if self.control_point_manager is None:
            return
        index = self.control_point_manager.get_selected_point()
        if index is None:
            return
        self.control_point_manager.remove_control_point(index)
        self._push_history()
        self._refresh_control_point_markers()
        self.status_message("Selected control point removed")

    def pin_selected_control_point(self):
        if self.control_point_manager is None:
            return
        index = self.control_point_manager.get_selected_point()
        if index is None:
            return
        self.control_point_manager.pin_point(index)
        self._push_history()
        self._refresh_control_point_markers()
        self.status_message("Selected control point pinned")

    def dissolve_region_between_anchors(self):
        """Straighten and blend the contour interval bounded by two anchors."""
        if not self.movement_plane_locked or self.control_point_manager is None or self.mesh_model is None:
            self.status_message("Select a 2D plane and pin two boundary points first")
            return
        source_points = np.asarray(self.control_point_manager.get_control_points(), dtype=float)
        if not self.control_point_manager.dissolve_between_anchors():
            self.status_message("Dissolve requires two non-adjacent pinned points")
            return
        target_points = np.asarray(self.control_point_manager.get_control_points(), dtype=float)
        bounds = self.mesh_model.get_bounds()
        plane = self.movement_plane_combo.currentText()
        plane_axes = {"XY": (0, 1), "XZ": (0, 2), "YZ": (1, 2)}[plane]
        radius = max(float(np.linalg.norm((bounds[1] - bounds[0])[list(plane_axes)])) * 0.25, 0.001)
        vertices = apply_plane_displacement_field(self.mesh_model.vertices, source_points, target_points, plane, radius)
        self.mesh_model.vertices = vertices
        if self.mesh_polydata is not None:
            self.mesh_polydata.points = vertices
        self._push_history()
        self._refresh_grid_actor()
        self._refresh_contour_actor()
        self._refresh_control_point_markers()
        self.mesh_changed.emit(self.mesh_model.vertices.copy(), self.mesh_model.faces.copy())
        self.status_message("Region dissolved smoothly between anchors")

    def auto_generate_contour_points(self, density=60, plane=None, announce=True):
        if not self.movement_plane_locked:
            self.status_message("Select XY, YZ, or XZ before editing")
            return
        if self.control_point_manager is None:
            return
        plane = plane or self.movement_plane_combo.currentText()
        count = self.control_point_manager.auto_generate_contour_points(density, plane)
        self._refresh_contour_actor()
        self._refresh_control_point_markers()
        if announce:
            self._push_history()
        else:
            self._replace_current_history()
        if announce:
            self.status_message(f"Generated {count} contour points on {plane}")

    def apply_deformation_from_control_points(self):
        if self.mesh_model is None or self.control_point_manager is None or self.deformation_engine is None:
            return
        control_points = self.control_point_manager.get_control_points()
        if not control_points:
            return
        displacements = []
        for idx, point in enumerate(control_points):
            if idx == self.control_point_manager.get_dragging_point():
                displacements.append(np.array([0.0, 0.0, 0.0]))
            else:
                displacements.append(np.array([0.0, 0.0, 0.0]))
        pinned = self.control_point_manager.get_pinned_points()
        radius = max(0.001, float(self._current_radius()) / 10.0)
        strength = float(self._current_strength()) / 100.0
        falloff = self._current_falloff()
        new_vertices = self.deformation_engine.apply_deformation(
            control_points,
            displacements,
            influence_radius=radius,
            falloff_type=falloff,
            strength=strength,
            pinned_indices=pinned,
        )
        self.mesh_model.vertices = np.asarray(new_vertices, dtype=float)
        if self.mesh_polydata is not None:
            self.mesh_polydata.points = new_vertices
        self._push_history()
        if self.plotter is not None:
            self.plotter.render()

    def _current_radius(self):
        return 20.0

    def _current_strength(self):
        return 50.0

    def _current_falloff(self):
        return "smoothstep"

    def status_message(self, message):
        window = self.window()
        if window is not None and hasattr(window, "status_bar"):
            window.status_bar.showMessage(message, 4000)

    @staticmethod
    def _pyvista_faces_to_triangles(faces):
        if len(faces) == 0:
            return np.empty((0, 3), dtype=int)
        faces = np.asarray(faces, dtype=int).ravel()
        triangles = []
        i = 0
        while i < len(faces):
            n = int(faces[i])
            triangles.append(faces[i + 1:i + 1 + n])
            i += n + 1
        return np.asarray(triangles, dtype=int)
