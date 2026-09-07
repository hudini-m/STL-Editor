from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QToolBar,
    QStatusBar,
    QPushButton,
    QFileDialog,
    QSplitter,
    QToolButton,
    QScrollArea,
    QFrame,
    QTabWidget,
)

from viewport import ViewportWidget
from ui.editing_panel import EditingPanel
from views.viewer_3d import Viewer3DWidget


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("STL Changer")
        self.setGeometry(100, 100, 1400, 900)
        self.setAcceptDrops(True)
        self.setMinimumSize(980, 640)
        self.setStyleSheet(self._style_sheet())

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)

        tabs = QTabWidget()
        main_layout.addWidget(tabs)

        editor_page = QWidget()
        editor_layout = QVBoxLayout(editor_page)
        splitter = QSplitter(Qt.Horizontal)
        editor_layout.addWidget(splitter)
        tabs.addTab(editor_page, "2D Editor")

        self.viewport = ViewportWidget()
        splitter.addWidget(self.viewport)

        panel_scroll = QScrollArea()
        panel_scroll.setWidgetResizable(True)
        panel_scroll.setFrameShape(QFrame.NoFrame)
        self.editing_panel = EditingPanel()
        panel_scroll.setWidget(self.editing_panel)
        splitter.addWidget(panel_scroll)
        splitter.setSizes([900, 500])

        self.viewer_3d = Viewer3DWidget()
        tabs.addTab(self.viewer_3d, "3D Inspection")

        # Build actions after the widgets they control have been created.
        self.create_menu_bar()
        self.create_toolbar()

        self.viewport.mesh_loaded.connect(self.on_mesh_loaded)
        self.viewport.mesh_changed.connect(self.viewer_3d.set_mesh)
        self.editing_panel.auto_generate_requested.connect(self.viewport.auto_generate_contour_points)
        self.editing_panel.add_point_requested.connect(self.viewport.add_control_point_at_center)
        self.editing_panel.clear_points_requested.connect(self.viewport.clear_control_points)
        self.editing_panel.delete_point_requested.connect(self.viewport.remove_selected_control_point)
        self.editing_panel.pin_selected_requested.connect(self.viewport.pin_selected_control_point)
        self.editing_panel.validate_requested.connect(self.viewport.validate_mesh)
        self.editing_panel.neighbor_count_changed.connect(self.viewport.set_local_neighbor_count)
        self.editing_panel.edit_mode_changed.connect(self.viewport.set_edit_mode)
        self.editing_panel.radius_changed.connect(self.viewport.set_influence_radius)
        self.editing_panel.strength_changed.connect(self.viewport.set_edit_strength)
        self.editing_panel.falloff_changed.connect(self.viewport.set_falloff)

    def create_menu_bar(self):
        menubar = self.menuBar()

        file_menu = menubar.addMenu("File")
        open_action = file_menu.addAction("Open STL")
        sample_action = file_menu.addAction("Load Sample Cube")
        save_action = file_menu.addAction("Save STL")
        file_menu.addSeparator()
        exit_action = file_menu.addAction("Exit")

        edit_menu = menubar.addMenu("Edit")
        undo_action = edit_menu.addAction("Undo")
        redo_action = edit_menu.addAction("Redo")
        reset_action = edit_menu.addAction("Reset")

        view_menu = menubar.addMenu("View")
        fit_action = view_menu.addAction("Fit 3D")
        xy_action = view_menu.addAction("XY Plane")
        yz_action = view_menu.addAction("YZ Plane")
        xz_action = view_menu.addAction("XZ Plane")

        open_action.triggered.connect(self.open_stl)
        sample_action.triggered.connect(self.load_sample)
        save_action.triggered.connect(self.save_stl)
        exit_action.triggered.connect(self.close)
        undo_action.triggered.connect(self.undo)
        redo_action.triggered.connect(self.redo)
        reset_action.triggered.connect(self.reset)
        fit_action.triggered.connect(self.viewport.fit_to_view)
        xy_action.triggered.connect(lambda: self.viewport.set_movement_plane("XY"))
        yz_action.triggered.connect(lambda: self.viewport.set_movement_plane("YZ"))
        xz_action.triggered.connect(lambda: self.viewport.set_movement_plane("XZ"))

    def create_toolbar(self):
        toolbar = QToolBar("Main Toolbar")
        self.addToolBar(toolbar)

        for text, callback in [
            ("Open STL", self.open_stl),
            ("Sample Cube", self.load_sample),
            ("Save STL", self.save_stl),
            ("Undo", self.undo),
            ("Redo", self.redo),
            ("Reset", self.reset),
            ("Fit 3D", self.viewport.fit_to_view),
            ("XY", lambda: self.viewport.set_movement_plane("XY")),
            ("YZ", lambda: self.viewport.set_movement_plane("YZ")),
            ("XZ", lambda: self.viewport.set_movement_plane("XZ")),
        ]:
            button = QToolButton()
            button.setText(text)
            button.setToolButtonStyle(Qt.ToolButtonTextOnly)
            button.clicked.connect(callback)
            toolbar.addWidget(button)

        toolbar.addSeparator()
        hint = QPushButton("Drop an STL into the viewport")
        hint.setEnabled(False)
        hint.setObjectName("toolbarHint")
        toolbar.addWidget(hint)

    def open_stl(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Open STL File", "", "STL Files (*.stl);;All Files (*)")
        if file_path:
            self.viewport.load_mesh(file_path)

    def load_sample(self):
        self.viewport.load_sample_mesh()

    def save_stl(self):
        file_path, _ = QFileDialog.getSaveFileName(self, "Save STL File", "", "STL Files (*.stl);;All Files (*)")
        if file_path:
            self.viewport.save_mesh(file_path)

    def undo(self):
        self.viewport.undo()

    def redo(self):
        self.viewport.redo()

    def reset(self):
        self.viewport.reset_view()

    def on_mesh_loaded(self, mesh_info):
        self.status_bar.showMessage(
            f"Loaded: {mesh_info['name']} - Vertices: {mesh_info['vertices']}, Faces: {mesh_info['faces']}"
        )
        self.editing_panel.set_mesh_info(mesh_info["name"], mesh_info["vertices"], mesh_info["faces"])

    @staticmethod
    def _style_sheet():
        return """
        QMainWindow, QWidget { background: #f4f7fb; color: #1f2937; }
        QMenuBar { background: #152238; color: #f8fafc; padding: 5px; }
        QMenuBar::item:selected, QMenu::item:selected { background: #2e6f95; }
        QMenu { background: #ffffff; color: #1f2937; border: 1px solid #d7e0ea; }
        QToolBar { background: #ffffff; border: 0; border-bottom: 1px solid #d7e0ea; spacing: 6px; padding: 7px 10px; }
        QToolButton, QPushButton { background: #ffffff; border: 1px solid #c7d4e2; border-radius: 5px; padding: 7px 10px; }
        QToolButton:hover, QPushButton:hover { background: #e8f2f7; border-color: #5592b0; }
        QToolButton:pressed, QPushButton:pressed { background: #d3e7ef; }
        QPushButton#toolbarHint { border: 0; color: #6b7d90; }
        QPushButton#primaryButton { background: #24536d; color: #ffffff; border-color: #24536d; }
        QPushButton#primaryButton:checked { background: #2e6f95; }
        QGroupBox { font-weight: 600; border: 1px solid #d7e0ea; border-radius: 7px; margin-top: 12px; padding: 12px 8px 8px; background: #ffffff; }
        QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 5px; color: #24536d; }
        QLabel { color: #526579; }
        QSlider::groove:horizontal { height: 5px; background: #d7e0ea; border-radius: 2px; }
        QSlider::handle:horizontal { width: 14px; margin: -5px 0; border-radius: 7px; background: #2e6f95; }
        QStatusBar { background: #152238; color: #dce8f1; }
        """
