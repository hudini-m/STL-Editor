from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QGroupBox,
    QLabel,
    QSlider,
    QSpinBox,
    QComboBox,
)


class EditingPanel(QWidget):
    auto_generate_requested = Signal(int)
    add_point_requested = Signal()
    clear_points_requested = Signal()
    delete_point_requested = Signal()
    pin_selected_requested = Signal()
    validate_requested = Signal()
    dissolve_region_requested = Signal()
    neighbor_count_changed = Signal(int)
    edit_mode_changed = Signal(str)
    radius_changed = Signal(int)
    strength_changed = Signal(int)
    falloff_changed = Signal(str)

    def __init__(self):
        super().__init__()
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 8, 12, 12)
        layout.setSpacing(8)

        control_group = QGroupBox("Control Point Edit")
        control_layout = QVBoxLayout(control_group)

        self.toggle_btn = QPushButton("Enable Control Point Edit")
        self.toggle_btn.setCheckable(True)
        self.toggle_btn.setObjectName("primaryButton")
        control_layout.addWidget(self.toggle_btn)
        control_layout.addWidget(QLabel("Drag points to edit. Shift-click toggles anchors. Middle-drag pans; right-drag zooms."))

        mode_row = QHBoxLayout()
        mode_row.addWidget(QLabel("Edit mode:"))
        self.edit_mode_combo = QComboBox()
        self.edit_mode_combo.addItems(["Smooth contour", "Point-wise"])
        mode_row.addWidget(self.edit_mode_combo)
        control_layout.addLayout(mode_row)

        auto_row = QHBoxLayout()
        self.auto_btn = QPushButton("Auto Generate Contour Points")
        self.auto_density = QSpinBox()
        self.auto_density.setRange(10, 300)
        self.auto_density.setValue(60)
        auto_row.addWidget(self.auto_btn)
        auto_row.addWidget(QLabel("Density"))
        auto_row.addWidget(self.auto_density)
        control_layout.addLayout(auto_row)

        self.add_point_btn = QPushButton("Add Control Point")
        self.delete_btn = QPushButton("Delete Control Point")
        self.clear_btn = QPushButton("Clear Points")
        self.pin_btn = QPushButton("Pin Selected Point")
        self.validate_btn = QPushButton("Validate Mesh")
        control_layout.addWidget(self.add_point_btn)
        control_layout.addWidget(self.delete_btn)
        control_layout.addWidget(self.clear_btn)
        control_layout.addWidget(self.pin_btn)
        control_layout.addWidget(self.validate_btn)
        layout.addWidget(control_group)

        deform_group = QGroupBox("Deformation Controls")
        deform_layout = QVBoxLayout(deform_group)

        neighbor_layout = QHBoxLayout()
        neighbor_layout.addWidget(QLabel("Local neighbors:"))
        self.neighbor_spin = QSpinBox()
        self.neighbor_spin.setRange(1, 30)
        self.neighbor_spin.setValue(4)
        neighbor_layout.addWidget(self.neighbor_spin)
        deform_layout.addLayout(neighbor_layout)

        radius_layout = QHBoxLayout()
        radius_label = QLabel("Influence Radius:")
        self.radius_slider = QSlider(Qt.Horizontal)
        self.radius_slider.setRange(1, 100)
        self.radius_slider.setValue(20)
        radius_layout.addWidget(radius_label)
        radius_layout.addWidget(self.radius_slider)
        deform_layout.addLayout(radius_layout)

        strength_layout = QHBoxLayout()
        strength_label = QLabel("Strength:")
        self.strength_slider = QSlider(Qt.Horizontal)
        self.strength_slider.setRange(1, 100)
        self.strength_slider.setValue(50)
        strength_layout.addWidget(strength_label)
        strength_layout.addWidget(self.strength_slider)
        deform_layout.addLayout(strength_layout)

        falloff_layout = QHBoxLayout()
        falloff_label = QLabel("Falloff:")
        self.falloff_combo = QComboBox()
        self.falloff_combo.addItems(["Linear", "Smoothstep", "Gaussian"])
        falloff_layout.addWidget(falloff_label)
        falloff_layout.addWidget(self.falloff_combo)
        deform_layout.addLayout(falloff_layout)
        layout.addWidget(deform_group)

        selection_group = QGroupBox("Selection Tools")
        selection_layout = QVBoxLayout(selection_group)
        self.brush_btn = QPushButton("Brush Selection")
        self.box_btn = QPushButton("Box Selection")
        self.connected_btn = QPushButton("Connected Region")
        self.dissolve_btn = QPushButton("Dissolve Region Between Anchors")
        selection_layout.addWidget(self.brush_btn)
        selection_layout.addWidget(self.box_btn)
        selection_layout.addWidget(self.connected_btn)
        selection_layout.addWidget(self.dissolve_btn)
        layout.addWidget(selection_group)

        transform_group = QGroupBox("Transform Tools")
        transform_layout = QVBoxLayout(transform_group)
        self.move_btn = QPushButton("Move")
        self.scale_btn = QPushButton("Scale")
        self.rotate_btn = QPushButton("Rotate")
        self.inflate_btn = QPushButton("Inflate")
        self.deflate_btn = QPushButton("Deflate")
        self.smooth_btn = QPushButton("Smooth")
        for btn in [self.move_btn, self.scale_btn, self.rotate_btn, self.inflate_btn, self.deflate_btn, self.smooth_btn]:
            transform_layout.addWidget(btn)
        layout.addWidget(transform_group)

        info_group = QGroupBox("Mesh Information")
        info_layout = QVBoxLayout(info_group)
        self.info_label = QLabel("No mesh loaded")
        self.info_label.setWordWrap(True)
        info_layout.addWidget(self.info_label)
        layout.addWidget(info_group)

        layout.addStretch()

        self.auto_btn.clicked.connect(lambda: self.auto_generate_requested.emit(self.auto_density.value()))
        self.add_point_btn.clicked.connect(self.add_point_requested.emit)
        self.clear_btn.clicked.connect(self.clear_points_requested.emit)
        self.delete_btn.clicked.connect(self.delete_point_requested.emit)
        self.pin_btn.clicked.connect(self.pin_selected_requested.emit)
        self.validate_btn.clicked.connect(self.validate_requested.emit)
        self.dissolve_btn.clicked.connect(self.dissolve_region_requested.emit)
        self.neighbor_spin.valueChanged.connect(self.neighbor_count_changed.emit)
        self.edit_mode_combo.currentTextChanged.connect(self.edit_mode_changed.emit)
        self.radius_slider.valueChanged.connect(self.radius_changed.emit)
        self.strength_slider.valueChanged.connect(self.strength_changed.emit)
        self.falloff_combo.currentTextChanged.connect(self.falloff_changed.emit)

    def set_mesh_info(self, name, vertices, faces):
        self.info_label.setText(f"{name}\nVertices: {vertices}\nFaces: {faces}")
