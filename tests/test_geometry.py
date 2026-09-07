import numpy as np
import trimesh

from geometry.contour import ContourExtractor
from geometry.deformation import apply_plane_displacement_field, flatten_plane_region
from geometry.mesh_model import MeshModel
from geometry.mesh_cutting import cut_mesh_to_contour_chord
from geometry.projection import lift_points, project_points
from interaction.control_points import ControlPointManager


def test_projection_round_trip_preserves_plane_axes():
    points = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    for plane, depth in [("XY", points[:, 2]), ("XZ", points[:, 1]), ("YZ", points[:, 0])]:
        rebuilt = lift_points(project_points(points, plane), depth, plane)
        np.testing.assert_allclose(rebuilt, points)


def test_contour_resampling_is_arc_length_based():
    contour = np.array([[0, 0], [4, 0], [4, 2], [0, 2]], dtype=float)
    samples = ContourExtractor(None).sample_contour(contour, 8)
    expected = np.array(
        [[0, 0], [1.5, 0], [3, 0], [4, 0.5], [4, 2], [2.5, 2], [1, 2], [0, 1.5]],
        dtype=float,
    )
    np.testing.assert_allclose(samples, expected)


def test_local_control_point_move_keeps_outside_points_fixed():
    model = MeshModel(np.zeros((12, 3)), np.zeros((0, 3), dtype=int))
    manager = ControlPointManager(model)
    for angle in np.linspace(0.0, 2.0 * np.pi, 12, endpoint=False):
        manager.add_control_point([np.cos(angle), np.sin(angle), 0.0])
    original = np.asarray(manager.get_control_points()).copy()
    target = original[6] + np.array([0.0, 0.4, 0.0])
    assert manager.move_point_locally(6, target, neighbor_count=2)
    moved = np.asarray(manager.get_control_points())
    np.testing.assert_allclose(moved[6], target)
    np.testing.assert_allclose(moved[3], original[3])
    np.testing.assert_allclose(moved[9], original[9])
    assert not np.allclose(moved[5], original[5]) and not np.allclose(moved[7], original[7])


def test_anchors_limit_the_cyclic_deformation_interval():
    model = MeshModel(np.zeros((10, 3)), np.zeros((0, 3), dtype=int))
    manager = ControlPointManager(model)
    for angle in np.linspace(0.0, 2.0 * np.pi, 10, endpoint=False):
        manager.add_control_point([np.cos(angle), np.sin(angle), 7.0])
    manager.pin_point(2)
    manager.pin_point(7)
    original = np.asarray(manager.get_control_points()).copy()
    target = original[4] + np.array([0.0, 0.5, 0.0])
    assert manager.move_point_locally(4, target, neighbor_count=4)
    moved = np.asarray(manager.get_control_points())
    np.testing.assert_allclose(moved[4], target)
    np.testing.assert_allclose(moved[[2, 7]], original[[2, 7]])
    np.testing.assert_allclose(moved[[0, 9]], original[[0, 9]])
    assert not np.allclose(moved[3], original[3]) and not np.allclose(moved[6], original[6])


def test_plane_field_is_local_and_preserves_xy_depth():
    vertices = np.array([[0, 0, 5], [1, 0, 5], [2, 0, 5], [20, 0, 5]], dtype=float)
    source = np.array([[0, 0, 5], [2, 0, 5]], dtype=float)
    target = np.array([[0, 2, 5], [2, 0, 5]], dtype=float)
    result = apply_plane_displacement_field(vertices, source, target, "XY", support_radius=3.0)
    assert result[0, 1] > 1.9
    assert result[1, 1] > 0.0
    np.testing.assert_allclose(result[3], vertices[3])
    np.testing.assert_array_equal(result[:, 2], vertices[:, 2])


def test_wraparound_anchor_interval_only_moves_wrapped_section():
    model = MeshModel(np.zeros((10, 3)), np.zeros((0, 3), dtype=int))
    manager = ControlPointManager(model)
    for angle in np.linspace(0.0, 2.0 * np.pi, 10, endpoint=False):
        manager.add_control_point([np.cos(angle), np.sin(angle), 2.0])
    manager.pin_point(8)
    manager.pin_point(2)
    original = np.asarray(manager.get_control_points()).copy()
    target = original[0] + np.array([0.0, 0.5, 0.0])
    assert manager.move_point_locally(0, target, neighbor_count=1)
    moved = np.asarray(manager.get_control_points())
    assert moved[0, 1] > original[0, 1]
    np.testing.assert_allclose(moved[3:8], original[3:8])
    np.testing.assert_allclose(moved[[8, 2]], original[[8, 2]])


def test_self_intersecting_drag_is_rejected_without_mutation():
    model = MeshModel(np.zeros((4, 3)), np.zeros((0, 3), dtype=int))
    manager = ControlPointManager(model)
    for point in [[0, 0, 0], [2, 0, 0], [2, 2, 0], [0, 2, 0]]:
        manager.add_control_point(point)
    manager.pin_point(1)
    manager.pin_point(3)
    original = np.asarray(manager.get_control_points()).copy()
    assert not manager.move_point_locally(2, [0, 0, 0], neighbor_count=1)
    np.testing.assert_allclose(np.asarray(manager.get_control_points()), original)


def test_dissolve_between_anchors_blends_only_the_bounded_interval():
    model = MeshModel(np.zeros((8, 3)), np.zeros((0, 3), dtype=int))
    manager = ControlPointManager(model)
    for point in [[0, 0, 0], [1, 2, 0], [2, -1, 0], [3, 3, 0], [4, 0, 0], [5, 1, 0]]:
        manager.add_control_point(point)
    manager.pin_point(0)
    manager.pin_point(4)
    assert manager.dissolve_between_anchors()
    points = np.asarray(manager.get_control_points())
    np.testing.assert_allclose(points[[0, 4]], [[0, 0, 0], [4, 0, 0]])
    np.testing.assert_allclose(points[1:4, 1], 0.0)
    np.testing.assert_allclose(points[5], [5, 1, 0])


def test_exact_point_move_changes_only_the_selected_contour_point():
    model = MeshModel(np.zeros((4, 3)), np.zeros((0, 3), dtype=int))
    manager = ControlPointManager(model)
    for point in [[0, 0, 0], [2, 0, 0], [2, 2, 0], [0, 2, 0]]:
        manager.add_control_point(point)
    original = np.asarray(manager.get_control_points()).copy()
    assert manager.move_point_exact(1, [2.5, 0.5, 0])
    moved = np.asarray(manager.get_control_points())
    np.testing.assert_allclose(moved[1], [2.5, 0.5, 0])
    np.testing.assert_allclose(moved[[0, 2, 3]], original[[0, 2, 3]])


def test_plane_displacement_preserves_depth_in_xz_and_yz():
    vertices = np.array([[0, 0, 5], [1, 2, 6], [4, 3, 8]], dtype=float)
    source = np.array([[1, 2, 6]], dtype=float)
    target = np.array([[3, 5, 99]], dtype=float)
    for plane, depth_axis in [("XZ", 1), ("YZ", 0)]:
        result = apply_plane_displacement_field(vertices, source, target, plane, support_radius=10.0)
        np.testing.assert_array_equal(result[:, depth_axis], vertices[:, depth_axis])


def test_unchanged_contour_points_do_not_cancel_visible_mesh_edit():
    vertices = np.array([[0.1, 0.0, 3.0], [10.0, 10.0, 3.0]])
    source = np.array([[0.0, 0.0, 3.0], *[[0.5, value, 3.0] for value in np.linspace(-1.0, 1.0, 20)]])
    target = source.copy()
    target[0, 1] = 2.0
    result = apply_plane_displacement_field(vertices, source, target, "XY", support_radius=2.0)
    assert result[0, 1] > 1.5
    np.testing.assert_allclose(result[1], vertices[1])


def test_dissolved_contour_displaces_matching_mesh_vertices():
    source = np.array([[0, 0, 2], [1, 2, 2], [2, -1, 2], [3, 3, 2], [4, 0, 2]], dtype=float)
    target = source.copy()
    target[1:4, 1] = 0.0
    result = apply_plane_displacement_field(source.copy(), source, target, "XY", support_radius=3.0)
    np.testing.assert_allclose(result[1:4, :2], target[1:4, :2])
    np.testing.assert_array_equal(result[:, 2], source[:, 2])


def test_flatten_region_places_curved_boundary_exactly_on_anchor_line():
    contour = np.array(
        [[0, 0, 7], [1, -2, 7], [2, -3, 7], [3, -2, 7], [4, 0, 7], [4, 4, 7], [0, 4, 7]],
        dtype=float,
    )
    result = flatten_plane_region(contour.copy(), contour, [0, 1, 2, 3, 4], "XY", support_radius=2.0)
    np.testing.assert_allclose(result[:5, 1], 0.0, atol=1e-10)
    np.testing.assert_allclose(result[[0, 4], :2], contour[[0, 4], :2])
    np.testing.assert_array_equal(result[:, 2], contour[:, 2])
    np.testing.assert_allclose(result[5:], contour[5:])


def test_anchor_chord_cut_changes_topology_and_caps_actual_mesh():
    mesh = trimesh.creation.icosphere(subdivisions=2, radius=10.0)
    contour = np.array(
        [[-10, 0, 0], [-7, -7, 0], [0, -10, 0], [7, -7, 0], [10, 0, 0], [0, 10, 0]],
        dtype=float,
    )
    cut = cut_mesh_to_contour_chord(mesh.vertices, mesh.faces, contour, [0, 1, 2, 3, 4], "XY")
    assert len(cut.faces) != len(mesh.faces)
    assert np.min(cut.vertices[:, 1]) >= -1e-8
    assert cut.is_watertight
    assert cut.is_winding_consistent
