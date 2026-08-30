import numpy as np
import trimesh

from geometry.mesh_validation import format_mesh_validation_report, inspect_mesh


def test_validation_uses_current_trimesh_degenerate_api_read_only():
    mesh = trimesh.Trimesh(
        vertices=np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float),
        faces=np.array([[0, 1, 2], [0, 0, 1]], dtype=int),
        process=False,
    )
    vertices_before = mesh.vertices.copy()
    faces_before = mesh.faces.copy()

    report = inspect_mesh(mesh)

    assert report["Degenerate faces"] == "1"
    assert "Degenerate faces: 1" in format_mesh_validation_report(mesh)
    np.testing.assert_array_equal(mesh.vertices, vertices_before)
    np.testing.assert_array_equal(mesh.faces, faces_before)


def test_validation_reports_current_trimesh_box():
    report = inspect_mesh(trimesh.creation.box())

    assert report["Degenerate faces"] == "0"
    assert report["Finite vertex coordinates"] == "Yes"
    assert report["Watertight"] == "Yes"
