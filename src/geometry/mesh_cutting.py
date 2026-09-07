import numpy as np
import trimesh

from geometry.projection import PLANE_AXES, project_points


def cut_mesh_to_contour_chord(vertices, faces, contour_points, region_indices, plane):
    """Remove the side of a mesh containing an anchored contour arc and cap the cut."""
    contour = np.asarray(contour_points, dtype=float)
    indices = np.asarray(region_indices, dtype=int)
    if len(indices) < 3:
        raise ValueError("Two non-adjacent anchors are required")

    arc = project_points(contour[indices], plane)
    chord = arc[-1] - arc[0]
    chord_length = float(np.linalg.norm(chord))
    if chord_length <= 1e-10:
        raise ValueError("Anchor points are too close together")

    signed_distance = chord[0] * (arc[1:-1, 1] - arc[0, 1]) - chord[1] * (
        arc[1:-1, 0] - arc[0, 0]
    )
    bulge_side = float(np.median(signed_distance))
    if abs(bulge_side) <= 1e-10:
        raise ValueError("The selected contour region is already straight")

    first_axis, second_axis, _ = PLANE_AXES[plane]
    normal_2d = np.array([-chord[1], chord[0]], dtype=float)
    # Trimesh keeps the positive side of the plane, so point the normal away
    # from the selected bulge to remove that region.
    if bulge_side > 0:
        normal_2d *= -1.0
    plane_normal = np.zeros(3, dtype=float)
    plane_normal[first_axis] = normal_2d[0]
    plane_normal[second_axis] = normal_2d[1]
    plane_normal /= np.linalg.norm(plane_normal)

    plane_origin = contour[indices[0]].copy()
    mesh = trimesh.Trimesh(vertices=np.asarray(vertices), faces=np.asarray(faces), process=False)
    cut = mesh.slice_plane(plane_origin=plane_origin, plane_normal=plane_normal, cap=True)
    if cut is None or len(cut.faces) == 0:
        raise ValueError("The selected cut removed the entire mesh")
    cut.remove_unreferenced_vertices()
    return cut
