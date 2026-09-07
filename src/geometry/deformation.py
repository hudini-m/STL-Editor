import numpy as np

from geometry.projection import PLANE_AXES, project_points


class DeformationEngine:
    def __init__(self, mesh_model):
        self.mesh_model = mesh_model

    def compute_distance_weights(self, control_points, vertices, influence_radius, falloff_type="smoothstep"):
        if len(control_points) == 0 or len(vertices) == 0 or influence_radius <= 0:
            return np.zeros(len(vertices), dtype=float)

        control_points = np.asarray(control_points, dtype=float)
        vertices = np.asarray(vertices, dtype=float)
        distances = np.linalg.norm(vertices[:, np.newaxis, :] - control_points[np.newaxis, :, :], axis=2)
        min_distances = np.min(distances, axis=1)
        weights = np.zeros(len(vertices), dtype=float)
        within_radius = min_distances <= influence_radius
        if not np.any(within_radius):
            return weights

        normalized = np.clip(min_distances[within_radius] / influence_radius, 0.0, 1.0)
        if falloff_type == "linear":
            weights[within_radius] = 1.0 - normalized
        elif falloff_type == "gaussian":
            sigma = max(influence_radius / 3.0, 1e-6)
            weights[within_radius] = np.exp(-0.5 * (min_distances[within_radius] ** 2) / (sigma ** 2))
        else:
            t = normalized
            weights[within_radius] = 1.0 - t * t * (3.0 - 2.0 * t)
        return weights

    def apply_deformation(
        self,
        control_points,
        displacements,
        influence_radius=1.0,
        falloff_type="smoothstep",
        strength=1.0,
        pinned_indices=None,
    ):
        if len(self.mesh_model.vertices) == 0:
            return self.mesh_model.vertices

        if pinned_indices is None:
            pinned_indices = []

        control_points = np.asarray(control_points, dtype=float)
        displacements = np.asarray(displacements, dtype=float)
        if len(control_points) == 0 or len(displacements) == 0:
            return self.mesh_model.vertices.copy()

        deformed_vertices = self.mesh_model.vertices.copy()
        weights = self.compute_distance_weights(control_points, self.mesh_model.vertices, influence_radius, falloff_type)

        for i, vertex_pos in enumerate(self.mesh_model.vertices):
            weight = weights[i]
            if weight <= 0:
                continue

            total_displacement = np.zeros(3, dtype=float)
            for cp_idx, (cp_pos, disp) in enumerate(zip(control_points, displacements)):
                if cp_idx in pinned_indices:
                    continue
                dist = np.linalg.norm(vertex_pos - cp_pos)
                if dist <= influence_radius:
                    local_weight = 1.0 - (dist / influence_radius)
                    total_displacement += disp * local_weight

            deformed_vertices[i] = vertex_pos + total_displacement * weight * strength

        return deformed_vertices

    def apply_smooth_deformation(
        self,
        control_points,
        displacements,
        influence_radius=1.0,
        falloff_type="smoothstep",
        strength=1.0,
    ):
        return self.apply_deformation(control_points, displacements, influence_radius, falloff_type, strength)

    def compute_vertex_displacement(
        self,
        vertex_idx,
        control_points,
        displacements,
        influence_radius=1.0,
        falloff_type="smoothstep",
    ):
        if len(control_points) == 0:
            return np.array([0.0, 0.0, 0.0])

        total_displacement = np.zeros(3, dtype=float)
        for cp_pos, disp in zip(np.asarray(control_points, dtype=float), np.asarray(displacements, dtype=float)):
            dist = np.linalg.norm(self.mesh_model.vertices[vertex_idx] - cp_pos)
            if dist <= influence_radius:
                local_weight = 1.0 - (dist / influence_radius)
                total_displacement += disp * local_weight
        return total_displacement


def apply_plane_displacement_field(vertices, source_points, target_points, plane, support_radius):
    """Apply a compact 2D weighted displacement field while preserving plane depth."""
    result = np.asarray(vertices, dtype=float).copy()
    source = project_points(source_points, plane)
    target = project_points(target_points, plane)
    if len(source) == 0 or support_radius <= 0:
        return result
    displacement = target - source
    moved = np.linalg.norm(displacement, axis=1) > 1e-10
    if not np.any(moved):
        return result
    # Unchanged contour points must not dilute a local edit. They previously made
    # a visibly moved contour produce almost no corresponding mesh deformation.
    source = source[moved]
    displacement = displacement[moved]
    projected_vertices = project_points(result, plane)
    distances = np.linalg.norm(projected_vertices[:, np.newaxis, :] - source[np.newaxis, :, :], axis=2)
    normalized = np.clip(distances / support_radius, 0.0, 1.0)
    weights = (1.0 - normalized * normalized * (3.0 - 2.0 * normalized)) * (distances <= support_radius)
    totals = weights.sum(axis=1)
    active = totals > 1e-12
    offsets = np.zeros_like(projected_vertices)
    blended = (weights[active] @ displacement) / totals[active, np.newaxis]
    envelope = np.max(weights[active], axis=1)
    offsets[active] = blended * envelope[:, np.newaxis]
    exact_matches = distances <= 1e-10
    for vertex_index in np.where(np.any(exact_matches, axis=1))[0]:
        control_index = int(np.argmax(exact_matches[vertex_index]))
        offsets[vertex_index] = displacement[control_index]
    first_axis, second_axis, _ = PLANE_AXES[plane]
    result[:, first_axis] += offsets[:, 0]
    result[:, second_axis] += offsets[:, 1]
    return result
