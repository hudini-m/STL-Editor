import numpy as np


class PickingManager:
    def __init__(self, mesh_model):
        self.mesh_model = mesh_model

    def pick_vertex(self, point, max_distance=1.0):
        if len(self.mesh_model.vertices) == 0:
            return None
        point = np.asarray(point, dtype=float)
        distances = np.linalg.norm(self.mesh_model.vertices - point, axis=1)
        min_idx = int(np.argmin(distances))
        if distances[min_idx] <= max_distance:
            return min_idx
        return None

    def pick_face(self, point):
        if len(self.mesh_model.faces) == 0:
            return None
        point = np.asarray(point, dtype=float)
        face_centers = self.mesh_model.vertices[self.mesh_model.faces].mean(axis=1)
        return int(np.argmin(np.linalg.norm(face_centers - point, axis=1)))

    def get_nearest_face(self, point):
        vertex_idx = self.pick_vertex(point)
        if vertex_idx is None:
            return None
        for i, face in enumerate(self.mesh_model.faces):
            if vertex_idx in face:
                return i
        return None

