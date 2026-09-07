import numpy as np
import trimesh
import pyvista as pv


class MeshModel:
    def __init__(self, vertices=None, faces=None):
        self.vertices = np.array(vertices, dtype=float) if vertices is not None else np.empty((0, 3), dtype=float)
        self.faces = np.array(faces, dtype=int) if faces is not None else np.empty((0, 3), dtype=int)
        self.original_vertices = self.vertices.copy()
        self.original_faces = self.faces.copy()
        self.mesh = None
        self.trimesh_obj = None

    def load_from_trimesh(self, trimesh_obj):
        self.trimesh_obj = trimesh_obj
        self.vertices = np.asarray(trimesh_obj.vertices, dtype=float)
        self.faces = np.asarray(trimesh_obj.faces, dtype=int)
        self.original_vertices = self.vertices.copy()
        self.original_faces = self.faces.copy()

    def load_from_pyvista(self, pyvista_mesh):
        self.mesh = pyvista_mesh
        self.vertices = np.asarray(pyvista_mesh.points, dtype=float)
        self.faces = self._pyvista_faces_to_triangles(np.asarray(pyvista_mesh.faces, dtype=int))
        self.original_vertices = self.vertices.copy()

    def to_trimesh(self):
        if self.trimesh_obj is not None:
            return self.trimesh_obj
        return trimesh.Trimesh(vertices=self.vertices, faces=self.faces, process=False)

    def to_pyvista(self):
        if self.mesh is not None:
            return self.mesh
        return pv.PolyData(self.vertices, self._triangles_to_pyvista_faces(self.faces))

    def get_vertex_count(self):
        return len(self.vertices)

    def get_face_count(self):
        return len(self.faces)

    def get_bounds(self):
        if len(self.vertices) == 0:
            return None
        return np.array([self.vertices.min(axis=0), self.vertices.max(axis=0)])

    def get_center(self):
        bounds = self.get_bounds()
        if bounds is None:
            return np.zeros(3)
        return (bounds[0] + bounds[1]) / 2.0

    def copy(self):
        new_model = MeshModel(self.vertices.copy(), self.faces.copy())
        new_model.original_vertices = self.original_vertices.copy()
        new_model.original_faces = self.original_faces.copy()
        return new_model

    @staticmethod
    def _triangles_to_pyvista_faces(faces):
        if len(faces) == 0:
            return np.empty((0,), dtype=int)
        faces = np.asarray(faces, dtype=int)
        if faces.ndim == 1:
            return faces
        counts = np.full((faces.shape[0], 1), faces.shape[1], dtype=int)
        return np.hstack([counts, faces]).ravel()

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
