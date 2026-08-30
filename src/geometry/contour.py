import numpy as np

from geometry.projection import project_points

try:
    from shapely.geometry import Polygon
    from shapely.ops import unary_union
except ImportError:  # pragma: no cover
    Polygon = None
    unary_union = None


class ContourExtractor:
    def __init__(self, mesh_model):
        self.mesh_model = mesh_model

    def extract_outer_contour(self, plane="XY"):
        """Return the concave outer boundary of the projected triangle union."""
        if Polygon is None:
            raise RuntimeError("The shapely package is required for silhouette extraction")
        vertices_2d = project_points(self.mesh_model.vertices, plane)
        polygons = []
        for face in self.mesh_model.faces:
            triangle = vertices_2d[np.asarray(face, dtype=int)]
            polygon = Polygon(triangle)
            if not polygon.is_empty and polygon.area > 1e-12:
                polygons.append(polygon)
        if not polygons:
            return np.empty((0, 2), dtype=float)
        boundary = unary_union(polygons).buffer(0)
        if boundary.geom_type == "MultiPolygon":
            boundary = max(boundary.geoms, key=lambda item: item.area)
        return np.asarray(boundary.exterior.coords[:-1], dtype=float)

    def sample_contour(self, contour, count=60):
        """Resample a closed contour at approximately equal arc-length intervals."""
        contour = np.asarray(contour, dtype=float)
        if len(contour) < 2 or count < 1:
            return np.empty((0, 2), dtype=float)
        closed = np.vstack([contour, contour[0]])
        segments = np.linalg.norm(np.diff(closed, axis=0), axis=1)
        cumulative = np.concatenate([[0.0], np.cumsum(segments)])
        perimeter = cumulative[-1]
        if perimeter <= 1e-12:
            return np.repeat(contour[:1], count, axis=0)
        targets = np.linspace(0.0, perimeter, count, endpoint=False)
        samples = []
        for target in targets:
            index = min(int(np.searchsorted(cumulative, target, side="right") - 1), len(contour) - 1)
            length = segments[index]
            ratio = 0.0 if length <= 1e-12 else (target - cumulative[index]) / length
            samples.append(closed[index] + ratio * (closed[index + 1] - closed[index]))
        return np.asarray(samples, dtype=float)

    def extract_contour(self, view_direction=None):
        return list(range(len(self.mesh_model.vertices)))

    def get_mesh_edges(self):
        if len(self.mesh_model.faces) == 0:
            return []
        edges = set()
        for face in self.mesh_model.faces:
            for index in range(len(face)):
                edge = tuple(sorted((int(face[index]), int(face[(index + 1) % len(face)]))))
                edges.add(edge)
        return list(edges)
