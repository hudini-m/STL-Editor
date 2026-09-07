import numpy as np


class ControlPointManager:
    def __init__(self, mesh_model):
        self.mesh_model = mesh_model
        self.control_points = []
        self.active_plane = "XY"
        self.contour_2d = np.empty((0, 2), dtype=float)
        self.selected_point = None
        self.dragging_point = None

    def add_control_point(self, position, pinned=False):
        self.control_points.append((np.array(position, dtype=float), bool(pinned)))

    def remove_control_point(self, index):
        if 0 <= index < len(self.control_points):
            del self.control_points[index]
            if self.selected_point == index:
                self.selected_point = None
            if self.dragging_point == index:
                self.dragging_point = None

    def clear_all_points(self):
        self.control_points = []
        self.selected_point = None
        self.dragging_point = None

    def get_control_points(self):
        return [cp[0] for cp in self.control_points]

    def get_pinned_points(self):
        return [i for i, (_, pinned) in enumerate(self.control_points) if pinned]

    def set_point_position(self, index, new_position):
        if 0 <= index < len(self.control_points):
            _, pinned = self.control_points[index]
            self.control_points[index] = (np.array(new_position, dtype=float), pinned)

    def select_point(self, index):
        if 0 <= index < len(self.control_points):
            self.selected_point = index

    def get_selected_point(self):
        return self.selected_point

    def is_point_pinned(self, index):
        if 0 <= index < len(self.control_points):
            return self.control_points[index][1]
        return False

    def pin_point(self, index):
        if 0 <= index < len(self.control_points):
            pos, _ = self.control_points[index]
            self.control_points[index] = (pos, True)

    def unpin_point(self, index):
        if 0 <= index < len(self.control_points):
            pos, _ = self.control_points[index]
            self.control_points[index] = (pos, False)

    def toggle_anchor(self, index):
        if self.is_point_pinned(index):
            self.unpin_point(index)
        else:
            self.pin_point(index)

    def get_count(self):
        return len(self.control_points)

    def start_dragging(self, point_index):
        if 0 <= point_index < len(self.control_points) and not self.is_point_pinned(point_index):
            self.dragging_point = point_index

    def stop_dragging(self):
        self.dragging_point = None

    def get_dragging_point(self):
        return self.dragging_point

    def update_dragged_point_position(self, new_position):
        if self.dragging_point is not None:
            _, pinned = self.control_points[self.dragging_point]
            self.control_points[self.dragging_point] = (np.array(new_position, dtype=float), pinned)

    def move_point_locally(self, index, new_position, neighbor_count=4):
        """Move one ordered contour point and smoothly move nearby points."""
        if not (0 <= index < len(self.control_points)) or self.is_point_pinned(index):
            return False
        count = len(self.control_points)
        if count < 2:
            self.set_point_position(index, new_position)
            return True
        old_position = self.control_points[index][0].copy()
        new_position = np.asarray(new_position, dtype=float)
        delta = new_position - old_position
        original = [(position.copy(), pinned) for position, pinned in self.control_points]
        anchors = self.get_pinned_points()
        anchored = self._move_between_anchors(index, delta, anchors)
        if not anchored:
            radius = max(1, min(int(neighbor_count), (count - 1) // 2))
            updates = []
            for offset in range(-radius, radius + 1):
                point_index = (index + offset) % count
                normalized = abs(offset) / radius
                smooth = self._smootherstep(normalized)
                weight = 1.0 - smooth
                position, pinned = self.control_points[point_index]
                if pinned and point_index != index:
                    weight = 0.0
                updates.append((point_index, position + delta * weight, pinned))
            for point_index, position, pinned in updates:
                self.control_points[point_index] = (np.asarray(position, dtype=float), pinned)
        if not self._is_valid_contour():
            self.control_points = original
            return False
        self._update_contour_from_points()
        return True

    def move_point_exact(self, index, new_position):
        """Move only one contour point, retaining the same validity checks."""
        if not (0 <= index < len(self.control_points)) or self.is_point_pinned(index):
            return False
        original = [(position.copy(), pinned) for position, pinned in self.control_points]
        position, pinned = self.control_points[index]
        self.control_points[index] = (np.asarray(new_position, dtype=float), pinned)
        if not self._is_valid_contour():
            self.control_points = original
            return False
        self._update_contour_from_points()
        return True

    @staticmethod
    def _smootherstep(value):
        value = float(np.clip(value, 0.0, 1.0))
        return value * value * value * (value * (value * 6.0 - 15.0) + 10.0)

    def get_active_region(self, index, neighbor_count=4):
        """Return the cyclic indices affected by a drag before it starts."""
        count = len(self.control_points)
        if not (0 <= index < count):
            return []
        anchors = self.get_pinned_points()
        if len(anchors) >= 2:
            start, end = self._nearest_anchor_pair(index, anchors)
            if start is not None and end is not None and start != end:
                region = []
                cursor = start
                while True:
                    region.append(cursor)
                    if cursor == end:
                        return region
                    cursor = (cursor + 1) % count
        radius = max(1, min(int(neighbor_count), (count - 1) // 2))
        return [(index + offset) % count for offset in range(-radius, radius + 1)]

    def _move_between_anchors(self, index, delta, anchors):
        """Apply smooth arc-length weights only within the nearest anchor interval."""
        if len(anchors) < 2:
            return False
        count = len(self.control_points)

        start, end = self._nearest_anchor_pair(index, anchors)
        if start is None or end is None or start == end:
            return False
        path = []
        cursor = start
        while True:
            path.append(cursor)
            if cursor == end:
                break
            cursor = (cursor + 1) % count
        if index not in path:
            return False

        from geometry.projection import project_points

        positions = np.asarray(self.get_control_points(), dtype=float)
        projected = project_points(positions[path], self.active_plane)
        lengths = np.linalg.norm(np.diff(projected, axis=0), axis=1)
        cumulative = np.concatenate([[0.0], np.cumsum(lengths)])
        selected_offset = path.index(index)
        selected_distance = cumulative[selected_offset]
        total = cumulative[-1]
        if selected_distance <= 1e-12 or total - selected_distance <= 1e-12:
            return False

        for offset, point_index in enumerate(path):
            distance = cumulative[offset]
            if distance <= selected_distance:
                ratio = distance / selected_distance
                weight = self._smootherstep(ratio)
            else:
                ratio = (distance - selected_distance) / (total - selected_distance)
                weight = 1.0 - self._smootherstep(ratio)
            position, pinned = self.control_points[point_index]
            if pinned:
                weight = 0.0
            self.control_points[point_index] = (position + delta * weight, pinned)
        return True

    def _nearest_anchor_pair(self, index, anchors):
        count = len(self.control_points)

        def find_anchor(step):
            for distance in range(1, count):
                candidate = (index + step * distance) % count
                if candidate in anchors:
                    return candidate
            return None

        return find_anchor(-1), find_anchor(1)

    def _is_valid_contour(self):
        from geometry.projection import project_points

        points = project_points(self.get_control_points(), self.active_plane)
        if len(points) < 3 or not np.all(np.isfinite(points)):
            return False
        closed = np.vstack([points, points[0]])
        lengths = np.linalg.norm(np.diff(closed, axis=0), axis=1)
        if np.min(lengths) <= 1e-8:
            return False
        pairwise = np.linalg.norm(points[:, np.newaxis, :] - points[np.newaxis, :, :], axis=2)
        for first in range(len(points)):
            for second in range(first + 1, len(points)):
                if second in (first + 1, len(points) - 1) and first == 0:
                    continue
                if second == first + 1:
                    continue
                if pairwise[first, second] <= 1e-8:
                    return False
        signed_area = 0.5 * abs(np.dot(points[:, 0], np.roll(points[:, 1], -1)) - np.dot(points[:, 1], np.roll(points[:, 0], -1)))
        if signed_area <= 1e-10:
            return True
        try:
            from shapely.geometry import LineString

            return LineString(closed).is_simple
        except ImportError:  # pragma: no cover
            return True

    def _update_contour_from_points(self):
        from geometry.projection import project_points

        if self.control_points:
            self.contour_2d = project_points(self.get_control_points(), self.active_plane)

    def auto_generate_contour_points(self, density=60, plane="XY"):
        from geometry.contour import ContourExtractor
        from geometry.projection import lift_points, plane_normal_axis, project_points

        if self.mesh_model is None:
            return 0

        extractor = ContourExtractor(self.mesh_model)
        contour = extractor.extract_outer_contour(plane)
        sampled_points = extractor.sample_contour(contour, int(density))
        projected_vertices = project_points(self.mesh_model.vertices, plane)
        normal_axis = plane_normal_axis(plane)
        self.clear_all_points()
        self.active_plane = plane
        self.contour_2d = np.asarray(contour, dtype=float)
        for point in sampled_points:
            nearest = int(np.argmin(np.linalg.norm(projected_vertices - point, axis=1)))
            depth = self.mesh_model.vertices[nearest, normal_axis]
            self.add_control_point(lift_points(point, depth, plane))
        return len(sampled_points)

    def generate_uniform_points(self, num_points=10):
        from geometry.contour import ContourExtractor

        if self.mesh_model is None:
            return 0

        extractor = ContourExtractor(self.mesh_model)
        points = extractor.generate_uniform_contour_points(num_points)
        for point in points:
            self.add_control_point(point)
        return len(points)
