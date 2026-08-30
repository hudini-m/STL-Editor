import numpy as np


PLANE_AXES = {
    "XY": (0, 1, 2),
    "XZ": (0, 2, 1),
    "YZ": (1, 2, 0),
}


def project_points(points, plane):
    axes = PLANE_AXES[plane]
    values = np.asarray(points, dtype=float)
    return values[..., [axes[0], axes[1]]]


def lift_points(points_2d, depth, plane):
    axes = PLANE_AXES[plane]
    values = np.asarray(points_2d, dtype=float)
    depth_values = np.asarray(depth, dtype=float)
    result = np.zeros(values.shape[:-1] + (3,), dtype=float)
    result[..., axes[0]] = values[..., 0]
    result[..., axes[1]] = values[..., 1]
    result[..., axes[2]] = depth_values
    return result


def plane_normal_axis(plane):
    return PLANE_AXES[plane][2]
