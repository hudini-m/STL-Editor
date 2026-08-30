import numpy as np


def inspect_mesh(mesh):
    """Collect read-only mesh diagnostics without repairing or changing the mesh."""
    report = {}

    try:
        report["Vertices"] = str(len(mesh.vertices))
    except Exception:
        report["Vertices"] = "unavailable"
    try:
        report["Faces"] = str(len(mesh.faces))
    except Exception:
        report["Faces"] = "unavailable"
    try:
        report["Finite vertex coordinates"] = "Yes" if np.all(np.isfinite(mesh.vertices)) else "No"
    except Exception:
        report["Finite vertex coordinates"] = "unavailable"
    try:
        mask = mesh.nondegenerate_faces()
        report["Degenerate faces"] = str(int(np.count_nonzero(~np.asarray(mask, dtype=bool))))
    except Exception:
        report["Degenerate faces"] = "unavailable"
    try:
        report["Watertight"] = "Yes" if mesh.is_watertight else "No"
    except Exception:
        report["Watertight"] = "unavailable"
    try:
        report["Winding consistent"] = "Yes" if mesh.is_winding_consistent else "No"
    except Exception:
        report["Winding consistent"] = "unavailable"
    try:
        report["Volume status"] = f"{mesh.volume:.6g}" if mesh.is_volume else "not meaningful"
    except Exception:
        report["Volume status"] = "unavailable"
    try:
        report["Dimensions (X x Y x Z)"] = " x ".join(f"{value:.6g}" for value in mesh.extents)
    except Exception:
        report["Dimensions (X x Y x Z)"] = "unavailable"
    try:
        counts = np.bincount(mesh.edges_unique_inverse, minlength=len(mesh.edges_unique))
        report["Boundary edges"] = str(int(np.count_nonzero(counts == 1)))
        report["Non-manifold edges"] = str(int(np.count_nonzero(counts > 2)))
    except Exception:
        report["Boundary edges"] = "unavailable"
        report["Non-manifold edges"] = "unavailable"
    return report


def format_mesh_validation_report(mesh):
    report = inspect_mesh(mesh)
    lines = ["Mesh Validation Results", "-----------------------"]
    lines.extend(f"{label}: {value}" for label, value in report.items())
    return "\n".join(lines)
