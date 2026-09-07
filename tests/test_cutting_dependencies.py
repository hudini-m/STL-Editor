def test_mesh_cutting_optional_dependencies_are_installed():
    import mapbox_earcut
    import networkx
    import rtree

    assert mapbox_earcut is not None
    assert networkx is not None
    assert rtree is not None
