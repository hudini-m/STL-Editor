from setuptools import setup, find_packages


setup(
    name="stl-changer",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.24.3",
        "trimesh>=4.0.9",
        "pyvista>=0.38.5",
        "vtk>=9.2.6,<9.3",
        "PySide6>=6.4.2",
        "pyvistaqt>=0.11.1",
        "scipy>=1.10.1",
        "shapely>=2.0.6",
    ],
    entry_points={
        "console_scripts": [
            "stl-changer=src.app:main",
        ],
    },
    author="STL Changer Developer",
    description="Desktop application for editing STL meshes",
)
