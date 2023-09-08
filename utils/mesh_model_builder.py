import math

import numpy as np
from discretize.base import BaseMesh


def get_indices_ellipsoid(center, a, b, c, cell_centers):
    if isinstance(cell_centers, BaseMesh):
        cell_centers = cell_centers.cell_centers

    # Validation: mesh and point (p0) live in the same dimensional space
    dim_mesh = np.size(cell_centers[0, :])
    assert len(center) == dim_mesh, "Dimension mismatch. len(p0) != dim_mesh"

    if dim_mesh == 3:
        # Define the points
        ind = ((((cell_centers[:, 0] - center[0]) ** 2) / (a ** 2)
                + ((cell_centers[:, 1] - center[1]) ** 2) / (b ** 2)
                + ((cell_centers[:, 2] - center[2]) ** 2) / (c ** 2)) < 1)
    else:
        ind = ()

    # Return a tuple
    return ind


def get_indices_cylinder(center, radius, height, cell_centers, along_axis="Y"):
    if isinstance(cell_centers, BaseMesh):
        cell_centers = cell_centers.cell_centers

    # Validation: mesh and point (p0) live in the same dimensional space
    dim_mesh = np.size(cell_centers[0, :])
    assert len(center) == dim_mesh, "Dimension mismatch. len(p0) != dim_mesh"

    if dim_mesh == 3 and along_axis == "Y":
        # Define the points
        a = (cell_centers[:, 0] - center[0]) ** 2 + (cell_centers[:, 2] - center[2]) ** 2 < radius ** 2
        b = (cell_centers[:, 1] > (center[1] - height))
        c = (cell_centers[:, 1] < (center[1] + height))
        ind = a & b & c
    else:
        ind = ()

    # Return a tuple
    return ind


def get_indices_smallest_block(center, hx, cell_centers):
    if isinstance(cell_centers, BaseMesh):
        cell_centers = cell_centers.cell_centers

    # Validation: mesh and point (p0) live in the same dimensional space
    dim_mesh = np.size(cell_centers[0, :])
    assert len(center) == dim_mesh, "Dimension mismatch. len(p0) != dim_mesh"

    if dim_mesh == 3:
        half_hx = hx / 2
        # Define the points
        a = np.abs(cell_centers[:, 0] - center[0]) < half_hx
        b = np.abs(cell_centers[:, 1] - center[1]) < half_hx
        c = np.abs(cell_centers[:, 2] - center[2]) < half_hx
        ind = a & b & c
    else:
        ind = ()

    # Return a tuple
    return ind
