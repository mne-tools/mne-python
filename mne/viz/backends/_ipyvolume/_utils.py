# -*- coding: utf-8 -*-
# All the code in this file has been adapted from vispy/vispy.
# Next step would be to bundle the original code from vispy
# in externals and keep here only the modifications.
#
# License: BSD (3-clause)
import numpy as np


def _translate(offset, dtype=None):
    """Translate by an offset (x, y, z).

    Parameters
    ----------
    offset : array-like, shape (3,)
        Translation in x, y, z.
    dtype : dtype | None
        Output type (if None, don't cast).

    Returns
    -------
    M : ndarray
        Transformation matrix describing the translation.
    """
    assert len(offset) == 3
    x, y, z = offset
    M = np.array([[1., 0., 0., 0.],
                 [0., 1., 0., 0.],
                 [0., 0., 1., 0.],
                 [x, y, z, 1.0]], dtype)
    return M


def _rotate(angle, axis, dtype=None):
    """Compute the 3x3 rotation matrix for rotation around a given axis.

    Parameters
    ----------
    angle : float
        The angle of rotation, in degrees.
    axis : ndarray
        The x, y, z coordinates of the axis direction vector.

    Returns
    -------
    M : ndarray
        Transformation matrix describing the rotation.
    """
    import math
    angle = np.radians(angle)
    assert len(axis) == 3
    x, y, z = axis / np.linalg.norm(axis)
    c, s = math.cos(angle), math.sin(angle)
    cx, cy, cz = (1 - c) * x, (1 - c) * y, (1 - c) * z
    M = np.array([[cx * x + c, cy * x - z * s, cz * x + y * s, .0],
                  [cx * y + z * s, cy * y + c, cz * y - x * s, 0.],
                  [cx * z - y * s, cy * z + x * s, cz * z + c, 0.],
                  [0., 0., 0., 1.]], dtype).T
    return M


def _create_cone(cols, radius=1.0, length=1.0):
    """Create a cone.

    Parameters
    ----------
    cols : int
        Number of faces.
    radius : float
        Base cone radius.
    length : float
        Length of the cone.

    Returns
    -------
    cone : MeshData
        Vertices and faces computed for a cone surface.
    """
    verts = np.empty((cols + 1, 3), dtype=np.float32)
    # compute vertices
    th = np.linspace(2 * np.pi, 0, cols + 1).reshape(1, cols + 1)
    verts[:-1, 2] = 0.0
    verts[:-1, 0] = radius * np.cos(th[0, :-1])  # x = r cos(th)
    verts[:-1, 1] = radius * np.sin(th[0, :-1])  # y = r sin(th)
    # Add the extremity
    verts[-1, 0] = 0.0
    verts[-1, 1] = 0.0
    verts[-1, 2] = length
    verts = verts.reshape((cols + 1), 3)  # just reshape: no redundant vertices
    # compute faces
    faces = np.empty((cols, 3), dtype=np.uint32)
    template = np.array([[0, 1]])
    for pos in range(cols):
        faces[pos, :-1] = template + pos
    faces[:, 2] = cols
    faces[-1, 1] = 0
    return verts, faces


def _create_arrow(rows, cols, radius=0.1, length=1.0,
                  cone_radius=None, cone_length=None):
    """Create a 3D arrow using a cylinder plus cone.

    Parameters
    ----------
    rows : int
        Number of rows.
    cols : int
        Number of columns.
    radius : float
        Base cylinder radius.
    length : float
        Length of the arrow.
    cone_radius : float
        Radius of the cone base.
           If None, then this defaults to 2x the cylinder radius.
    cone_length : float
        Length of the cone.
           If None, then this defaults to 1/3 of the arrow length.

    Returns
    -------
    arrow : MeshData
        Vertices and faces computed for a cone surface.
    """
    # create the cylinder
    cyl_verts = None
    if cone_radius is None:
        cone_radius = radius * 2.0
    if cone_length is None:
        con_L = length / 3.0
        cyl_L = length * 2.0 / 3.0
    else:
        cyl_L = max(0, length - cone_length)
        con_L = min(cone_length, length)
    if cyl_L != 0:
        cyl_verts, cyl_faces = _create_cylinder(rows, cols,
                                                radius=[radius, radius],
                                                length=cyl_L)
    # create the cone
    con_verts, con_faces = _create_cone(cols, radius=cone_radius, length=con_L)
    verts = con_verts
    nbr_verts_con = verts.size // 3
    faces = con_faces
    if cyl_verts is not None:
        trans = np.array([[0.0, 0.0, cyl_L]])
        verts = np.vstack((verts + trans, cyl_faces))
        faces = np.vstack((faces, cyl_faces + nbr_verts_con))
    return verts, faces


def _create_cylinder(rows, cols, radius=[1.0, 1.0], length=1.0, offset=False):
    """Create a cylinder.

    Parameters
    ----------
    rows : int
        Number of rows.
    cols : int
        Number of columns.
    radius : tuple of float
        Cylinder radii.
    length : float
        Length of the cylinder.
    offset : bool
        Rotate each row by half a column.

    Returns
    -------
    cylinder : MeshData
        Vertices and faces computed for a cylindrical surface.
    """
    verts = np.empty((rows + 1, cols, 3), dtype=np.float32)
    if isinstance(radius, int):
        radius = [radius, radius]  # convert to list
    # compute vertices
    th = np.linspace(2 * np.pi, 0, cols).reshape(1, cols)
    # radius as a function of z
    r = np.linspace(radius[0], radius[1], num=rows + 1,
                    endpoint=True).reshape(rows + 1, 1)
    verts[..., 2] = np.linspace(0, length, num=rows + 1,
                                endpoint=True).reshape(rows + 1, 1)  # z
    if offset:
        # rotate each row by 1/2 column
        th = th + ((np.pi / cols) * np.arange(rows + 1).reshape(rows + 1, 1))
    verts[..., 0] = r * np.cos(th)  # x = r cos(th)
    verts[..., 1] = r * np.sin(th)  # y = r sin(th)
    # just reshape: no redundant vertices...
    verts = verts.reshape((rows + 1) * cols, 3)
    # compute faces
    faces = np.empty((rows * cols * 2, 3), dtype=np.uint32)
    rowtemplate1 = (((np.arange(cols).reshape(cols, 1) +
                      np.array([[0, 1, 0]])) % cols) +
                    np.array([[0, 0, cols]]))
    rowtemplate2 = (((np.arange(cols).reshape(cols, 1) +
                      np.array([[0, 1, 1]])) % cols) +
                    np.array([[cols, 0, cols]]))
    for row in range(rows):
        start = row * cols * 2
        faces[start:start + cols] = rowtemplate1 + row * cols
        faces[start + cols:start + (cols * 2)] = rowtemplate2 + row * cols
    return verts, faces


def _create_sphere(rows, cols, radius, offset=True):
    """Create a sphere.

    Parameters
    ----------
    rows : int
        Number of rows (for method='latitude' and 'cube').
    cols : int
        Number of columns (for method='latitude' and 'cube').
    depth : int
        Number of depth segments (for method='cube').
    radius : float
        Sphere radius.
    offset : bool
        Rotate each row by half a column (for method='latitude').
    subdivisions : int
        Number of subdivisions to perform (for method='ico')
    method : str
        Method for generating sphere. Accepts 'latitude' for latitude-
        longitude, 'ico' for icosahedron, and 'cube' for cube based
        tessellation.

    Returns
    -------
    sphere : MeshData
        Vertices and faces computed for a spherical surface.
    """
    verts = np.empty((rows + 1, cols, 3), dtype=np.float32)
    # compute vertices
    phi = (np.arange(rows + 1) * np.pi / rows).reshape(rows + 1, 1)
    s = radius * np.sin(phi)
    verts[..., 2] = radius * np.cos(phi)
    th = ((np.arange(cols) * 2 * np.pi / cols).reshape(1, cols))
    if offset:
        # rotate each row by 1/2 column
        th = th + ((np.pi / cols) * np.arange(rows + 1).reshape(rows + 1, 1))
    verts[..., 0] = s * np.cos(th)
    verts[..., 1] = s * np.sin(th)
    # remove redundant vertices from top and bottom
    verts = verts.reshape((rows + 1) * cols, 3)[cols - 1:-(cols - 1)]

    # compute faces
    faces = np.empty((rows * cols * 2, 3), dtype=np.uint32)
    rowtemplate1 = (((np.arange(cols).reshape(cols, 1) +
                      np.array([[1, 0, 0]])) % cols) +
                    np.array([[0, 0, cols]]))
    rowtemplate2 = (((np.arange(cols).reshape(cols, 1) +
                      np.array([[1, 0, 1]])) % cols) +
                    np.array([[0, cols, cols]]))
    for row in range(rows):
        start = row * cols * 2
        faces[start:start + cols] = rowtemplate1 + row * cols
        faces[start + cols:start + (cols * 2)] = rowtemplate2 + row * cols
    # cut off zero-area triangles at top and bottom
    faces = faces[cols:-cols]

    # adjust for redundant vertices that were removed from top and bottom
    vmin = cols - 1
    faces[faces < vmin] = vmin
    faces -= vmin
    vmax = verts.shape[0] - 1
    faces[faces > vmax] = vmax
    return verts, faces


def _check_vertices_shape(vertices):
    """Check vertices shape."""
    if vertices.shape[1] <= 3:
        verts = vertices
    elif vertices.shape[1] == 4:
        verts = vertices[:, :-1]
    else:
        verts = None
    return verts


def _isoline(vertices, tris, vertex_data, levels):
    """Generate an isocurve from vertex data in a surface mesh.

    Parameters
    ----------
    vertices : ndarray, shape (Nv, 3)
        Vertex coordinates.
    tris : ndarray, shape (Nf, 3)
        Indices of triangular element into the vertices array.
    vertex_data : ndarray, shape (Nv,)
        data at vertex.
    levels : ndarray, shape (Nl,)
        Levels at which to generate an isocurve

    Returns
    -------
    lines : ndarray, shape (Nvout, 3)
        Vertex coordinates for lines points
    connects : ndarray, shape (Ne, 2)
        Indices of line element into the vertex array.
    vertex_level: ndarray, shape (Nvout,)
        level for vertex in lines

    Notes
    -----
    Uses a marching squares algorithm to generate the isolines.
    """
    lines, connects, vertex_level, level_index = (None, None, None, None)
    if not all([isinstance(x, np.ndarray) for x in (vertices, tris,
                vertex_data, levels)]):
        raise ValueError('all inputs must be numpy arrays')
    verts = _check_vertices_shape(vertices)
    if (verts is not None and tris.shape[1] == 3 and
            vertex_data.shape[0] == verts.shape[0]):
        edges = np.vstack((tris.reshape((-1)),
                           np.roll(tris, -1, axis=1).reshape((-1)))).T
        edge_datas = vertex_data[edges]
        edge_coors = verts[edges].reshape(tris.shape[0] * 3, 2, 3)
        for lev in levels:
            # index for select edges with vertices have only False - True
            # or True - False at extremity
            index = (edge_datas >= lev)
            index = index[:, 0] ^ index[:, 1]  # xor calculation
            # Selectect edge
            edge_datas_Ok = edge_datas[index, :]
            xyz = edge_coors[index]
            # Linear interpolation
            ratio = np.array([(lev - edge_datas_Ok[:, 0]) /
                              (edge_datas_Ok[:, 1] - edge_datas_Ok[:, 0])])
            point = xyz[:, 0, :] + ratio.T * (xyz[:, 1, :] - xyz[:, 0, :])
            nbr = point.shape[0] // 2
            if connects is None:
                lines = point
                connects = np.arange(0, nbr * 2).reshape((nbr, 2))
                vertex_level = np.zeros(len(point)) + lev
                level_index = np.array(len(point))
            else:
                connect = np.arange(0, nbr * 2).reshape((nbr, 2)) + \
                    len(lines)
                connects = np.append(connects, connect, axis=0)
                lines = np.append(lines, point, axis=0)
                vertex_level = np.append(vertex_level,
                                         np.zeros(len(point)) +
                                         lev)
                level_index = np.append(level_index, np.array(len(point)))
            vertex_level = vertex_level.reshape((vertex_level.size, 1))

    return lines, connects, vertex_level, level_index
