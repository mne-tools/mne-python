"""Set of tools to interact with Freesurfer data
"""

# Authors: Alexandre Gramfort <gramfort@nmr.mgh.harvard.edu>
#
# License: BSD (3-clause)

import numpy as np


def fread3(fobj):
    """Docstring"""
    b1 = np.fromfile(fobj, ">u1", 1)[0]
    b2 = np.fromfile(fobj, ">u1", 1)[0]
    b3 = np.fromfile(fobj, ">u1", 1)[0]
    return (b1 << 16) + (b2 << 8) + b3


def read_curvature(filepath):
    """Load in curavature values from the ?h.curv file."""
    with open(filepath, "rb") as fobj:
        magic = fread3(fobj)
        if magic == 16777215:
            vnum = np.fromfile(fobj, ">i4", 3)[0]
            curv = np.fromfile(fobj, ">f4", vnum)
        else:
            vnum = magic
            fread3(fobj)
            curv = np.fromfile(fobj, ">i2", vnum) / 100
        bin_curv = 1 - np.array(curv != 0, np.int)
    return bin_curv


def read_surface(filepath):
    """Load in a Freesurfer surface mesh in triangular format."""
    with open(filepath, "rb") as fobj:
        magic = fread3(fobj)
        if magic == 16777215:
            raise NotImplementedError("Quadrangle surface format reading not "
                                      "implemented")
        elif magic != 16777214:
            raise ValueError("File does not appear to be a Freesurfer surface")
        create_stamp = fobj.readline()
        blankline = fobj.readline()
        del blankline
        vnum = np.fromfile(fobj, ">i4", 1)[0]
        fnum = np.fromfile(fobj, ">i4", 1)[0]
        vertex_coords = np.fromfile(fobj, ">f4", vnum * 3).reshape(vnum, 3)
        faces = np.fromfile(fobj, ">i4", fnum * 3).reshape(fnum, 3)
    return vertex_coords, faces
