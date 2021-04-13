"""
.. _tut-fix-meshes:

Editing BEM surfaces in Blender
===============================

Sometimes when creating a BEM model the surfaces need manual correction because
of a series of problems that can arise (e.g. intersection between surfaces).
Here, we will see how this can be achieved by exporting the surfaces to the 3D
modeling program `Blender <https://blender.org>`_, editing them, and
re-importing them.

This tutorial is based on https://github.com/ezemikulan/blender_freesurfer by
Ezequiel Mikulan.

.. contents:: Page contents
   :local:
   :depth: 2

"""

# Authors: Marijn van Vliet <w.m.vanvliet@gmail.com>
#          Ezequiel Mikulan <e.mikulan@gmail.com>
#          Manorama Kadwani <manorama.kadwani@gmail.com>
#
# License: BSD (3-clause)

# sphinx_gallery_thumbnail_path = '_static/blender_import_obj/blender_import_obj2.jpg'  # noqa

import os
import os.path as op
import shutil
import mne

data_path = mne.datasets.sample.data_path()
subjects_dir = op.join(data_path, 'subjects')
bem_dir = op.join(subjects_dir, 'sample', 'bem', 'flash')
###############################################################################
# Exporting surfaces to Blender
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# In this tutorial, we are working with the MNE-Sample set, for which the
# surfaces have no issues. To demonstrate how to fix problematic surfaces, we
# are going to manually place one of the inner-skull vertices outside the
# outer-skill mesh.
#
# We then convert the surfaces to `.obj
# <https://en.wikipedia.org/wiki/Wavefront_.obj_file>`_ files and create a new
# folder called ``conv`` inside the FreeSurfer subject folder to keep them in.

# Put the converted surfaces in a separate 'conv' folder
conv_dir = op.join(subjects_dir, 'sample', 'conv')
os.makedirs(conv_dir, exist_ok=True)

# Load the inner skull surface and create a problem
# The metadata is empty in this example. In real study, we want to write the
# original metadata to the fixed surface file. Set read_metadata=True to do so.
coords, faces = mne.read_surface(op.join(bem_dir, 'inner_skull.surf'))
coords[0] *= 1.1  # Move the first vertex outside the skull

# Write the inner skull surface as an .obj file that can be imported by
# Blender.
mne.write_surface(op.join(conv_dir, 'inner_skull.obj'), coords, faces,
                  overwrite=True)

# Also convert the outer skull surface.
coords, faces = mne.read_surface(op.join(bem_dir, 'outer_skull.surf'))
mne.write_surface(op.join(conv_dir, 'outer_skull.obj'), coords, faces,
                  overwrite=True)

###############################################################################
# Editing in Blender
# ^^^^^^^^^^^^^^^^^^
#
# We can now open Blender and import the surfaces. Go to *File > Import >
# Wavefront (.obj)*. Navigate to the ``conv`` folder and select the file you
# want to import. Make sure to select the *Keep Vert Order* option. You can
# also select the *Y Forward* option to load the axes in the correct direction
# (RAS):
#
# .. image:: ../../_static/blender_import_obj/blender_import_obj1.jpg
#    :width: 800
#    :alt: Importing .obj files in Blender
#
# For convenience, you can save these settings by pressing the ``+`` button
# next to *Operator Presets*.
#
# Repeat the procedure for all surfaces you want to import (e.g. inner_skull
# and outer_skull).
#
# You can now edit the surfaces any way you like. See the
# `Beginner Blender Tutorial Series
# <https://www.youtube.com/playlist?list=PLxLGgWrla12dEW5mjO09kR2_TzPqDTXdw>`_
# to learn how to use Blender. Specifically, `part 2
# <http://www.youtube.com/watch?v=RaT-uG5wgUw&t=5m30s>`_ will teach you how to
# use the basic editing tools you need to fix the surface.
#
# .. image:: ../../_static/blender_import_obj/blender_import_obj2.jpg
#    :width: 800
#    :alt: Editing surfaces in Blender
#
# Using the fixed surfaces in MNE-Python
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# In Blender, you can export a surface as an .obj file by selecting it and go
# to *File > Export > Wavefront (.obj)*. You need to again select the *Y
# Forward* option and check the *Keep Vertex Order* box.
#
# .. image:: ../../_static/blender_import_obj/blender_import_obj3.jpg
#    :width: 200
#    :alt: Exporting .obj files in Blender
#
#
# Each surface needs to be exported as a separate file. We recommend saving
# them in the ``conv`` folder and ending the file name with ``_fixed.obj``,
# although this is not strictly necessary.
#
# In order to be able to run this tutorial script top to bottom, we here
# simulate the edits you did manually in Blender using Python code:

coords, faces = mne.read_surface(op.join(conv_dir, 'inner_skull.obj'))
coords[0] /= 1.1  # Move the first vertex back inside the skull
mne.write_surface(op.join(conv_dir, 'inner_skull_fixed.obj'), coords, faces,
                  overwrite=True)

###############################################################################
# Back in Python, you can read the fixed .obj files and save them as
# FreeSurfer .surf files. For the :func:`mne.make_bem_model` function to find
# them, they need to be saved using their original names in the ``surf``
# folder, e.g. ``bem/inner_skull.surf``. Be sure to first backup the original
# surfaces in case you make a mistake!

# Read the fixed surface
coords, faces = mne.read_surface(op.join(conv_dir, 'inner_skull_fixed.obj'))

# Backup the original surface
shutil.copy(op.join(bem_dir, 'inner_skull.surf'),
            op.join(bem_dir, 'inner_skull_orig.surf'))

# Overwrite the original surface with the fixed version
# In real study you should provide the correct metadata using ``volume_info=``
# This could be accomplished for example with:
#
# _, _, vol_info = mne.read_surface(op.join(bem_dir, 'inner_skull.surf'),
#                                   read_metadata=True)
# mne.write_surface(op.join(bem_dir, 'inner_skull.surf'), coords, faces,
#                   volume_info=vol_info, overwrite=True)

###############################################################################
# Editing the head surfaces
# ^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Sometimes the head surfaces are faulty and require manual editing. We use
# :func:`mne.write_head_bem` to convert the fixed surfaces to ``.fif`` files.
#
# Low-resolution head
# ~~~~~~~~~~~~~~~~~~~
#
# For EEG forward modeling, it is possible that ``outer_skin.surf`` would be
# manually edited. In that case, remember to save the fixed version of
# ``-head.fif`` from the edited surface file for coregistration.

# Load the fixed surface
coords, faces = mne.read_surface(op.join(bem_dir, 'outer_skin.surf'))

# Make sure we are in the correct directory
head_dir = op.dirname(bem_dir)

# Remember to backup the original head file in advance!
# Overwrite the original head file
#
# mne.write_head_bem(op.join(head_dir, 'sample-head.fif'), coords, faces,
#                    overwrite=True)

###############################################################################
# High-resolution head
# ~~~~~~~~~~~~~~~~~~~~
#
# We use :func:`mne.read_bem_surfaces` to read the head surface files. After
# editing, we again output the head file with :func:`mne.write_head_bem`.
# Here we use ``-head.fif`` for speed.

# If ``-head-dense.fif`` does not exist, you need to run
# ``mne make_scalp_surfaces`` first.
# [0] because a list of surfaces is returned
surf = mne.read_bem_surfaces(op.join(head_dir, 'sample-head.fif'))[0]

# For consistency only
coords = surf['rr']
faces = surf['tris']

# Write the head as an .obj file for editing
mne.write_surface(op.join(conv_dir, 'sample-head.obj'),
                  coords, faces, overwrite=True)

# Usually here you would go and edit your meshes.
#
# Here we just use the same surface as if it were fixed
# Read in the .obj file
coords, faces = mne.read_surface(op.join(conv_dir, 'sample-head.obj'))

# Remember to backup the original head file in advance!
# Overwrite the original head file
#
# mne.write_head_bem(op.join(head_dir, 'sample-head.fif'), coords, faces,
#                    overwrite=True)

###############################################################################
# That's it! You are ready to continue with your analysis pipeline (e.g.
# running :func:`mne.make_bem_model`).
#
# What if you still get an error?
# ---------------------------------
#
# When editing BEM surfaces/meshes in Blender, make sure to use
# tools that do not change the number or order of vertices, or the geometry
# of triangular faces. For example, avoid the extrusion tool, because it
# duplicates the extruded vertices.
#
# Below are some examples of errors you might encounter when running the
# `mne.make_bem_model` function, and the likely causes of those errors.
#
#
# 1. Cannot decimate to requested ico grade
#
#    This error is caused by having too few or too many vertices. The full
#    error is something like:
#
#    .. code-block:: console
#
#       RuntimeError: Cannot decimate to requested ico grade 4. The provided
#       BEM surface has 20516 triangles, which cannot be isomorphic with a
#       subdivided icosahedron. Consider manually decimating the surface to a
#       suitable density and then use ico=None in make_bem_model.
#
# 2. Surface inner skull has topological defects
#
#    This error can occur when trying to match the original number of
#    triangles by removing vertices. The full error looks like:
#
#    .. code-block:: console
#
#       RuntimeError: Surface inner skull has topological defects: 12 / 20484
#       vertices have fewer than three neighboring triangles [733, 1014, 2068,
#       7732, 8435, 8489, 10181, 11120, 11121, 11122, 11304, 11788]
#
# 3. Surface inner skull is not complete
#
#    This error (like the previous error) reflects a problem with the surface
#    topology (i.e., the expected pattern of vertices/edges/faces is
#    disrupted).
#
#    .. code-block:: console
#
#       RuntimeError: Surface inner skull is not complete (sum of solid
#       angles yielded 0.999668, should be 1.)
#
# 4. Triangle ordering is wrong
#
#    This error reflects a mismatch between how the surface is represented in
#    memory (the order of the vertex/face definitions) and what is expected by
#    MNE-Python.  The full error is:
#
#    .. code-block:: console
#
#       RuntimeError: The source surface has a matching number of
#       triangles but ordering is wrong
#
#
# For any of these errors, it is usually easiest to start over with the
# unedited BEM surface and try again, making sure to only *move* vertices and
# faces without *adding* or *deleting* any. For example,
# select a circle of vertices, then press :kbd:`G` to drag them to the desired
# location. Smoothing a group of selected vertices in Blender (by
# right-clicking and selecting "Smooth Vertices") can also be helpful.
