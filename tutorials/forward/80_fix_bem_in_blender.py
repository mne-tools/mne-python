# -*- coding: utf-8 -*-
"""
.. _tut-fix-meshes:

============================
Fixing BEM and head surfaces
============================

Sometimes when creating a BEM model the surfaces need manual correction because
of a series of problems that can arise (e.g. intersection between surfaces).
Here, we will see how this can be achieved by exporting the surfaces to the 3D
modeling program `Blender <https://blender.org>`_, editing them, and
re-importing them. We will also give a simple example of how to use
:ref:`pymeshfix <tut-fix-meshes-pymeshfix>` to fix topological problems.

Much of this tutorial is based on
https://github.com/ezemikulan/blender_freesurfer by Ezequiel Mikulan.
"""

# Authors: Marijn van Vliet <w.m.vanvliet@gmail.com>
#          Ezequiel Mikulan <e.mikulan@gmail.com>
#          Manorama Kadwani <manorama.kadwani@gmail.com>
#
# License: BSD-3-Clause

# %%

# sphinx_gallery_thumbnail_path = '_static/blender_import_obj/blender_import_obj2.jpg'  # noqa

import os
import shutil
import mne

data_path = mne.datasets.sample.data_path()
subjects_dir = data_path / 'subjects'
bem_dir = subjects_dir / 'sample' / 'bem' / 'flash'
surf_dir = subjects_dir / 'sample' / 'surf'

# %%
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
conv_dir = subjects_dir / 'sample' / 'conv'
os.makedirs(conv_dir, exist_ok=True)

# Load the inner skull surface and create a problem
# The metadata is empty in this example. In real study, we want to write the
# original metadata to the fixed surface file. Set read_metadata=True to do so.
coords, faces = mne.read_surface(bem_dir / 'inner_skull.surf')
coords[0] *= 1.1  # Move the first vertex outside the skull

# Write the inner skull surface as an .obj file that can be imported by
# Blender.
mne.write_surface(conv_dir / 'inner_skull.obj', coords, faces, overwrite=True)

# Also convert the outer skull surface.
coords, faces = mne.read_surface(bem_dir / 'outer_skull.surf')
mne.write_surface(conv_dir / 'outer_skull.obj', coords, faces, overwrite=True)

# %%
# Editing in Blender
# ^^^^^^^^^^^^^^^^^^
#
# See the following video tutorial for how to import, edit and export
# surfaces in Blender (step-by-step instructions are also below):
#
# ..  youtube:: JBIaX7VaTZk
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
# <https://www.youtube.com/watch?v=nIoXOplUvAw&list=PLjEaoINr3zgFX8ZsChQVQsuDSjEqdWMAD&index=1&t=0s>`_
# to learn how to use Blender. Specifically, `part 2
# <https://youtu.be/imdYIdv8F4w?t=712>`_ will teach you how to
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

coords, faces = mne.read_surface(conv_dir / 'inner_skull.obj')
coords[0] /= 1.1  # Move the first vertex back inside the skull
mne.write_surface(conv_dir / 'inner_skull_fixed.obj', coords, faces,
                  overwrite=True)

# %%
# Back in Python, you can read the fixed .obj files and save them as
# FreeSurfer .surf files. For the :func:`mne.make_bem_model` function to find
# them, they need to be saved using their original names in the ``surf``
# folder, e.g. ``bem/inner_skull.surf``. Be sure to first backup the original
# surfaces in case you make a mistake!

# Read the fixed surface
coords, faces = mne.read_surface(conv_dir / 'inner_skull_fixed.obj')

# Backup the original surface
shutil.copy(bem_dir / 'inner_skull.surf', bem_dir / 'inner_skull_orig.surf')

# Overwrite the original surface with the fixed version
# In real study you should provide the correct metadata using ``volume_info=``
# This could be accomplished for example with:
#
# _, _, vol_info = mne.read_surface(bem_dir / 'inner_skull.surf',
#                                   read_metadata=True)
# mne.write_surface(bem_dir / 'inner_skull.surf', coords, faces,
#                   volume_info=vol_info, overwrite=True)

# %%
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
coords, faces = mne.read_surface(bem_dir / 'outer_skin.surf')

# Make sure we are in the correct directory
head_dir = bem_dir.parent

# Remember to backup the original head file in advance!
# Overwrite the original head file
#
# mne.write_head_bem(head_dir / 'sample-head.fif', coords, faces,
#                    overwrite=True)

# %%
# High-resolution head
# ~~~~~~~~~~~~~~~~~~~~
#
# We use :func:`mne.read_bem_surfaces` to read the head surface files. After
# editing, we again output the head file with :func:`mne.write_head_bem`.
# Here we use ``-head.fif`` for speed.

# If ``-head-dense.fif`` does not exist, you need to run
# ``mne make_scalp_surfaces`` first.
# [0] because a list of surfaces is returned
surf = mne.read_bem_surfaces(head_dir / 'sample-head.fif')[0]

# For consistency only
coords = surf['rr']
faces = surf['tris']

# Write the head as an .obj file for editing
mne.write_surface(conv_dir / 'sample-head.obj',
                  coords, faces, overwrite=True)

# Usually here you would go and edit your meshes.
#
# Here we just use the same surface as if it were fixed
# Read in the .obj file
coords, faces = mne.read_surface(conv_dir / 'sample-head.obj')

# Remember to backup the original head file in advance!
# Overwrite the original head file
#
# mne.write_head_bem(head_dir / 'sample-head.fif', coords, faces,
#                    overwrite=True)

# %%
# .. note:: See also :ref:`tut-fix-meshes-smoothing` for a possible alternative
#           high-resolution head fix using FreeSurfer smoothing instead of
#           blender.
#
# Blender editing tips
# ~~~~~~~~~~~~~~~~~~~~
#
# A particularly useful operation is the *Shrinkwrap* functionality, that will
# restrict one surface inside another surface (for example the Skull inside the
# Outer Skin). Here is how to use it:
#
# ① Select surface
#     Select the surface that is creating the problem.
# ② Select vertices
#     In *Edit Mode*, press :kbd:`C` to use the circle selection tool to
#     select the vertices that are outside.
# ③-⑤ Group vertices
#     In the *Object Data Properties* tab use the ``+`` button
#     to add a *Vertex Group* and click *Assign* to assign the current
#     selection to the group.
# ⑥-⑧ Shrinkwrap
#     In the *Modifiers* tab go to *Add Modifier* add a
#     *Shrinkwrap* modifier and set it to snap *Inside* with the outer surface
#     as the *Target* and the *Group* that you created before as the
#     *Vertex Group*. You can then use the *Offset* parameter to adjust the
#     distance.
#
# .. note:: If nothing happens, the normals of the inner skull surface
#           may be inverted. To fix this select all the vertices or faces
#           of the inner skull and go to Mesh>Normals>Flip in the toolbar.
#
# ⑨ Apply
#     In *Object Mode* click on the down-pointing arrow of the *Shrinkwrap*
#     modifier and click on *Apply*.
#
# .. image:: ../../_static/blender_import_obj/blender_import_obj4.jpg
#    :alt: Shrinkwrap functionality in Blender
#
# That's it! You are ready to continue with your analysis pipeline (e.g.
# running :func:`mne.make_bem_model`).
#
# What if you still get an error?
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Blender operations
# ~~~~~~~~~~~~~~~~~~
# When editing BEM surfaces/meshes in Blender, make sure to use
# tools that do not change the number or order of vertices, or the geometry
# of triangular faces. For example, avoid the extrusion tool, because it
# duplicates the extruded vertices.
#
# .. _tut-fix-meshes-smoothing:
#
# Cleaning up a bad dense head surface by smoothing
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# If you get a rough head surfaces when using :ref:`mne make_scalp_surfaces`,
# consider smoothing your T1 ahead of time with a Gaussian kernel with
# FreeSurfer using something like the following within the subject's ``mri``
# directory:
#
# .. code-block:: console
#
#     $ mri_convert --fwhm 3 T1.mgz T1_smoothed_3.mgz
#
# Here the ``--fwhm`` argument determines how much smoothing (in mm) to apply.
# Then delete ``SUBJECTS_DIR/SUBJECT/surf/lh.seghead``, and re-run
# :ref:`mne make_scalp_surfaces` with the additional arguments
# ``--mri="T1_smoothed_3.mgz" --overwrite``, and you should get cleaner
# surfaces.
#
# Topological errors
# ~~~~~~~~~~~~~~~~~~
# MNE-Python requires that meshes satisfy some topological checks to ensure
# that subsequent processing like BEM solving and electrode projection can
# work properly.
#
# Below are some examples of errors you might encounter when working with
# meshes in MNE-Python, and the likely causes of those errors.
#
# 1. Cannot decimate to requested ico grade
#      This error is caused by having too few or too many vertices. The full
#      error is something like:
#
#      .. code-block:: text
#
#         RuntimeError: Cannot decimate to requested ico grade 4. The provided
#         BEM surface has 20516 triangles, which cannot be isomorphic with a
#         subdivided icosahedron. Consider manually decimating the surface to a
#         suitable density and then use ico=None in make_bem_model.
#
# 2. Surface inner skull has topological defects
#      This error can occur when trying to match the original number of
#      triangles by removing vertices. The full error looks like:
#
#      .. code-block:: text
#
#         RuntimeError: Surface inner skull has topological defects: 12 / 20484
#         vertices have fewer than three neighboring triangles [733, 1014,
#          2068, 7732, 8435, 8489, 10181, 11120, 11121, 11122, 11304, 11788]
#
# 3. Surface inner skull is not complete
#      This error (like the previous error) reflects a problem with the surface
#      topology (i.e., the expected pattern of vertices/edges/faces is
#      disrupted).
#
#      .. code-block:: text
#
#         RuntimeError: Surface inner skull is not complete (sum of solid
#         angles yielded 0.999668, should be 1.)
#
# 4. Triangle ordering is wrong
#      This error reflects a mismatch between how the surface is represented in
#      memory (the order of the vertex/face definitions) and what is expected
#      by MNE-Python. The full error is:
#
#      .. code-block:: text
#
#         RuntimeError: The source surface has a matching number of triangles
#         but ordering is wrong
#
# Dealing with topology in blender
# --------------------------------
# For any of these errors, it is usually easiest to start over with the
# unedited BEM surface and try again, making sure to only *move* vertices and
# faces without *adding* or *deleting* any. For example,
# select a circle of vertices, then press :kbd:`G` to drag them to the desired
# location. Smoothing a group of selected vertices in Blender (by
# right-clicking and selecting "Smooth Vertices") can also be helpful.
#
# .. _tut-fix-meshes-pymeshfix:
#
# Dealing with topology using pymeshfix
# -------------------------------------
# `pymeshfix <https://pymeshfix.pyvista.org/>`__ is a GPL-licensed Python
# module designed to produce water-tight meshes that satisfy the topological
# checks listed above. For example, if your
# ``'SUBJECTS_DIR/SUBJECT/surf/lh.seghead'`` has topological problems and
# :ref:`tut-fix-meshes-smoothing` does not work, you can try fixing the mesh
# instead. After installing ``pymeshfix`` using conda or pip, from within the
# ``'SUBJECTS_DIR/SUBJECT/surf'`` directory, you could try::
#
#     >>> import pymeshfix
#     >>> rr, tris = mne.read_surface('lh.seghead')
#     >>> rr, tris = pymeshfix.clean_from_arrays(rr, tris)
#     >>> mne.write_surface('lh.seghead', rr, tris)
#
# In some cases this could fix the topology such that a subsequent call to
# :ref:`mne make_scalp_surfaces` will succeed without needing to use the
# ``--force`` parameter.
