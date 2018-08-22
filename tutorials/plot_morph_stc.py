# -*- coding: utf-8 -*-
r"""
================================================================
Morphing source estimates: Moving data from one brain to another
================================================================

Morphing refers to the operation of transferring
:ref:`source estimates <sphx_glr_auto_tutorials_plot_source_estimate_object.py`
from one anatomy to another. It is commonly referred as realignment in fMRI
literature. This operation is necessary for group studies as one needs
then data in a common space.

In this tutorial we will morph different kinds of source estimation results
between individual subject spaces using :class:`mne.SourceMorph` object.

We will use precomputed data and morph surface and volume source estimates to a
reference anatomy. The common space of choice will be FreeSurfer's 'fsaverage'
See :ref:`sphx_glr_auto_tutorials_plot_background_freesurfer.py` for more
information.

Furthermore we will convert our volume source estimate into a NIfTI image using
:meth:`morph.as_volume <mne.SourceMorph.as_volume>`.

.. contents::
    :local:

Why morphing?
=============

Modern neuroimaging techniques, such as source reconstruction or fMRI analyses,
make use of advanced mathematical models and hardware to map brain activity
patterns into a subject specific anatomical brain space.

This enables the study of spatio-temporal brain activity. The representation of
spatio-temporal brain data is often mapped onto the anatomical brain structure
to relate functional and anatomical maps. Thereby activity patterns are
overlaid with anatomical locations that supposedly produced the activity.
Anatomical MR images are often used as such or are transformed into an inflated
surface representations to serve as  "canvas" for the visualization.

In order to compute group level statistics, data representations across
subjects must be morphed to a common frame, such that anatomically and
functional similar structures are represented at the same spatial location for
*all subjects equally*.

Since brains vary, morphing comes into play to tell us how the data
produced by subject A, would be represented on the brain of subject B.

See also this :ref:`tutorial on surface source estimation
<sphx_glr_auto_tutorials_plot_mne_solutions.py>`
or this :ref:`example on volumetric source estimation
<sphx_glr_auto_examples_inverse_plot_compute_mne_inverse_volume.py>`.

Morphing **volume** source estimates
====================================

A volumetric source estimate represents functional data in a volumetric 3D
space. The difference between a volumetric representation and a "mesh" (
commonly referred to as "3D-model"), is that the volume is "filled" while the
mesh is "empty". Thus it is not only necessary to morph the points of the
outer hull, but also the "content" of the volume.

In MNE-Python, volumetric source estimates are represented as
:class:`mne.VolSourceEstimate`. The morph was successful if functional data of
Subject A overlaps with anatomical data of Subject B, in the same way it does
for Subject A.

Setting up :class:`mne.SourceMorph` for :class:`mne.VolSourceEstimate`
----------------------------------------------------------------------

Morphing volumetric data from subject A to subject B requires a non-linear
registration step between the anatomical T1 image of subject A to
the anatomical T1 image of subject B.

MNE-Python uses the Symmetric Diffeomorphic Registration from
dipy_ (See
`tutorial <http://nipy.org/dipy/examples_built/syn_registration_3d.html>`_
from dipy_ for reference and details).

:class:`mne.SourceMorph` uses segmented anatomical MR images computed
using :ref:`FreeSurfer <sphx_glr_auto_tutorials_plot_background_freesurfer.py>`
to compute the transformations. In order tell SourceMorph which MRIs to use,
``subject_from`` and ``subject_to`` need to be defined as the name of the
respective folder in FreeSurfer's home directory.

See :ref:`sphx_glr_auto_examples_inverse_plot_morph_volume_stc.py`
usage and for more details on:

    - How to create a SourceMorph object for volumetric data

    - Apply it to VolSourceEstimate

    - Get the output is NIfTI format

    - Save a SourceMorph object to disk

Here is an example result for morphing **volume** source estimates:

:ref:`sphx_glr_auto_examples_inverse_plot_morph_volume_stc.py`

.. image:: ../../_images/sphx_glr_plot_morph_volume_stc_001.png

Morphing **surface** source estimates
=====================================

A surface source estimate represents data relative to a 3-dimensional mesh of
the cortical surface computed using FreeSurfer. This mesh is defined by
its vertices. If we want to morph our data from one brain to another, then
this translates to finding the correct transformation to transform each
vertex from Subject A into a corresponding vertex of Subject B. Under the hood
:ref:`FreeSurfer <sphx_glr_auto_tutorials_plot_background_freesurfer.py>`
uses spherical representations to compute the morph.

In MNE-Python, surface source estimates are represented as
:class:`mne.SourceEstimate` or :class:`mne.VectorSourceEstimate`. Those can
be used together with :class:`mne.SourceSpaces` or without.

The morph was successful if functional data of Subject A overlaps with
anatomical surface data of Subject B, in the same way it does for Subject A.

See :ref:`sphx_glr_auto_examples_inverse_plot_morph_surface_stc.py`
usage and for more details:

    - How to create a SourceMorph object for surface data

    - Apply it to SourceEstimate or VectorSourceEstimate

    - Save a SourceMorph object to disk

Please see also Gramfort *et al.* (2013) [1]_.

Example result for morphing **surface** source estimates:

:ref:`sphx_glr_auto_examples_inverse_plot_morph_surface_stc.py`

.. image:: ../../_images/sphx_glr_plot_morph_surface_stc_001.png

References
==========
.. [1] Gramfort, A., Luessi, M., Larson, E., Engemann, D. A., Strohmeier, D.,
       Brodbeck, C., ... & Hämäläinen, M. (2013). MEG and EEG data analysis
       with MNE-Python. Frontiers in neuroscience, 7, 267.

.. _dipy: http://nipy.org/dipy/
"""
