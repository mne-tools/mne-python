# -*- coding: utf-8 -*-
r"""
================================================================
Morphing source estimates: Moving data from one brain to another
================================================================

Morphing refers to the operation of transferring
:ref:`source estimates <sphx_glr_auto_tutorials_plot_object_source_estimate.py>`
from one anatomy to another. It is commonly referred as realignment in fMRI
literature. This operation is necessary for group studies as one needs
then data in a common space.

In this tutorial we will morph different kinds of source estimation results
between individual subject spaces using :class:`mne.SourceMorph` object.

We will use precomputed data and morph surface and volume source estimates to a
reference anatomy. The common space of choice will be FreeSurfer's 'fsaverage'
See :ref:`sphx_glr_auto_tutorials_plot_background_freesurfer.py` for more
information. Method used for cortical surface data in based
on spherical registration [1]_ and Symmetric Diffeomorphic Registration (SDR)
for volumic data [2]_.

Furthermore we will convert our volume source estimate into a NIfTI image using
:meth:`morph.apply(..., output='nifti1') <mne.SourceMorph.apply>`.

In order to morph :class:`labels <mne.Label>` between subjects allowing the
definition of labels in a one brain and transforming them to anatomically
analogous labels in another use :func:`mne.Label.morph`.

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

MNE-Python uses the Symmetric Diffeomorphic Registration [2]_ as implemented
in dipy_ [3]_ (See
`tutorial <http://nipy.org/dipy/examples_built/syn_registration_3d.html>`_
from dipy_ for more details).

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

Morphing **surface** source estimates
=====================================

A surface source estimate represents data relative to a 3-dimensional mesh of
the cortical surface computed using FreeSurfer. This mesh is defined by
its vertices. If we want to morph our data from one brain to another, then
this translates to finding the correct transformation to transform each
vertex from Subject A into a corresponding vertex of Subject B. Under the hood
:ref:`FreeSurfer <sphx_glr_auto_tutorials_plot_background_freesurfer.py>`
uses spherical representations to compute the morph, as relies on so
called *morphing maps*.

The morphing maps
-----------------

The MNE software accomplishes morphing with help of morphing
maps which can be either computed on demand or precomputed.
The morphing is performed with help
of the registered spherical surfaces (``lh.sphere.reg`` and ``rh.sphere.reg`` )
which must be produced in FreeSurfer.
A morphing map is a linear mapping from cortical surface values
in subject A (:math:`x^{(A)}`) to those in another
subject B (:math:`x^{(B)}`)

.. math::    x^{(B)} = M^{(AB)} x^{(A)}\ ,

where :math:`M^{(AB)}` is a sparse matrix
with at most three nonzero elements on each row. These elements
are determined as follows. First, using the aligned spherical surfaces,
for each vertex :math:`x_j^{(B)}`, find the triangle :math:`T_j^{(A)}` on the
spherical surface of subject A which contains the location :math:`x_j^{(B)}`.
Next, find the numbers of the vertices of this triangle and set
the corresponding elements on the *j* th row of :math:`M^{(AB)}` so that
:math:`x_j^{(B)}` will be a linear interpolation between the triangle vertex
values reflecting the location :math:`x_j^{(B)}` within the triangle
:math:`T_j^{(A)}`.

It follows from the above definition that in general

.. math::    M^{(AB)} \neq (M^{(BA)})^{-1}\ ,

*i.e.*,

.. math::    x_{(A)} \neq M^{(BA)} M^{(AB)} x^{(A)}\ ,

even if

.. math::    x^{(A)} \approx M^{(BA)} M^{(AB)} x^{(A)}\ ,

*i.e.*, the mapping is *almost* a bijection.

Morphing maps can be computed on the fly or read with
:func:`mne.read_morph_map`. Precomputed maps are
located in ``$SUBJECTS_DIR/morph-maps``.

The names of the files in ``$SUBJECTS_DIR/morph-maps`` are
of the form:

 <*A*> - <*B*> -``morph.fif`` ,

where <*A*> and <*B*> are names of subjects. These files contain the maps
for both hemispheres, and in both directions, *i.e.*, both :math:`M^{(AB)}`
and :math:`M^{(BA)}`, as defined above. Thus the files
<*A*> - <*B*> -``morph.fif`` or <*B*> - <*A*> -``morph.fif`` are
functionally equivalent. The name of the file produced depends on the role
of <*A*> and <*B*> in the analysis.

About smoothing
---------------

The current estimates are normally defined only in a decimated
grid which is a sparse subset of the vertices in the triangular
tessellation of the cortical surface. Therefore, any sparse set
of values is distributed to neighboring vertices to make the visualized
results easily understandable. This procedure has been traditionally
called smoothing but a more appropriate name
might be smudging or blurring in
accordance with similar operations in image processing programs.

In MNE software terms, smoothing of the vertex data is an
iterative procedure, which produces a blurred image :math:`x^{(N)}` from
the original sparse image :math:`x^{(0)}` by applying
in each iteration step a sparse blurring matrix:

.. math::    x^{(p)} = S^{(p)} x^{(p - 1)}\ .

On each row :math:`j` of the matrix :math:`S^{(p)}` there
are :math:`N_j^{(p - 1)}` nonzero entries whose values
equal :math:`1/N_j^{(p - 1)}`. Here :math:`N_j^{(p - 1)}` is
the number of immediate neighbors of vertex :math:`j` which
had non-zero values at iteration step :math:`p - 1`.
Matrix :math:`S^{(p)}` thus assigns the average
of the non-zero neighbors as the new value for vertex :math:`j`.
One important feature of this procedure is that it tends to preserve
the amplitudes while blurring the surface image.

Once the indices non-zero vertices in :math:`x^{(0)}` and
the topology of the triangulation are fixed the matrices :math:`S^{(p)}` are
fixed and independent of the data. Therefore, it would be in principle
possible to construct a composite blurring matrix

.. math::    S^{(N)} = \prod_{p = 1}^N {S^{(p)}}\ .

However, it turns out to be computationally more effective
to do blurring with an iteration. The above formula for :math:`S^{(N)}` also
shows that the smudging (smoothing) operation is linear.

From theory to practice
-----------------------

In MNE-Python, surface source estimates are represented as
:class:`mne.SourceEstimate` or :class:`mne.VectorSourceEstimate`. Those can
be used together with :class:`mne.SourceSpaces` or without.

The morph was successful if functional data of Subject A overlaps with
anatomical surface data of Subject B, in the same way it does for Subject A.

See :ref:`sphx_glr_auto_examples_inverse_plot_morph_surface_stc.py`
usage and for more details:

    - How to create a :class:`mne.SourceMorph` object using
      :func:`mne.compute_source_morph` for surface data

    - Apply it to :class:`mne.SourceEstimate` or
      :class:`mne.VectorSourceEstimate`

    - Save a :class:`mne.SourceMorph` object to disk

Please see also Gramfort *et al.* (2013) [4]_.

References
==========
.. [1] Greve D. N., Van der Haegen L., Cai Q., Stufflebeam S., Sabuncu M.
       R., Fischl B., Brysbaert M.
       A Surface-based Analysis of Language Lateralization and Cortical
       Asymmetry. Journal of Cognitive Neuroscience 25(9), 1477-1492, 2013.
.. [2] Avants, B. B., Epstein, C. L., Grossman, M., & Gee, J. C. (2009).
       Symmetric Diffeomorphic Image Registration with Cross- Correlation:
       Evaluating Automated Labeling of Elderly and Neurodegenerative
       Brain, 12(1), 26-41.
.. [3] Garyfallidis E, Brett M, Amirbekian B, Rokem A, van der Walt S,
       Descoteaux M, Nimmo-Smith I and Dipy Contributors (2014). DIPY, a
       library for the analysis of diffusion MRI data. Frontiers in
       Neuroinformatics, vol.8, no.8.
.. [4] Gramfort A., Luessi M., Larson E., Engemann D. A., Strohmeier D.,
       Brodbeck C., Goj R., Jas. M., Brooks T., Parkkonen L. & Hämäläinen, M.
       (2013). MEG and EEG data analysis with MNE-Python. Frontiers in
       neuroscience, 7, 267.

.. _dipy: http://nipy.org/dipy/
"""  # noqa: E501
