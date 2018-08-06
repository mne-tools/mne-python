# -*- coding: utf-8 -*-
r"""
===========================================
Morphing source estimates using SourceMorph
===========================================

In this tutorial we will morph different kinds of source estimation results
between individual subject spaces using :class:`mne.SourceMorph`.
For group level statistical analyses subject specific results have to be mapped
to a common space.

We will use precomputed data and morph surface and volume source estimates to a
common space. The common space of choice will be FreeSurfer's "fsaverage" See
:ref:`sphx_glr_auto_tutorials_plot_background_freesurfer.py` for more
information.

Furthermore we will convert our volume source estimate into a NIfTI image using
:meth:`morph.as_volume <mne.SourceMorph.as_volume>`.

.. contents::
    :local:

Why morphing?
=============

Modern neuroimaging techniques such as source reconstruction or fMRI analyses,
make use of advanced mathematical models and hardware to map brain activity
patterns into a subject specific anatomical brain space.

This enables the study of spatio-temporal brain activity. Amongst many others,
the representation of spatio-temporal brain data is often mapped onto the
anatomical brain structure to relate functional and anatomical maps. Thereby
activity patterns are overlaid with anatomical locations that supposedly
produced the activity. Anatomical MR images are often used as such or are
transformed into an inflated surface representations to serve as  "canvas" for
the functional projection. This projection must hence be in the same space.

In order to compute group level statistics, data representations across
subjects must be morphed to a common frame, such that anatomically and
functional similar structures are represented at the same spatial location for
*all subjects equally*.

Since brains vary, morphing comes into play, to tell us how the data
produced by subject A would be represented on the brain of subject B.

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
:class:`mne.VolSourceEstimate`. Only together with the corresponding
:class:`mne.SourceSpaces`, the morphing operation can be performed. The
reason for this is, that the source estimate carries only "functionally
relevant" data points (excluding e.g. the time series of voxel outside the
brain), whereas the source space, carries the information on how to
extrapolate this data into the subject specific anatomical space.

The morph was successful if functional data of Subject A overlaps with
anatomical data of Subject B, in the same way it does for Subject A.

See :ref:`sphx_glr_auto_examples_inverse_plot_morph_volume_stc.py` for an
example of such a morph.

Setting up SourceMorph for VolSourceEstimate
--------------------------------------------

See: also :ref:`sphx_glr_auto_examples_inverse_plot_morph_volume_stc.py`.

We know that the morph from Subject A to subject B employs a precomputation
of a morph volume. The respective morphing operation will be non-linear and
this is why a single transformation matrix would not be sufficient.

:class:`mne.SourceMorph` will use segmented anatomical MR images computed
using FreeSurfer to compute the morph map. In order tell SourceMorph which
MRIs to use, ``subject_from`` and ``subject_to`` need to be defined as the
name of the respective folder in the FreeSurfer's home directory.

``subject_from`` can also be inferred from src, subject_to set is to
'fsaverage' by default and subjects_dir can be None when set in the
environment. In that case SourceMorph can be initialized taking src as only
argument (for better understanding more keyword arguments are defined here).

The default parameter setting for *spacing* will cause the reference volumes
to be resliced before computing the transform. A value of '5' would cause
the function to reslice to an isotropic voxel size of 5 mm. The higher this
value the less accurate but faster the computation will be.

    >>> morph = SourceMorph(subject_from='sample',  # Default: None
    >>>                     subject_to='fsaverage',  # Default
    >>>                     subjects_dir=subjects_dir,  # Default: None
    >>>                     src=src,  # Default: None
    >>>                     spacing=5)  # Default

Apply morph to VolSourceEstimate
--------------------------------

The morph will be applied to the source estimate data, by giving it as the
first argument to the morph we computed above. Note that
:meth:`morph() <mne.SourceMorph.__call__>` can take the same input arguments
as :meth:`morph.as_volume() <mne.SourceMorph.as_volume>` to return a NIfTI
image instead of a MNE-Python representation of the source estimate.

    >>> stc_fsaverage = morph(stc)

Convert morphed VolSourceEstimate into NIfTI
--------------------------------------------

We can convert our morphed source estimate into a NIfTI volume using
:meth:`morph.as_volume() <mne.SourceMorph.as_volume>`. We provided our
morphed source estimate as first argument. All following keyword arguments can
be used to modify the output image.

Note that ``apply_morph=False``, that is the morph will not be applied because
the data has already been morphed. Set ``apply_morph=True`` to output
unmorphed data as a volume. Further :meth:`morph() <mne.SourceMorph.__call__>`
can be used to output a volume as well, taking the same input arguments.
Provide ``as_volume=True`` when calling the :class:`mne.SourceMorph` instance.
In that case however apply_morph will of course be True by default.

    >>> img = morph.as_volume(stc_fsaverage,  # morphed VolSourceEstimate
    >>>                       mri_resolution=True,  # Default: False
    >>>                       mri_space=True,  # Default: mri_resolution
    >>>                       apply_morph=False,  # Default
    >>>                       format='nifti1')  # Default

Example result for morphing **volume** source estimates
--------------------------------------------------------

:ref:`sphx_glr_auto_examples_inverse_plot_morph_volume_stc.py`

.. image:: ../../_images/sphx_glr_plot_morph_volume_stc_001.png

Morphing **surface** source estimates
=====================================

See: also :ref:`sphx_glr_auto_examples_inverse_plot_morph_surface_stc.py`.

A surface source estimate represents data relative to a 3-dimensional mesh of
the inflated brain surface computed using FreeSurfer. This mesh is defined by
its vertices. If we want to morph our data from one brain to another, then
this translates to finding the correct transformation to transform each
vertex from Subject A into a corresponding vertex of Subject B.

In MNE-Python, surface source estimates are represented as
:class:`mne.SourceEstimate` or :class:`mne.VectorSourceEstimate`. Those can
be used together with :class:`mne.SourceSpaces` or without.

The morph was successful if functional data of Subject A overlaps with
anatomical surface data of Subject B, in the same way it does for Subject A.

See :ref:`sphx_glr_auto_examples_inverse_plot_morph_surface_stc.py` for an
example of such a morph.

Setting up SourceMorph for SourceEstimate
-----------------------------------------

We know that surface source estimates are represented as lists of vertices. If
that is not entirely clear, we ask ourselves
:ref:`sphx_glr_auto_tutorials_plot_object_sourceestimate.py`

The respective list of our data can either be obtained from
:class:`mne.SourceSpaces` (src) or from the data we want to morph itself. If
src is not provided, the morph will not be precomputed but instead will be
prepared for morphing when calling.

This works only with (Vector) :class:`SourceEstimate <mne.SourceEstimate>`.
See :class:`mne.SourceMorph` for additional parameter settings. We keep the
default parameters for *src* and *spacing*.

Since the default of spacing (resolution of surface mesh) is 5 and subject_to
was set to 'fsaverage', SourceMorph will use default vertices to morph
(``[np.arange(10242)] * 2``).

If src was not defined, the morph will actually not be precomputed, because
we lack the vertices *from* that we want to compute. Instead the morph will
be set up and when applying it, the actual transformation will be computed on
the fly.

    >>> morph = SourceMorph(subject_from='sample',  # Default: None
    >>>                     subject_to='fsaverage',  # Default
    >>>                     subjects_dir=subjects_dir,  # Default: None
    >>>                     src=None,  # Default
    >>>                     spacing=5)  # Default

Apply morph to (Vector) SourceEstimate
--------------------------------------

The morph will be applied to the source estimate data, by giving it as the
first argument to the morph we computed above.

    >>> stc_fsaverage = morph(stc)

Example result for morphing **surface** source estimates
--------------------------------------------------------

:ref:`sphx_glr_auto_examples_inverse_plot_morph_surface_stc.py`

.. image:: ../../_images/sphx_glr_plot_morph_surface_stc_001.png

Reading and writing SourceMorph from and to disk
================================================

An instance of SourceMorph can be saved, by calling
:meth:`morph.save <mne.SourceMorph.save>`.

    >>> morph.save('my-file-name')

This methods allows for specification of a filename. The morph will be save
in ".h5" format. If no file extension is provided, "-morph.h5" will be
appended to the respective defined filename.

In turn, reading a saved source morph can be achieved by using
:func:`mne.read_source_morph`:

    >>> from mne import read_source_morph
    >>> morph = read_source_morph('my-file-name-morph.h5')

Shortcuts
=========

In addition to the functionality, demonstrated above, SourceMorph can be used
slightly different as well, in order to enhance user comfort.

For instance, it is possible to directly obtain a NIfTI image when calling
the SourceMorph instance, but setting 'as_volume=True'. If so, the __call__()
function takes the same input arguments as
:meth:`morph.as_volume <mne.SourceMorph.as_volume>`.

Moreover it can be decided whether to actually apply the morph or not by
setting the 'apply_morph' argument to True in order to morph the source
estimate and convert it into a volume in one go:

    >>> img = morph(stc, as_volume=True, apply_morph=True)

Once the environment is set up correctly, SourceMorph can be used without
creating an instance and assigning it to a variable. Instead the __init__
and __call__ methods of SourceMorph can be daisy chained into a handy
one-liner:

    >>> stc_fsaverage = mne.SourceMorph(src=src)(stc)
"""
