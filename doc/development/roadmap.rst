Roadmap
=======

This page describes some of the major medium- to long-term goals for
MNE-Python. These are goals that require substantial effort and/or
API design considerations. Some of these may be suitable for Google Summer of
Code projects, while others require more extensive work.

.. contents:: Page contents
   :local:

Open
----

Clustering statistics API
^^^^^^^^^^^^^^^^^^^^^^^^^
The current clustering statistics code has limited functionality. It should be
re-worked to create a new ``cluster_based_statistic`` or similar function.

The new API will likely be along the lines of::

   cluster_stat(obs, design, *, alpha=0.05, cluster_alpha=0.05, ...)

with:

``obs`` : :class:`pandas.DataFrame`
    Has columns like "subject", "condition", and "data".
    The "data" column holds things like :class:`mne.Evoked`,
    :class:`mne.SourceEstimate`, :class:`mne.Spectrum`, etc.
``design`` : `str`
    Likely Wilkisson notation to mirror :func:`patsy.dmatrices` (e.g., this is
    is used by :class:`statsmodels.regression.linear_model.OLS`). Getting from the
    string to the design matrix could be done via Patsy or more likely
    `Formulaic <https://matthewwardrop.github.io/formulaic/>`__.

This generic API will support mixed within- and between-subjects designs,
different statistical functions/tests, etc. This should be achievable without
introducing any significant speed penalty (e.g., < 10% slower) compared to the existing
more specialized/limited functions, since most computation cost is in clustering rather
than statistical testing.

Clear tutorials will be needed to:

1. Show how different contrasts can be done (toy data).
2. Show some common analyses on real data (time-freq, sensor space, source space, etc.)

Regression tests will be written to ensure equivalent outputs when compared to FieldTrip
for cases that FieldTrip also supports.

More details are in :gh:`4859`.

Modernization of realtime processing
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

LSL has become the de facto standard for streaming data from EEG/MEG systems.
We should deprecate `MNE-Realtime`_ in favor of the newly minted `MNE-LSL`_.
We should then fully support MNE-LSL using modern coding best practices such as CI
integration.

In progress
-----------

Diversity, Equity, and Inclusion (DEI)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
MNE-Python is committed to recruiting and retaining a diverse pool of
contributors, see :gh:`8221`.

First-class OPM support
^^^^^^^^^^^^^^^^^^^^^^^
MNE-Python has support for reading some OPM data formats such as FIF and FIL/QuSpin.
Support should be added for other manufacturers, and standard preprocessing routines
should be added to deal with coregistration adjustment and OPM-specific artifacts.
See for example :gh:`11275`, :gh:`11276`, :gh:`11579`, :gh:`12179`.

Deep source modeling
^^^^^^^^^^^^^^^^^^^^
Existing source modeling and inverse routines are not explicitly designed to
deal with deep sources. Advanced algorithms exist from MGH for enhancing
deep source localization, and these should be implemented and vetted in
MNE-Python. See :gh:`6784`.

Time-frequency classes
^^^^^^^^^^^^^^^^^^^^^^
Our current codebase implements classes related to :term:`TFRs <tfr>` that
remain incomplete. We should implement new classes from the ground up
that can hold frequency data (``Spectrum``), cross-spectral data
(``CrossSpectrum``), multitaper estimates (``MultitaperSpectrum``), and
time-varying estimates (``Spectrogram``). These should work for
continuous, epoched, and averaged sensor data, as well as source-space brain
data.

See related issues :gh:`6290`, :gh:`7671`, :gh:`8026`, :gh:`8724`, :gh:`9045`,
and PRs :gh:`6609`, :gh:`6629`, :gh:`6672`, :gh:`6673`, :gh:`8397`, and
:gh:`8892`.

3D visualization
^^^^^^^^^^^^^^^^
Historically we have used Mayavi for 3D visualization, but have faced
limitations and challenges with it. We should work to use some other backend
(e.g., PyVista) to get major improvements, such as:

1. *Proper notebook support (through ``ipyvtklink``)* (complete; updated to use ``trame``)
2. *Better interactivity with surface plots* (complete)
3. Time-frequency plotting (complementary to volume-based
   :ref:`time-frequency-viz`)
4. Integration of multiple functions as done in ``mne_analyze``, e.g.,
   simultaneous source estimate viewing, field map
   viewing, head surface display, etc. These are all currently available in
   separate functions, but we should be able to combine them in a single plot
   as well.

The meta-issue for tracking to-do lists for surface plotting is :gh:`7162`.

.. _documentation-updates:

Documentation updates
^^^^^^^^^^^^^^^^^^^^^
Our documentation has many minor issues, which can be found under the tag
:gh:`labels/DOC`.


Completed
---------

Improved sEEG/ECoG/DBS support
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
iEEG-specific pipeline steps such as electrode localization and visualizations
are now available in `MNE-gui-addons`_.

Access to open EEG/MEG databases
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Open EEG/MEG databases are now more easily accessible via standardized tools such as
`openneuro-py`_.

Eye-tracking support
^^^^^^^^^^^^^^^^^^^^
We had a GSoC student funded to improve support for eye-tracking data, see
`the GSoC proposal <https://summerofcode.withgoogle.com/programs/2023/projects/nUP0jGKi>`__
for details. An EyeLink data reader and analysis/plotting functions are now available.

Pediatric and clinical MEG pipelines
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
MNE-Python provides automated analysis of BIDS-compliant datasets via
`MNE-BIDS-Pipeline`_. Functionality from the
`mnefun <https://labsn.github.io/mnefun/overview.html>`__ pipeline,
which has been used extensively for pediatric data analysis at `I-LABS`_,
now provides better support for pediatric and clinical data processing.
Multiple processing steps (e.g., eSSS), sanity checks (e.g., cHPI quality),
and reporting (e.g., SSP joint plots, SNR plots) have been added.

Integrate OpenMEEG via improved Python bindings
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
`OpenMEEG`_ is a state-of-the art solver for
forward modeling in the field of brain imaging with MEG/EEG. It solves
numerically partial differential equations (PDE). It is written in C++ with
Python bindings written in SWIG.
The ambition of the project is to integrate OpenMEEG into MNE offering to MNE
the ability to solve more forward problems (cortical mapping, intracranial
recordings, etc.). Tasks that have been completed:

- Cleanup Python bindings (remove useless functions, check memory managements,
  etc.)
- Understand how MNE encodes info about sensors (location, orientation,
  integration points etc.) and allow OpenMEEG to be used.
- Modernize CI systems (e.g., using ``cibuildwheel``).
- Automated deployment on PyPI and conda-forge.

.. _time-frequency-viz:

Time-frequency visualization
^^^^^^^^^^^^^^^^^^^^^^^^^^^^
We implemented a viewer for interactive visualization of volumetric
source-time-frequency (5-D) maps on MRI slices (orthogonal 2D viewer).
`NutmegTrip <https://github.com/fieldtrip/fieldtrip/tree/master/contrib/nutmegtrip>`__
(written by Sarang Dalal) provides similar functionality in MATLAB in
conjunction with FieldTrip. Example of NutmegTrip's source-time-frequency mode
in action (click for link to YouTube):

.. image:: https://i.ytimg.com/vi/xKdjZZphdNc/maxresdefault.jpg
   :target: https://www.youtube.com/watch?v=xKdjZZphdNc
   :width: 50%

See :func:`mne-gui-addons:mne_gui_addons.view_vol_stc`.

Distributed computing support
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
`MNE-BIDS-Pipeline`_ has been enhanced with support for cloud computing
via `Dask`_ and :doc:`joblib <joblib:auto_examples/parallel/distributed_backend_simple>`.
After configuring Dask to use local or remote distributed computing resources,
MNE-BIDS-Pipeline can readily make use of remote workers to parallelize
processing across subjects.

2D visualization
^^^^^^^^^^^^^^^^
`This goal <https://mne.tools/0.22/overview/roadmap.html#2d-visualization>`__
was completed under CZI `EOSS2`_. Some additional enhancements that could also
be implemented are listed in :gh:`7751`.

Tutorial / example overhaul
^^^^^^^^^^^^^^^^^^^^^^^^^^^
`This goal <https://mne.tools/0.22/overview/roadmap.html#tutorial-example-overhaul>`__
was completed under CZI `EOSS2`_. Ongoing documentation needs are listed in
:ref:`documentation-updates`.

Cluster computing images
^^^^^^^^^^^^^^^^^^^^^^^^
As part of `this goal <https://mne.tools/0.22/overview/roadmap.html#cluster-computing>`__,
we created docker images suitable for cloud computing via `MNE-Docker`_.

.. _I-LABS: http://ilabs.washington.edu/
