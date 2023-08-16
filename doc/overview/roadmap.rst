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
In particular, the new API should:

1. Support mixed within- and between-subjects designs, different statistical
   functions, etc. This should be done via a ``design`` argument that mirrors
   :func:`patsy.dmatrices` or similar community standard (e.g., this is what
   is used by :class:`statsmodels.regression.linear_model.OLS`).
2. Have clear tutorials showing how different contrasts can be done (toy data).
3. Have clear tutorials showing some common analyses on real data (time-freq,
   sensor space, source space, etc.)
4. Not introduce any significant speed penalty (e.g., < 10% slower) compared
   to the existing, more specialized/limited functions.

More details are in :gh:`4859`.

Access to open EEG/MEG databases
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
We should improve the access to open EEG/MEG databases via the
:mod:`mne.datasets` module, in other words improve our dataset fetchers.
We have physionet, but much more. Having a consistent API to access multiple
data sources would be great. See :gh:`2852` and :gh:`3585` for some ideas,
as well as:

- `OpenNEURO <https://openneuro.org>`__
    "A free and open platform for sharing MRI, MEG, EEG, iEEG, and ECoG data."
    See for example :gh:`6687`.
- `Human Connectome Project Datasets <http://www.humanconnectome.org/data>`__
    Over a 3-year span (2012-2015), the Human Connectome Project (HCP) scanned
    1,200 healthy adult subjects. The available data includes MR structural
    scans, behavioral data and (on a subset of the data) resting state and/or
    task MEG data.
- `MMN dataset <http://www.fil.ion.ucl.ac.uk/spm/data/eeg_mmn>`__
    Used for tutorial/publications applying DCM for ERP analysis using SPM.
- Kymata datasets
    Current and archived EMEG measurement data, used to test hypotheses in the
    Kymata atlas. The participants are healthy human adults listening to the
    radio and/or watching films, and the data is comprised of (averaged) EEG
    and MEG sensor data and source current reconstructions.
- `BNCI Horizon <https://bnci-horizon-2020.eu/database/data-sets>`__
    BCI datasets.

In progress
-----------

Eye-tracking support
^^^^^^^^^^^^^^^^^^^^
We had a GSoC student funded to improve support for eye-tracking data, see
`the GSoC proposal <https://summerofcode.withgoogle.com/programs/2023/projects/nUP0jGKi>`__
for details.

Diversity, Equity, and Inclusion (DEI)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
MNE-Python is committed to recruiting and retaining a diverse pool of
contributors, see :gh:`8221`.

First-class OPM support
^^^^^^^^^^^^^^^^^^^^^^^
MNE-Python has support for reading some OPM data formats such as FIF, but
support is still rudimentary. Support should be added for other manufacturers,
and standard (and/or novel) preprocessing routines should be added to deal with
coregistration adjustment, forward modeling, and OPM-specific artifacts.

Deep source modeling
^^^^^^^^^^^^^^^^^^^^
Existing source modeling and inverse routines are not explicitly designed to
deal with deep sources. Advanced algorithms exist from MGH for enhancing
deep source localization, and these should be implemented and vetted in
MNE-Python.

Better sEEG/ECoG/DBS support
^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Some support already exists for iEEG electrodes in MNE-Python thanks in part
to standard abstractions. However, iEEG-specific pipeline steps (e.g.,
electrode localization) and visualizations (e.g., per-shaft topo plots,
:ref:`time-frequency-viz`) are missing. MNE-Python should work with members of
the ECoG/sEEG community to work with or build in existing tools, and extend
native functionality for depth electrodes.

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

Pediatric and clinical MEG pipelines
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
MNE-Python is in the process of providing automated analysis of BIDS-compliant
datasets, see `MNE-BIDS-Pipeline`_. By incorporating functionality from the
`mnefun <https://labsn.github.io/mnefun/overview.html>`__ pipeline,
which has been used extensively for pediatric data analysis at `I-LABS`_,
better support for pediatric and clinical data processing can be achieved.
Multiple processing steps (e.g., eSSS), sanity checks (e.g., cHPI quality),
and reporting (e.g., SSP joint plots, SNR plots) will be implemented.

Statistics efficiency
^^^^^^^^^^^^^^^^^^^^^
A key technique in functional neuroimaging analysis is clustering brain
activity in adjacent regions prior to statistical analysis. An important
clustering algorithm — threshold-free cluster enhancement (TFCE) — currently
relies on computationally expensive permutations for hypothesis testing.
A faster, probabilistic version of TFCE (pTFCE) is available, and we are in the
process of implementing this new algorithm.

3D visualization
^^^^^^^^^^^^^^^^
Historically we have used Mayavi for 3D visualization, but have faced
limitations and challenges with it. We should work to use some other backend
(e.g., PyVista) to get major improvements, such as:

1. *Proper notebook support (through ipyvtklink)* (complete)
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

Integrate OpenMEEG via improved Python bindings
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
`OpenMEEG <http://openmeeg.github.io>`__ is a state-of-the art solver for
forward modeling in the field of brain imaging with MEG/EEG. It solves
numerically partial differential equations (PDE). It is written in C++ with
Python bindings written in `SWIG <https://github.com/openmeeg/openmeeg>`__.
The ambition of the project is to integrate OpenMEEG into MNE offering to MNE
the ability to solve more forward problems (cortical mapping, intracranial
recordings, etc.). Tasks that have been completed:

- Cleanup Python bindings (remove useless functions, check memory managements,
  etc.)
- Understand how MNE encodes info about sensors (location, orientation,
  integration points etc.) and allow OpenMEEG to be used.
- Modernize CI systems (e.g., using ``cibuildwheel``).

See `OpenMEEG`_ for details.

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
