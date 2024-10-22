Roadmap
=======

This page describes some of the major medium- to long-term goals for
MNE-Python. These are goals that require substantial effort and/or
API design considerations. Some of these may be suitable for Google Summer of
Code projects, while others require more extensive work.


Open
----

Type Annotations
^^^^^^^^^^^^^^^^

We would like to have type annotations for as much of our codebase as is practicable.
The main motivation for this is to improve the end-user experience when writing analysis
code that uses MNE-Python (i.e., code-completion suggestions, which rely on static
analysis / type hints). The main discussion of how to go about this is in :gh:`12243`.
Some piecemeal progress has been made (e.g., :gh:`12250`) but there isn't currently
anyone actively chipping away at this, hence its status as "open" rather than "in
progress".

Docstring De-duplication
^^^^^^^^^^^^^^^^^^^^^^^^

For many years, MNE-Python has used a technique borrowed from SciPy (called
`doccer <https://github.com/scipy/scipy/blob/f054bfe5514f35dd47f06b0c2f762b7a857a63b0/scipy/_lib/doccer.py>`__)
for improving the consistency of parameter names and descriptions that recur across our
API. For example, parameters for number of parallel jobs to use, for specifying random
seeds, or for controlling the appearance of a colorbar on a plot --- all of these appear
in multiple functions/methods in MNE-Python. The approach works by re-defining a
function's ``__doc__`` attribute at import time, filling in placeholders in the
docstring's parameter list with fully spelled-out equivalents (which are stored in a big
dictionary called the ``docdict``). There are two major downsides:

1. Many docstrings can't be read (at least not in full) while browsing the source code.
2. Static code analyzers don't have access to the completed docstrings, so things like
   hover-tooltips in IDEs are less useful than they would be if the docstrings were
   complete in-place.

A possible route forward:

- Convert all docstrings to be fully spelled out in the source code.
- Instead of maintaining the ``docdict``, maintain a registry of sets of
  function+parameter combinations that ought to be identical.
- Add a test that the entries in the registry are indeed identical, so that
  inconsistencies cannot be introduced in existing code.
- Add a test that parses docstrings in any *newly added* functions and looks for
  parameter names that maybe should be added to the registry of identical docstrings.
- To allow for parameter descriptions that should be *nearly* identical (e.g., the same
  except one refers to :class:`~mne.io.Raw` objects and the other refers to
  :class:`~mne.Epochs` objects), consider using regular expressions to check the
  "identity" of the parameter descriptions.

The main discussion is in :gh:`8218`; a wider discussion among maintainers of other
packages in the Scientific Python Ecosystem is
`here <https://github.com/scientific-python/summit-2024/issues/27>`__.

Containerization
^^^^^^^^^^^^^^^^

Users sometimes encounter difficulty getting a working MNE-Python environment on shared
resources (such as compute clusters), due to various problems (old versions of package
managers or graphics libraries, lack of sufficient permissions, etc). Providing a
robust and up-to-date containerized distribution of MNE-Python would alleviate some of
these issues. Initial efforts can be seen in the
`MNE-Docker repository <https://github.com/mne-tools/mne-docker>`__; these efforts
should be revived, brought up-to-date as necessary, and integrated into our normal
release process so that the images do not become stale.

Education
^^^^^^^^^

Live workshops/tutorials/trainings on MNE-Python have historically been organized
*ad-hoc* rather than centrally. Instructors for these workshops are often approached
directly by the organization or group desiring to host the training, and there is often
no way for users outside that group to attend (or even learn about the opportunity). At
a minimum, we would like to have a process for keeping track of educational events that
feature MNE-Python or other tools in the MNE suite. Ideally, we would go further and
initiate a recurring series of tutorials that could be advertised widely. Such events
might even provide a small revenue stream for MNE-Python, to support things like
continuous integration costs.


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
Historically our codebase had classes related to :term:`TFRs <tfr>` that
were incomplete. New classes are being built from the ground up:

- new classes :class:`~mne.time_frequency.Spectrum` and
  :class:`~mne.time_frequency.EpochsSpectrum` (implemented in :gh:`10184`, with
  follow-up tweaks and bugfixes in :gh:`11178`, :gh:`11259`, :gh:`11280`, :gh:`11345`,
  :gh:`11418`, :gh:`11563`, :gh:`11680`, :gh:`11682`, :gh:`11778`, :gh:`11921`,
  :gh:`11978`, :gh:`12747`), and corresponding array-based constructors
  :class:`~mne.time_frequency.SpectrumArray` and
  :class:`~mne.time_frequency.EpochsSpectrumArray` (:gh:`11803`).

- new class :class:`~mne.time_frequency.RawTFR` and updated classes
  :class:`~mne.time_frequency.EpochsTFR` and :class:`~mne.time_frequency.AverageTFR`,
  and corresponding array-based constructors :class:`~mne.time_frequency.RawTFRArray`,
  :class:`~mne.time_frequency.EpochsTFRArray` and
  :class:`~mne.time_frequency.AverageTFRArray` (implemented in :gh:`11282`, with
  follow-ups in :gh:`12514`, :gh:`12842`).

- new/updated classes for source-space frequency and time-frequency data are not yet
  implemented.

Other related issues: :gh:`6290`, :gh:`7671`, :gh:`8026`, :gh:`8724`, :gh:`9045`,
and PRs: :gh:`6609`, :gh:`6629`, :gh:`6672`, :gh:`6673`, :gh:`8397`, :gh:`8892`.

Modernization of realtime processing
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

LSL has become the de facto standard for streaming data from EEG/MEG systems.
We should deprecate `MNE-Realtime`_ in favor of the newly minted `MNE-LSL`_.
We should then fully support MNE-LSL using modern coding best practices such as CI
integration.

Core components of commonly used real-time processing pipelines should be implemented in
MNE-LSL, including but not limited to realtime IIR filtering, artifact rejection,
montage and reference setting, and online averaging. Integration with standard
MNE-Python plotting routines (evoked joint plots, topomaps, etc.) should be
supported with continuous updating.

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
    :class:`mne.SourceEstimate`, :class:`mne.time_frequency.Spectrum`, etc.
``design`` : `str`
    Likely Wilkinson notation to mirror :func:`patsy.dmatrices` (e.g., this is
    is used by :class:`statsmodels.regression.linear_model.OLS`). Getting from the
    string to the design matrix could be done via Patsy or more likely
    `Formulaic <https://matthewwardrop.github.io/formulaic/>`__.

This generic API will support mixed within- and between-subjects designs,
different statistical functions/tests, etc. This should be achievable without
introducing any significant speed penalty (e.g., < 10% slower) compared to the existing
more specialized/limited functions, since most computation cost is in clustering rather
than statistical testing.

The clustering function will return a user-friendly ``ClusterStat`` object or similar
that retains information about dimensionality, significance, etc. and facilitates
plotting and interpretation of results.

Clear tutorials will be needed to:

1. Show how different contrasts can be done (toy data).
2. Show some common analyses on real data (time-freq, sensor space, source space, etc.)

Regression tests will be written to ensure equivalent outputs when compared to FieldTrip
for cases that FieldTrip also supports.

More details are in :gh:`4859`; progress in :gh:`12663`.


.. _documentation-updates:

Documentation updates
^^^^^^^^^^^^^^^^^^^^^
Our documentation has many minor issues, which can be found under the tag
:gh:`labels/DOC`.


Completed
---------

3D visualization
^^^^^^^^^^^^^^^^
Historically we used Mayavi for 3D visualization, but faced limitations and challenges
with it. We switched to PyVista to get major improvements, such as:

1. *Proper notebook support (through ``ipyvtklink``)* (complete; updated to use ``trame``)
2. *Better interactivity with surface plots* (complete)
3. Time-frequency plotting (complementary to volume-based
   :ref:`time-frequency-viz`)
4. Integration of multiple functions as done in ``mne_analyze``, e.g.,
   simultaneous source estimate viewing, field map
   viewing, head surface display, etc. These were all available in
   separate functions, but can now be combined in a single plot.

The meta-issue tracking to-do lists for surface plotting was :gh:`7162`.

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
