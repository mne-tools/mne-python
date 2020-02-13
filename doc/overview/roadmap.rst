.. _roadmap:

Roadmap
=======

This page describes some of the major medium- to long-term goals for
MNE-Python. These are goals that require substantial effort and/or
API design considerations. Some of these may be suitable for Google Summer of
Code projects, while others require more extensive work.

.. contents:: Page contents
   :local:
   :depth: 1


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


3D visualization
^^^^^^^^^^^^^^^^

Historically we have used Mayavi for 3D visualization, but have faced
limitations and challenges with it. We should work to use some other backend
(e.g., PyVista) to get major improvements, such as:

1. Proper notebook support (through vtkjs)
2. Better interactivity with surface plots
3. Time-frequency plotting (complementary to volume-based
   :ref:`time-frequency-viz`)
4. Integration of multiple functions as done in ``mne_analyze``, e.g.,
   simultaneous source estimate viewing, field map
   viewing, head surface display, etc. These are all currently available in
   separate functions, but we should be able to combine them in a single plot
   as well.

One such issue for tracking TODO lists for surface plotting is :gh:`7162`.


2D visualization
^^^^^^^^^^^^^^^^

Our 2D code has a lot of useful functionality, but suffers from several
systemic problems:

1. It was written by many people for many specific use cases over time,
   without ensuring that code was generalized. This means that some functions
   are available only in some plotting modes (e.g., grouping by channel types
   to obtain a butterfly plot can be done in :func:`mne.viz.plot_raw` but not
   :func:`mne.viz.plot_epochs`.
2. The code base has many redundant but not identical pieces of code
   (copy-paste-modify rather than generalize-reuse / DRY) that make maintenance
   particularly challenging.

By (extensively) refactoring our code, we would improve the end-user experience
and decrease long-term maintenance costs.


.. _time-frequency-viz:

Time-frequency visualization
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

We should implement a viewer for interactive visualization of volumetric
source-time-frequency (5-D) maps on MRI slices (orthogonal 2D viewer).
`NutmegTrip <https://github.com/fieldtrip/fieldtrip/tree/master/contrib/nutmegtrip>`__
(written by Sarang Dalal) provides similar functionality in Matlab in
conjunction with FieldTrip. Example of NutmegTrip's source-time-frequency mode
in action (click for link to YouTube):

.. image:: https://i.ytimg.com/vi/xKdjZZphdNc/maxresdefault.jpg
   :target: https://www.youtube.com/watch?v=xKdjZZphdNc
   :width: 50%


Cluster computing
^^^^^^^^^^^^^^^^^

Currently, cloud computing with M/EEG data requires multiple manual steps,
including remote environment setup, data transfer, monitoring of remote jobs,
and retrieval of output data/results. These steps are usually not specific to
the analysis of interest, and thus should be something that can be taken care
of by MNE. Subgoals consist of:

- Leverage dask and joblib or other libs to allow simple integration with MNE processing steps. Ideally this would be achieved in practice by:

  - One-time (or per-project) setup steps, setting up host keys, access tokens,
    etc.
  - In code, switch to cloud computing rather than local computing via a simple
    change of n_jobs parameter, and/or context manager like with::

        with use_dask(...):
           ...

- Develop a (short as possible) example that shows people how to run a minimal
  task remotely, including setting up access, cluster, nodes, etc.
- Adapt
  MNE-study-template_ code to use cloud computing (optionally, based on
  config) rather than local resources.


Tutorial / example overhaul
^^^^^^^^^^^^^^^^^^^^^^^^^^^

We want our tutorials to get users up to speed on:

1. How to do M/EEG analyses in principle, and
2. How to do M/EEG analyses in MNE-Python in particular

So far some of our tutorials have been rewritten, but we still have a long way
to go. Relevant tracking issues can be found under the tag :gh:`labels/DOC`.


Coregistration / 3D viewer
^^^^^^^^^^^^^^^^^^^^^^^^^^

:ref:`gen_mne_coreg` is an excellent tool for coregistration, but is limited
by being tied to Mayavi, Traits, and TraitsUI. We should first refactor in
several (mostly) separable steps:

1. Responsive code to use traitlets
2. GUI elements to use PyQt5 (rather than TraitsUI/pyface)
3. 3D plotting to use our abstracted 3D viz functions rather than Mayavi

Once this is done, we can effectively switch to a PyVista backend.


BIDS Integration
^^^^^^^^^^^^^^^^

MNE-Python should facilitate analyzing BIDS-compliant datasets thanks to
integration with the MNE-BIDS package. For more
information, see https://github.com/mne-tools/mne-bids.


Access to open EEG/MEG databases
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

We should improve the access to open EEG/MEG databases via the
:mod:`mne.datasets` module, in other words improve our dataset fetchers.
We have physionet, but much more. Having a consistent API to access multiple
data sources would be great. See :gh:`2852` and :gh:`3585` for some ideas,
as well as:

- `OpenNEURO <https://openneuro.org>`__
    "A free and open platform for sharing MRI, MEG, EEG, iEEG, and ECoG data."
- `Human Connectome Project Datasets <http://www.humanconnectome.org/data>`__
    Over a 3-year span (2012-2015), the Human Connectome Project (HCP) scanned
    1,200 healthy adult subjects. The available data includes MR structural
    scans, behavioral data and (on a subset of the data) resting state and/or
    task MEG data.
- `MMN dataset <http://www.fil.ion.ucl.ac.uk/spm/data/eeg_mmn>`__
    Used for tutorial/publications applying DCM for ERP analysis using SPM.
- `Kymata Datasets <https://kymata-atlas.org/datasets>`__.
    Current and archived EMEG measurement data, used to test hypotheses in the
    Kymata atlas. The participants are healthy human adults listening to the
    radio and/or watching films, and the data is comprised of (averaged) EEG
    and MEG sensor data and source current reconstructions.
- `BrainSignals <http://www.brainsignals.de>`__
    A website that lists a number of MEG datasets available for download.

.. LINKS
.. _MNE-study-template: https://github.com/mne-tools/mne-study-template
