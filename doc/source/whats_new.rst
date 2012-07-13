What's new
==========

Current
-----------

Changelog
~~~~~~~~~

   - speedup + reduce memory of mne.morph_data by `Alex Gramfort`_.

   - Backporting scipy.signa.firwin2 so filtering works with old scipy by `Alex Gramfort`_.

   - LCMV Beamformer by `Alex Gramfort`_.

   - Add support for reading named channel selections by `Martin Luessi`_.

   - Add Raw.filter method to more easily band pass data by `Alex Gramfort`_.

   - Add tmin + tmax parameters in mne.compute_covariance to estimate noise covariance in epochs baseline without creating new epochs by `Alex Gramfort`_.

   - Add support for sLORETA in apply_inverse, apply_inverse_raw, apply_inverse_epochs (API Change) by `Alex Gramfort`_.

   - Add method to regularize a noise covariance by `Alex Gramfort`_.

   - Read and write measurement info in forward and inverse operators for interactive visualization in mne_analyze by `Alex Gramfort`_.

   - New mne_compute_proj_ecg.py and mne_compute_proj_eog.py scripts to estimate ECG/EOG PCA/SSP vectors by `Alex Gramfort`_ and `Martin Luessi`_.

   - Wrapper function and script (mne_maxfilter.py) for Elekta Neuromag MaxFilter(TM) by `Martin Luessi`_
   
   - Add method to eliminate stimulation artifacts from raw data by linear interpolation or windowing by `Daniel Strohmeier`_.

.. _changes_0_3:

Version 0.3
-----------

Changelog
~~~~~~~~~

   - Sign flip computation for robust label average of signed values by `Alex Gramfort`_.

   - Reading and writing of .w files by `Martin Luessi`_.

   - Support for modifying Raw object and allow raw data preloading with memory mapping by `Martin Luessi`_ and `Alex Gramfort`_.

   - Support of arithmetic of Evoked data (useful to concatenate between runs and compute contrasts) by `Alex Gramfort`_.

   - Support for computing sensor space data from a source estimate using an MNE forward solution by `Martin Luessi`_.

   - Support of arithmetic of Covariance by `Alex Gramfort`_.

   - Write BEM surfaces in Python  by `Alex Gramfort`_.

   - Filtering operations and apply_function interface for Raw object by `Martin Luessi`_.

   - Support for complex valued raw fiff files and computation of analytic signal for Raw object by `Martin Luessi`_.

   - Write inverse operators (surface and volume) by `Alex Gramfort`_.

   - Covariance matrix computation with multiple event types by `Martin Luessi`_.

   - New tutorial in the documentation and new classes and functions reference page by `Alex Gramfort`_.

Authors
~~~~~~~~~

The committer list for this release is the following (preceded by number
of commits):

    * 80  Alexandre Gramfort
    * 51  Martin Luessi

Version 0.2
-----------

Changelog
~~~~~~~~~

   - New stats functions for FDR correction and Bonferroni by `Alex Gramfort`_.

   - Faster time-frequency using downsampling trick by `Alex Gramfort`_.

   - Support for volume source spaces by `Alex Gramfort`_ (requires next MNE release or nightly).

   - Improved Epochs handling by `Martin Luessi`_ (slicing, drop_bad_epochs).

   - Bug fix in Epochs + ECG detection by Manfred Kitzbichler.

   - New pick_types_evoked function by `Alex Gramfort`_.

   - SourceEstimate now supports algebra by `Alex Gramfort`_.

API changes summary
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Here are the code migration instructions when upgrading from mne-python
version 0.1:

  - New return values for the function find_ecg_events

Authors
~~~~~~~~~

The committer list for this release is the following (preceded by number
of commits):

    * 33  Alexandre Gramfort
    * 12  Martin Luessi
    *  2  Yaroslav Halchenko
    *  1  Manfred Kitzbichler

.. _Alex Gramfort: http://alexandre.gramfort.net

.. _Martin Luessi: http://ivpl.eecs.northwestern.edu/people/mluessi

.. _Yaroslav Halchenko: http://www.onerussian.com/

.. _Daniel Strohmeier: http://www.tu-ilmenau.de/bmti/fachgebiete/biomedizinische-technik/dipl-ing-daniel-strohmeier/