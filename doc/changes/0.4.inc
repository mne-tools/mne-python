.. _changes_0_4:

Version 0.4
-----------

Changelog
~~~~~~~~~

- Add function to compute source PSD using minimum norm by `Alex Gramfort`_

- L21 Mixed Norm Estimates (MxNE) by `Alex Gramfort`_ and `Daniel Strohmeier`_

- Generation of simulated evoked responses by `Alex Gramfort`_, `Daniel Strohmeier`_, and `Martin Luessi`_

- Fit AR models to raw data for temporal whitening by `Alex Gramfort`_.

- speedup + reduce memory of mne.morph_data by `Alex Gramfort`_.

- Backporting scipy.signal.firwin2 so filtering works with old scipy by `Alex Gramfort`_.

- LCMV Beamformer for evoked data, single trials, and raw data by `Alex Gramfort`_ and `Martin Luessi`_.

- Add support for reading named channel selections by `Martin Luessi`_.

- Add Raw.filter method to more easily band pass data by `Alex Gramfort`_.

- Add tmin + tmax parameters in mne.compute_covariance to estimate noise covariance in epochs baseline without creating new epochs by `Alex Gramfort`_.

- Add support for sLORETA in apply_inverse, apply_inverse_raw, apply_inverse_epochs (API Change) by `Alex Gramfort`_.

- Add method to regularize a noise covariance by `Alex Gramfort`_.

- Read and write measurement info in forward and inverse operators for interactive visualization in mne_analyze by `Alex Gramfort`_.

- New mne_compute_proj_ecg.py and mne_compute_proj_eog.py scripts to estimate ECG/EOG PCA/SSP vectors by `Alex Gramfort`_ and `Martin Luessi`_.

- Wrapper function and script (mne_maxfilter.py) for Elekta Neuromag MaxFilter(TM) by `Martin Luessi`_

- Add method to eliminate stimulation artifacts from raw data by linear interpolation or windowing by `Daniel Strohmeier`_.

Authors
~~~~~~~

The committer list for this release is the following (preceded by number
of commits):

* 118 Alexandre Gramfort
* 81  Martin Luessi
* 15  Daniel Strohmeier
*  4  Christian Brodbeck
*  4  Louis Thibault
*  2  Brad Buran
