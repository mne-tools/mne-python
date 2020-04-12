.. _changes_0_6:

Version 0.6
-----------

Changelog
~~~~~~~~~

- Linear (and zeroth-order) detrending for Epochs and Evoked by `Eric Larson`_

- Label morphing between subjects by `Eric Larson`_

- Define events based on time lag between reference and target event by `Denis Engemann`_

- ICA convenience function implementing an automated artifact removal workflow by `Denis Engemann`_

- Bad channels no longer included in epochs by default by `Eric Larson`_

- Support for diagonal noise covariances in inverse methods and rank computation by `Eric Larson`_

- Support for using CUDA in FFT-based FIR filtering (method='fft') and resampling by `Eric Larson`_

- Optimized FFT length selection for faster overlap-add filtering by `Martin Luessi`_

- Ability to exclude bad channels from evoked plots or shown them in red by `Martin Luessi`_

- Option to show both hemispheres when plotting SourceEstimate with PySurfer by `Martin Luessi`_

- Optimized Raw reading and epoching routines to limit memory copies by `Eric Larson`_

- Advanced options to save raw files in short or double precision by `Eric Larson`_

- Option to detect decreasing events using find_events by `Simon Kornblith`_

- Option to change default stim_channel used for finding events by `Eric Larson`_

- Use average patch normal from surface-oriented forward solution in inverse calculation when possible by `Eric Larson`_

- Function to plot drop_log from Epochs instance by `Eric Larson`_

- Estimate rank of Raw data by `Eric Larson`_

- Support reading of BTi/4D data by `Denis Engemann`_

- Wrapper for generating forward solutions by `Eric Larson`_

- Averaging forward solutions by `Eric Larson`_

- Events now contain the pre-event stim channel value in the middle column, by `Christian Brodbeck`_

- New function ``mne.find_stim_steps`` for finding all steps in a stim channel by `Christian Brodbeck`_

- Get information about FIFF files using mne.fiff.show_fiff() by `Eric Larson`_

- Compute forward fields sensitivity maps by `Alex Gramfort`_ and `Eric Larson`_

- Support reading of KIT data by `Teon Brooks`_ and `Christian Brodbeck`_

- Raw data visualization by `Eric Larson`_

- Smarter SourceEstimate object that contains linear inverse kernel and sensor space data for fast time-frequency transforms in source space by `Martin Luessi`_

- Add example of decoding/MVPA on MEG sensor data by `Alex Gramfort`_

- Add support for non-paired tests in spatiotemporal cluster stats by `Alex Gramfort`_

- Add unified SSP-projector API for Raw, Epochs and Evoked objects by `Denis Engemann`_, `Alex Gramfort`_ `Eric Larson`_ and `Martin Luessi`_

- Add support for delayed SSP application at evoked stage `Denis Engemann`_, `Alex Gramfort`_, `Eric Larson`_ and `Martin Luessi`_

- Support selective parameter updating in functions taking dicts as arguments by `Denis Engemann`_

- New ICA method ``sources_as_epochs`` to create Epochs in ICA space by `Denis Engemann`_

- New method in Evoked and Epoch classes to shift time scale by `Mainak Jas`_

- Added option to specify EOG channel(s) when computing PCA/SSP projections for EOG artifacts by `Mainak Jas`_

- Improved connectivity interface to allow combinations of signals, e.g., seed time series and source estimates, by `Martin Luessi`_

- Effective connectivity estimation using Phase Slope Index (PSI) by `Martin Luessi`_

- Support for threshold-free cluster enhancement (TFCE) by `Eric Larson`_

- Support for "hat" variance regularization by `Eric Larson`_

- Access source estimates as Pandas DataFrame by `Denis Engemann`_.

- Add example of decoding/MVPA on MEG source space data by `Denis Engemann`_

- Add support for --tstart option in mne_compute_proj_eog.py by `Alex Gramfort`_

- Add two-way repeated measures ANOVA for mass-univariate statistics by `Denis Engemann`_, `Eric Larson`_ and `Alex Gramfort`_

- Add function for summarizing clusters from spatio-temporal-cluster permutation tests by `Denis Engemann`_ and `Eric Larson`_

- Add generator support for ``lcmv_epochs`` by `Denis Engemann`_

- Gamma-MAP sparse source localization method by `Martin Luessi`_ and `Alex Gramfort`_

- Add regular expression and substring support for selecting parcellation labels by `Denis Engemann`_

- New plot_evoked option for interactive and reversible selection of SSP projection vectors by `Denis Engemann`_

- Plot 2D flat topographies with interpolation for evoked and SSPs by `Christian Brodbeck`_ and `Alex Gramfort`_

- Support delayed SSP applicationon for 2D flat topographies by `Denis Engemann`_ and `Christian Brodbeck`_ and `Alex Gramfort`_

- Allow picking maximum power source, a.k.a. "optimal", orientation in LCMV beamformers by `Roman Goj`_, `Alex Gramfort`_, `Denis Engemann`_ and `Martin Luessi`_

- Add sensor type scaling parameter to plot_topo by `Andrew Dykstra`_, `Denis Engemann`_  and `Eric Larson`_

- Support delayed SSP application in plot_topo by `Denis Engemann`_

API
~~~

- Deprecated use of fiff.pick_types without specifying exclude -- use either [] (none), ``bads`` (bad channels), or a list of string (channel names).

- Depth bias correction in dSPM/MNE/sLORETA make_inverse_operator is now done like in the C code using only gradiometers if present, else magnetometers, and EEG if no MEG channels are present.

- Fixed-orientation inverse solutions need to be made using `fixed=True` option (using non-surface-oriented forward solutions if no depth weighting is used) to maintain compatibility with MNE C code.

- Raw.save() will only overwrite the destination file, if it exists, if option overwrite=True is set.

- mne.utils.set_config(), get_config(), get_config_path() moved to mne namespace.

- Raw constructor argument proj_active deprecated -- use proj argument instead.

- Functions from the mne.mixed_norm module have been moved to the mne.inverse_sparse module.

- Deprecate CTF compensation (keep_comp and dest_comp) in Epochs and move it to Raw with a single compensation parameter.

- Remove artifacts module. Artifacts- and preprocessing related functions can now be found in mne.preprocessing.

Authors
~~~~~~~

The committer list for this release is the following (preceded by number
of commits):

* 340  Eric Larson
* 330  Denis A. Engemann
* 204  Alexandre Gramfort
*  72  Christian Brodbeck
*  66  Roman Goj
*  65  Martin Luessi
*  37  Teon Brooks
*  18  Mainak Jas
*   9  Simon Kornblith
*   7  Daniel Strohmeier
*   6  Romain Trachel
*   5  Yousra BEKHTI
*   5  Brad Buran
*   1  Andrew Dykstra
*   1  Christoph Dinh
