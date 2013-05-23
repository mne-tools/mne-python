What's new
==========

.. _changes_0_6:

Current
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

   - New function `mne.find_stim_steps` for finding all steps in a stim channel by `Christian Brodbeck`_

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

   - New ICA method `sources_as_epochs` to create Epochs in ICA space by `Denis Engemann`_

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

   - Add generator support for lcmv_epochs by `Denis Engemann`_

   - Gamma-MAP sparse source localization method by `Martin Luessi`_ and `Alex Gramfort`_

API
~~~

   - Deprecated use of fiff.pick_types without specifying exclude -- use either [] (none), 'bads' (bad channels), or a list of string (channel names).

   - Depth bias correction in dSPM/MNE/sLORETA make_inverse_operator is now done like in the C code using only gradiometers if present, else magnetometers, and EEG if no MEG channels are present.

   - Fixed-orientation inverse solutions need to be made using 'fixed=True' option (using non-surface-oriented forward solutions if no depth weighting is used) to maintain compatibility with MNE C code.

   - Raw.save() will only overwrite the destination file, if it exists, if option overwrite=True is set.

   - mne.utils.set_config(), get_config(), get_config_path() moved to mne namespace.

   - Raw constructor argument proj_active deprecated -- use proj argument instead.

   - Functions from the mne.mixed_norm module have been moved to the mne.inverse_sparse module.

.. _changes_0_5:

Version 0.5
-----------

Changelog
~~~~~~~~~

   - Multi-taper PSD estimation for single epochs in source space using minimum norm by `Martin Luessi`_

   - Read and visualize .dip files obtained with xfit or mne_dipole_fit by `Alex Gramfort`_

   - Make EEG layout by `Eric Larson`_

   - Ability to specify SSP projectors when computing covariance from raw by `Eric Larson`_

   - Read and write txt based event files (.eve or .txt) by `Eric Larson`_

   - Pass qrs threshold to preprocessing functions by `Eric Larson`_

   - Compute SSP projections from continuous raw data by `Eric Larson`_

   - Support for applied SSP projections when loading Raw by `Eric Larson`_ and `Alex Gramfort`_

   - Support for loading Raw stored in different fif files by `Eric Larson`_

   - IO of many Evoked in a single fif file + compute Epochs.standard_error by `Eric Larson`_ and `Alex Gramfort`_

   - ICA computation on Raw and Epochs with automatic component selection by `Denis Engemann`_ and `Alex Gramfort`_

   - Saving ICA sources to fif files and creating ICA topography layouts by
     `Denis Engemann`_

   - Save and restore ICA session to and from fif by `Denis Engemann`_

   - Export raw, epochs and evoked data as data frame to the pandas library by `Denis Engemann`_

   - Export raw, epochs and evoked data to the nitime library by `Denis Engemann`_

   - Copy methods for raw and epochs objects by `Denis Engemann`_, `Martin Luessi`_ and `Alex Gramfort`_

   - New raw objects method to get the time at certain indices by `Denis Engemann`_ and `Alex Gramfort`_

   - Plot method for evoked objects by `Denis Engemann`_

   - Enhancement of cluster-level stats (speed and memory efficiency) by `Eric Larson`_ and `Martin Luessi`_

   - Reading of source space distances by `Eric Larson`_

   - Support for filling / smoothing labels and speedup of morphing by `Eric Larson`_

   - Adding options for morphing by `Eric Larson`_

   - Plotting functions for time frequency and epochs image topographies by `Denis Engemann`_ and `Alex Gramfort`_

   - Plotting ERP/ERF images by `Alex Gramfort`_

   - See detailed subplot when cliking on a channel inside a topography plot by `Martin Luessi`_, `Eric Larson`_ and `Denis Engemann`_

   - Misc channel type support plotting functions by `Denis Engemann`_

   - Improved logging support by `Eric Larson`_

   - Whitening of evoked data for plotting and quality checking by `Alex Gramfort`_

   - Transparent I/O of gzipped fif files (as .fif.gz) by `Eric Larson`_

   - Spectral connectivity estimation in sensor and source space by `Martin Luessi`_

   - Read and write Epochs in FIF files by `Alex Gramfort`_

   - Resampling of Raw, Epochs, and Evoked by `Eric Larson`_

   - Creating epochs objects for different conditions and accessing conditions via user-defined name by `Denis Engemann`_ , `Eric Larson`_, `Alex Gramfort`_ and `Christian Brodbeck`_

   - Visualizing evoked responses from different conditions in one topography plot by `Denis Engemann`_ and `Alex Gramfort`_

   - Support for L21 MxNE solver using coordinate descent using scikit-learn by `Alex Gramfort`_ and `Daniel Strohmeier`_

   - Support IIR filters (butterworth, chebyshev, bessel, etc.) by `Eric Larson`_

   - Read labels from FreeSurfer parcellation by  `Martin Luessi`_

   - Combining labels in source space by `Christian Brodbeck`_

   - Read and write source spaces, surfaces and coordinate transforms to and from files by `Christian Brodbeck`_

   - Downsample epochs by `Christian Brodbeck`_ and `Eric Larson`_

   - New labels class for handling source estimates by `Christian Brodbeck`_, `Martin Luessi`_  and `Alex Gramfort`_

   - New plotting routines to easily display SourceEstimates using PySurfer by `Alex Gramfort`_

   - Function to extract label time courses from SourceEstimate(s) by `Martin Luessi`_

   - Function to visualize connectivity as circular graph by `Martin Luessi`_ and `Alex Gramfort`_

   - Time-frequency Mixed Norm Estimates (TF-MxNE) by `Alex Gramfort`_ and `Daniel Strohmeier`_


API
~~~
   - Added nave parameter to source_induced_power() and source_band_induced_power(), use nave=1 by default (wrong nave was used before).

   - Use mne.layout.read_layout instead of mne.layout.Layout to read a layout file (.lout)

   - Use raw.time_as_index instead of time_to_index (still works but is deprecated).

   - The artifacts module (mne.artifacts) is now merged into mne.preprocessing

   - Epochs objects now also take dicts as values for the event_id argument. They now can represent multiple conditions.

Authors
~~~~~~~~~

The committer list for this release is the following (preceded by number
of commits):

   * 313  Eric Larson
   * 226  Alexandre Gramfort
   * 219  Denis A. Engemann
   * 104  Christian Brodbeck
   *  85  Martin Luessi
   *   6  Daniel Strohmeier
   *   4  Teon Brooks
   *   1  Dan G. Wakeman


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
~~~~~~~~~

The committer list for this release is the following (preceded by number
of commits):

   * 118 Alexandre Gramfort
   * 81  Martin Luessi
   * 15  Daniel Strohmeier
   *  4  Christian Brodbeck
   *  4  Louis Thibault
   *  2  Brad Buran

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

.. _Martin Luessi: http://www.nmr.mgh.harvard.edu/martinos/people/showPerson.php?people_id=1600

.. _Yaroslav Halchenko: http://www.onerussian.com/

.. _Daniel Strohmeier: http://www.tu-ilmenau.de/bmti/fachgebiete/biomedizinische-technik/dipl-ing-daniel-strohmeier/

.. _Eric Larson: http://faculty.washington.edu/larsoner/

.. _Denis Engemann: https://github.com/dengemann

.. _Christian Brodbeck: https://github.com/christianmbrodbeck

.. _Simon Kornblith: http://simonster.com

.. _Teon Brooks: https://files.nyu.edu/tlb331/public/

.. _Mainak Jas: http://ltl.tkk.fi/wiki/Mainak_Jas
