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

- Saving ICA sources to fif files and creating ICA topography layouts by `Denis Engemann`_

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
~~~~~~~

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
