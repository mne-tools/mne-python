.. _changes_0_7:

Version 0.7
-----------

Changelog
~~~~~~~~~

- Add capability for real-time feedback via trigger codes using StimServer and StimClient classes by `Mainak Jas`_

- New decoding module for MEG analysis containing sklearn compatible transformers by `Mainak Jas`_ and `Alex Gramfort`_

- New realtime module containing RtEpochs, RtClient and MockRtClient class by `Martin Luessi`_, `Christopher Dinh`_, `Alex Gramfort`_, `Denis Engemann`_ and `Mainak Jas`_

- Allow picking normal orientation in LCMV beamformers by `Roman Goj`_, `Alex Gramfort`_, `Denis Engemann`_ and `Martin Luessi`_

- Add printing summary to terminal for measurement info by `Denis Engemann`_

- Add read and write info attribute ICA objects by `Denis Engemann`_

- Decoding with Common Spatial Patterns (CSP) by `Romain Trachel`_ and `Alex Gramfort`_

- Add ICA ``plot_topomap`` function and method for displaying the spatial sensitivity of ICA sources by `Denis Engemann`_

- Plotting multiple brain views at once by `Eric Larson`_

- Reading head positions from raw FIFF files by `Eric Larson`_

- Add decimation parameter to ICA.decompose*  methods by `Denis Engemann`_ and `Alex Gramfort`_

- Add rejection buffer to ICA.decompose* methods by `Denis Engemann`_ and `Alex Gramfort`_

- Improve ICA computation speed and memory usage by `Denis Engemann`_ and `Alex Gramfort`_

- Add polygonal surface decimation function to preprocess head surfaces for coregistration by `Denis Engemann`_ and `Alex Gramfort`_

- DICS time-frequency beamforming for epochs, evoked and for estimating source power by `Roman Goj`_, `Alex Gramfort`_ and `Denis Engemann`_

- Add method for computing cross-spectral density (CSD) from epochs and class for storing CSD data by `Roman Goj`_, `Alex Gramfort`_ and `Denis Engemann`_

- Add trellis plot function and method for visualizing single epochs by `Denis Engemann`_

- Add fiducials read/write support by `Christian Brodbeck`_ and `Alex Gramfort`_

- Add select / drop bad channels in `plot_raw` on click by `Denis Engemann`_

- Add `ico` and `oct` source space creation in native Python by `Eric Larson`_

- Add interactive rejection of bad trials in ``plot_epochs`` by `Denis Engemann`_

- Add morph map calculation by `Eric Larson`_ and `Martin Luessi`_

- Add volume and discrete source space creation and I/O support by `Eric Larson`_

- Time-frequency beamforming to obtain spectrograms in source space using LCMV and DICS by `Roman Goj`_, `Alex Gramfort`_ and `Denis Engemann`_

- Compute epochs power spectral density function by `Denis Engemann`_

- Plot raw power spectral density by `Eric Larson`_

- Computing of distances along the cortical surface by `Eric Larson`_

- Add reading BEM solutions by `Eric Larson`_

- Add forward solution calculation in native Python by `Eric Larson`_

- Add (Neuro)debian license compatibility by `Eric Larson`_

- Automatic QRS threshold selection for ECG events by `Eric Larson`_

- Add Travis continuous integration service by `Denis Engemann`_

- Add SPM face data set by `Denis Engemann`_ `Martin Luessi`_ and `Alex Gramfort`_

- Support reading of EDF+,BDF data by `Teon Brooks`_

- Tools for scaling MRIs (mne.scale_mri) by `Christian Brodbeck`_

- GUI for head-MRI coregistration (mne.gui.coregistration) by `Christian Brodbeck`_

- GUI for ki2fiff conversion (mne.gui.kit2fiff) by `Christian Brodbeck`_

- Support reading of EEG BrainVision data by `Teon Brooks`_

- Improve CTF compensation handling by `Martin Luessi`_ and `Eric Larson`_

- Improve and extend automated layout guessing by `Denis Engemann`_

- Add Continuum Analytics Anaconda support by `Denis Engemann`_

- Add `subtract evoked` option to beamformers by `Andrew Dykstra`_

- Add new `transform` method to SourceEstimate(s) by `Andrew Dykstra`_

API
~~~

- The pick_normal parameter for minimum norm solvers has been renamed as ``pick_ori`` and normal orientation picking is now achieved by passing the value "normal" for the `pick_ori` parameter.

- ICA objects now expose the measurement info of the object fitted.

- Average EEG reference is now added by default to Raw instances.

- Removed deprecated read/write_stc/w, use SourceEstimate methods instead

- The ``chs`` argument in ``mne.layouts.find_layout`` is deprecated and will be removed in MNE-Python 0.9. Use ``info`` instead.

- ``plot_evoked`` and ``Epochs.plot`` now open a new figure by default. To plot on an existing figure please specify the `axes` parameter.


Authors
~~~~~~~

The committer list for this release is the following (preceded by number
of commits):

* 336  Denis A. Engemann
* 202  Eric Larson
* 193  Roman Goj
* 138  Alexandre Gramfort
*  99  Mainak Jas
*  75  Christian Brodbeck
*  60  Martin Luessi
*  40  Teon Brooks
*  29  Romain Trachel
*  28  Andrew Dykstra
*  12  Mark Wronkiewicz
*  10  Christoph Dinh
*   8  Alan Leggitt
*   3  Yaroslav Halchenko
*   3  Daniel Strohmeier
*   2  Mads Jensen
*   2  Praveen Sripad
*   1  Luke Bloy
*   1  Emanuele Olivetti
*   1  Yousra BEKHTI
