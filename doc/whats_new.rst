What's new
==========
..
    Note, we are now using links to highlight new functions and classes.
    Please be sure to follow the examples below like :func:`mne.stats.f_mway_rm`, so the whats_new page will have a link to the function/class documentation.

.. _changes_0_11:

Version 0.11
------------

Changelog
~~~~~~~~~

    - Maxwell filtering (SSS) implemented in :func:`mne.preprocessing.maxwell_filter` by `Mark Wronkiewicz`_ as part of Google Summer of Code, with help from `Samu Taulu`_, `Jukka Nenonen`_, and `Jussi Nurminen`_. Our implementation includes support for:

        - Fine calibration

        - Cross-talk correction

        - Temporal SSS (tSSS)

        - Head position translation

        - Internal component regularization

    - Compensation for movements using Maxwell filtering on epoched data in :func:`mne.epochs.average_movements` by `Eric Larson`_ and `Samu Taulu`_

    - Add reader for Nicolet files in :func:`mne.io.read_raw_nicolet` by `Jaakko Leppakangas`_

    - Add FIFF persistence for ICA labels by `Denis Engemann`_

    - Display ICA labels in :func:`mne.viz.plot_ica_scores` and :func:`mne.viz.plot_ica_sources` (for evoked objects) by `Denis Engemann`_

    - Plot spatially color coded lines in :func:`mne.Evoked.plot` by `Jona Sassenhagen`_ and `Jaakko Leppakangas`_

    - Add reader for CTF data in :func:`mne.io.read_raw_ctf` by `Eric Larson`_

    - Add support for Brainvision v2 in :func:`mne.io.read_raw_brainvision` by `Teon Brooks`_
    
    - Improve speed of generalization across time :class:`mne.decoding.GeneralizationAcrossTime` decoding up to a factor of seven by `Jean-Remi King`_ and `Federico Raimondo`_ and `Denis Engemann`_.

    - Add the explained variance for each principal component, ``explained_var``, key to the :class:`mne.io.Projection` by `Teon Brooks`_

    - Added methods :func:`mne.Epochs.add_eeg_average_proj`, :func:`mne.io.Raw.add_eeg_average_proj`, and :func:`mne.Evoked.add_eeg_average_proj` to add an average EEG reference.

    - Add reader for EEGLAB data in :func:`mne.io.read_raw_eeglab` and :func:`mne.read_epochs_eeglab` by `Mainak Jas`_

BUG
~~~

    - Fix bug that prevented homogeneous bem surfaces to be displayed in HTML reports by `Denis Engemann`_

    - Added safeguards against ``None`` and negative values in reject and flat parameters in :class:`mne.Epochs` by `Eric Larson`_

    - Fix train and test time window-length in :class:`mne.decoding.GeneralizationAcrossTime` by `Jean-Remi King`_

    - Added lower bound in :func:`mne.stats.linear_regression` on p-values ``p_val`` (and resulting ``mlog10_p_val``) using double floating point arithmetic limits by `Eric Larson`_

    - Fix channel name pick in :func:`mne.Evoked.get_peak` method by `Alex Gramfort`_

    - Fix drop percentages to take into account ``ignore`` option in :func:`mne.viz.plot_drop_log` and :func:`mne.Epochs.plot_drop_log` by `Eric Larson`_.

    - :class:`mne.EpochsArray` no longer has an average EEG reference silently added (but not applied to the data) by default. Use :func:`mne.EpochsArray.add_eeg_ref` to properly add one.

API
~~~

    - :func:`mne.io.read_raw_brainvision` now has ``event_id`` argument to assign non-standard trigger events to a trigger value by `Teon Brooks`_

    - :func:`mne.read_epochs` now has ``add_eeg_ref=False`` by default, since average EEG reference can be added before writing or after reading using the method :func:`mne.Epochs.add_eeg_ref`.

    - :class:`mne.EpochsArray` no longer has an average EEG reference silently added (but not applied to the data) by default. Use :func:`mne.EpochsArray.add_eeg_average_proj` to properly add one.

Authors
~~~~~~~

The committer list for this release is the following (preceded by number of commits):

   171  Eric Larson
   117  Jaakko Leppakangas
    58  Jona Sassenhagen
    52  Mainak Jas
    46  Alexandre Gramfort
    33  Denis A. Engemann
    28  Teon Brooks
    24  Clemens Brunner
    23  Christian Brodbeck
    15  Mark Wronkiewicz
    10  Jean-Remi King
     5  Marijn van Vliet
     3  Fede Raimondo
     2  Alexander Rudiuk
     2  emilyps14
     2  lennyvarghese
     1  Marian Dovgialo

.. _changes_0_10:

Version 0.10
------------

Changelog
~~~~~~~~~

    - Add support for generalized M-way repeated measures ANOVA for fully balanced designs with :func:`mne.stats.f_mway_rm` by `Denis Engemann`_

    - Add epochs browser to interactively view and manipulate epochs with :func:`mne.viz.plot_epochs` by `Jaakko Leppakangas`_

    - Speed up TF-MxNE inverse solver with block coordinate descent by `Daniel Strohmeier`_ and `Yousra Bekhti`_

    - Speed up zero-phase overlap-add (default) filtering by a factor of up to 2 using linearity by `Ross Maddox`_ and `Eric Larson`_

    - Add support for scaling and adjusting the number of channels/time per view by `Jaakko Leppakangas`_

    - Add support to toggle the show/hide state of all sections with a single keypress ('t') in :class:`mne.report.Report` by `Mainak Jas`_

    - Add support for BEM model creation :func:`mne.make_bem_model` by `Eric Larson`_

    - Add support for BEM solution computation :func:`mne.make_bem_solution` by `Eric Larson`_

    - Add ICA plotters for raw and epoch components by `Jaakko Leppakangas`_

    - Add new object :class:`mne.decoding.TimeDecoding` for decoding sensors' evoked response across time by `Jean-Remi King`_

    - Add command ``mne freeview_bem_surfaces`` to quickly check BEM surfaces with Freeview by `Alex Gramfort`_.

    - Add support for splitting epochs into multiple files in :func:`mne.Epochs.save` by `Mainak Jas`_ and `Alex Gramfort`_

    - Add support for jointly resampling a raw object and event matrix to avoid issues with resampling status channels by `Marijn van Vliet`_

    - Add new method :class:`mne.preprocessing.Xdawn` for denoising and decoding of ERP/ERF by `Alexandre Barachant`_

    - Add support for plotting patterns/filters in :class:`mne.decoding.csp.CSP` and :class:`mne.decoding.base.LinearModel` by `Romain Trachel`_

    - Add new object :class:`mne.decoding.base.LinearModel` for decoding M/EEG data and interpreting coefficients of linear models with patterns attribute by `Romain Trachel`_ and `Alex Gramfort`_

    - Add support to append new channels to an object from a list of other objects by `Chris Holdgraf`_

    - Add interactive plotting of topomap from time-frequency representation by `Jaakko Leppakangas`_

    - Add ``plot_topo`` method to ``Evoked`` object by `Jaakko Leppakangas`_

    - Add fetcher :mod:`mne.datasets.brainstorm` for datasets used by Brainstorm in their tutorials by `Mainak Jas`_

    - Add interactive plotting of single trials by right clicking on channel name in epochs browser by `Jaakko Leppakangas`_

    - New logos and logo generation script by `Daniel McCloy`_

    - Add ability to plot topomap with a "skirt" (channels outside of the head circle) by `Marijn van Vliet`_

    - Add multiple options to ICA infomax and extended infomax algorithms (number of subgaussian components, computation of bias, iteration status printing), enabling equivalent computations to those performed by EEGLAB by `Jair Montoya Martinez`_

    - Add :func:`mne.Epochs.apply_baseline` method to ``Epochs`` objects by `Teon Brooks`_

    - Add ``preload`` argument to :func:`mne.read_epochs` to enable on-demand reads from disk by `Eric Larson`_

    - Big rewrite of simulation module by `Yousra Bekhti`_, `Mark Wronkiewicz`_, `Eric Larson`_ and `Alex Gramfort`_. Allows to simulate raw with artefacts (ECG, EOG) and evoked data, exploiting the forward solution. See :func:`mne.simulation.simulate_raw`, :func:`mne.simulation.simulate_evoked` and :func:`mne.simulation.simulate_sparse_stc`

    - Add :func:`mne.Epochs.load_data` method to :class:`mne.Epochs` by `Teon Brooks`_

    - Add support for drawing topomaps by selecting an area in :func:`mne.Evoked.plot` by `Jaakko Leppakangas`_

    - Add support for finding peaks in evoked data in :func:`mne.Evoked.plot_topomap` by `Jona Sassenhagen`_ and `Jaakko Leppakangas`_

    - Add source space morphing in :func:`morph_source_spaces` and :func:`SourceEstimate.to_original_src` by `Eric Larson`_ and `Denis Engemann`_

   - Adapt ``corrmap`` function (Viola et al. 2009) to semi-automatically detect similar ICs across data sets by `Jona Sassenhagen`_ and `Denis Engemann`_ and `Eric Larson`_

   - New ``mne flash_bem`` command to compute BEM surfaces from Flash MRI images by `Lorenzo Desantis`_, `Alex Gramfort`_ and `Eric Larson`_. See :func:`mne.bem.utils.make_flash_bem`.

   - New gfp parameter in :func:`mne.Evoked.plot` method to display Global Field Power (GFP) by `Eric Larson`_.

    - Add :func:`mne.report.Report.add_slider_to_section` methods to :class:`mne.report.Report` by `Teon Brooks`_

BUG
~~~

    - Fix ``mne.io.add_reference_channels`` not setting ``info[nchan]`` correctly by `Federico Raimondo`_

    - Fix ``mne.stats.bonferroni_correction`` reject mask output to use corrected p-values by `Denis Engemann`_

    - Fix FFT filter artifacts when using short windows in overlap-add by `Eric Larson`_

    - Fix picking channels from forward operator could return a channel ordering different from ``info['chs']`` by `Chris Bailey`_

    - Fix dropping of events after downsampling stim channels by `Marijn van Vliet`_

    - Fix scaling in :func:``mne.viz.utils._setup_vmin_vmax`` by `Jaakko Leppakangas`_

    - Fix order of component selection in :class:`mne.decoding.csp.CSP` by `Clemens Brunner`_

API
~~~

    - Rename and deprecate ``mne.viz.plot_topo`` for ``mne.viz.plot_evoked_topo`` by `Jaakko Leppakangas`_

    - Deprecated :class: `mne.decoding.transformer.ConcatenateChannels` and replaced by :class: `mne.decoding.transformer.EpochsVectorizer` by `Romain Trachel`_

    - Deprecated `lws` and renamed `ledoit_wolf` for the ``reg`` argument in :class:`mne.decoding.csp.CSP` by `Romain Trachel`_

    - Redesigned and rewrote :func:`mne.Epochs.plot` (no backwards compatibility) during the GSOC 2015 by `Jaakko Leppakangas`_, `Mainak Jas`_, `Federico Raimondo`_ and `Denis Engemann`_

    - Deprecated and renamed :func:`mne.viz.plot_image_epochs` for :func:`mne.plot.plot_epochs_image` by `Teon Brooks`_

    - ``picks`` argument has been added to :func:`mne.time_frequency.tfr_morlet`, :func:`mne.time_frequency.tfr_multitaper` by `Teon Brooks`_

    - :func:`mne.io.Raw.preload_data` has been deprecated for :func:`mne.io.Raw.load_data` by `Teon Brooks`_

    - ``RawBrainVision`` objects now always have event channel ``'STI 014'``, and recordings with no events will have this channel set to zero by `Eric Larson`_

Authors
~~~~~~~

The committer list for this release is the following (preceded by number of commits):

   273  Eric Larson
   270  Jaakko Leppakangas
   194  Alexandre Gramfort
   128  Denis A. Engemann
   114  Jona Sassenhagen
   107  Mark Wronkiewicz
    97  Teon Brooks
    81  Lorenzo De Santis
    55  Yousra Bekhti
    54  Jean-Remi King
    48  Romain Trachel
    45  Mainak Jas
    40  Alexandre Barachant
    32  Marijn van Vliet
    26  Jair Montoya
    22  Chris Holdgraf
    16  Christopher J. Bailey
     7  Christian Brodbeck
     5  Natalie Klein
     5  Fede Raimondo
     5  Alan Leggitt
     5  Roan LaPlante
     5  Ross Maddox
     4  Dan G. Wakeman
     3  Daniel McCloy
     3  Daniel Strohmeier
     1  Jussi Nurminen

.. _changes_0_9:

Version 0.9
-----------

Changelog
~~~~~~~~~

   - Add support for mayavi figures in ``add_section`` method in Report by `Mainak Jas`_

   - Add extract volumes of interest from freesurfer segmentation and setup as volume source space by `Alan Leggitt`_

   - Add support to combine source spaces of different types by `Alan Leggitt`_

   - Add support for source estimate for mixed source spaces by `Alan Leggitt`_

   - Add ``SourceSpaces.save_as_volume`` method by `Alan Leggitt`_

   - Automatically compute proper box sizes when generating layouts on the fly by `Marijn van Vliet`_

   - Average evoked topographies across time points by `Denis Engemann`_

   - Add option to Report class to save images as vector graphics (SVG) by `Denis Engemann`_

   - Add events count to ``mne.viz.plot_events`` by `Denis Engemann`_

   - Add support for stereotactic EEG (sEEG) channel type by `Marmaduke Woodman`_

   - Add support for montage files by `Denis Engemann`_, `Marijn van Vliet`_, `Jona Sassenhagen`_, `Alex Gramfort`_ and `Teon Brooks`_

   - Add support for spatiotemporal permutation clustering on sensors by `Denis Engemann`_

   - Add support for multitaper time-frequency analysis by `Hari Bharadwaj`_

   - Add Stockwell (S) transform for time-frequency representations by `Denis Engemann`_ and `Alex Gramfort`_

   - Add reading and writing support for time frequency data (AverageTFR objects) by  `Denis Engemann`_

   - Add reading and writing support for digitizer data, and function for adding dig points to info by `Teon Brooks`_

   - Add  ``plot_projs_topomap`` method to ``Raw``, ``Epochs`` and ``Evoked`` objects by `Teon Brooks`_

   - Add EEG (based on spherical splines) and MEG (based on field interpolation) bad channel interpolation method to ``Raw``, ``Epochs`` and ``Evoked`` objects
     by `Denis Engemann`_ and `Mainak Jas`_

   - Add parameter to ``whiten_evoked``, ``compute_whitener`` and ``prepare_noise_cov`` to set the exact rank by `Martin Luessi`_ and `Denis Engemann`_

   - Add fiff I/O for processing history and MaxFilter info by `Denis Engemann`_ and `Eric Larson`_

   - Add automated regularization with support for multiple sensor types to ``compute_covariance`` by `Denis Engemann`_ and `Alex Gramfort`_

   - Add ``Evoked.plot_white`` method to diagnose the quality of the estimated noise covariance and its impact on spatial whitening by `Denis Engemann`_ and `Alex Gramfort`_

   - Add ``mne.evoked.grand_average`` function to compute grand average of Evoked data while interpolating bad EEG channels if necessary by `Mads Jensen`_ and `Alex Gramfort`_

   - Improve EEG referencing support and add support for bipolar referencing by `Marijn van Vliet`_ and `Alex Gramfort`_

   - Enable TFR calculation on Evoked objects by `Eric Larson`_

   - Add support for combining Evoked datasets with arbitrary weights (e.g., for oddball paradigms) by `Eric Larson`_ and `Alex Gramfort`_

   - Add support for concatenating a list of Epochs objects by `Denis Engemann`_

   - Labels support subtraction (``label_1 - label_2``) by `Christian Brodbeck`_

   - Add GeneralizationAcrossTime object with support for cross-condition generalization by `Jean-Remi King`_ and `Denis Engemann`_

   - Add support for single dipole fitting by `Eric Larson`_

   - Add support for spherical models in forward calculations by `Eric Larson`_

   - Add support for SNR estimation by `Eric Larson`_

   - Add support for Savitsky-Golay filtering of Evoked and Epochs by `Eric Larson`_

   - Add support for adding an empty reference channel to data by `Teon Brooks`_

   - Add reader function ``mne.io.read_raw_fif`` for Raw FIF files by `Teon Brooks`_

   - Add example of creating MNE objects from arbitrary data and NEO files by `Jaakko Leppakangas`_

   - Add ``plot_psd`` and ``plot_psd_topomap`` methods to epochs by `Yousra Bekhti`_, `Eric Larson`_ and `Denis Engemann`_

   - ``evoked.pick_types``, ``epochs.pick_types``, and ``tfr.pick_types`` added by `Eric Larson`_

   - ``rename_channels`` and ``set_channel_types`` added as methods to ``Raw``, ``Epochs`` and ``Evoked`` objects by `Teon Brooks`_

   - Add RAP-MUSIC inverse method by `Yousra Bekhti`_ and `Alex Gramfort`_

   - Add ``evoked.as_type`` to  allow remapping data in MEG channels to virtual magnetometer or gradiometer channels by `Mainak Jas`_

   - Add :func:`mne.report.Report.add_bem_to_section`, :func:`mne.report.Report.add_htmls_to_section` methods to :class:`mne.report.Report` by `Teon Brooks`_

   - Add support for KIT epochs files with ``read_epochs_kit`` by `Teon Brooks`_

   - Add whitening plots for evokeds to ``mne.Report`` by `Mainak Jas`_

   - Add ``DigMontage`` class and reader to interface with digitization info by `Teon Brooks`_ and `Christian Brodbeck`_

   - Add ``set_montage`` method to the ``Raw``, ``Epochs``, and ``Evoked`` objects by `Teon Brooks`_ and `Denis Engemann`_

   - Add support for capturing sensor positions when clicking on an image by `Chris Holdgraf`_

   - Add support for custom sensor positions when creating Layout objects by `Chris Holdgraf`_

BUG
~~~

   - Fix energy conservation for STFT with tight frames by `Daniel Strohmeier`_

   - Fix incorrect data matrix when tfr was plotted with parameters ``tmin``, ``tmax``, ``fmin`` and ``fmax`` by `Mainak Jas`_

   - Fix channel names in topomaps by `Alex Gramfort`_

   - Fix mapping of ``l_trans_bandwidth`` (to low frequency) and ``h_trans_bandwidth`` (to high frequency) in ``_BaseRaw.filter`` by `Denis Engemann`_

   - Fix scaling source spaces when distances have to be recomputed by `Christian Brodbeck`_

   - Fix repeated samples in client to FieldTrip buffer by `Mainak Jas`_ and `Federico Raimondo`_

   - Fix highpass and lowpass units read from Brainvision vhdr files by `Alex Gramfort`_

   - Add missing attributes for BrainVision and KIT systems needed for resample by `Teon Brooks`_

   - Fix file extensions of SSP projection files written by mne commands (from _proj.fif to -prof.fif) by `Alex Gramfort`_

   - Generating EEG layouts no longer requires digitization points by `Marijn van Vliet`_

   - Add missing attributes to BTI, KIT, and BrainVision by `Eric Larson`_

   - The API change to the edf, brainvision, and egi break backwards compatibility for when importing eeg data by `Teon Brooks`_

   - Fix bug in ``mne.viz.plot_topo`` if ylim was passed for single sensor layouts by `Denis Engemann`_

   - Average reference projections will no longer by automatically added after applying a custom EEG reference by `Marijn van Vliet`_

   - Fix picks argument to filter in n dimensions (affects FilterEstimator), and highpass filter in FilterEstimator by `Mainak Jas`_

   - Fix beamformer code LCMV/DICS for CTF data with reference channels by `Denis Engemann`_ and `Alex Gramfort`_

   - Fix scalings for bad EEG channels in ``mne.viz.plot_topo`` by `Marijn van Vliet`_

   - Fix EGI reading when no events are present by `Federico Raimondo`_

   - Add functionality to determine plot limits automatically or by data percentiles by `Mark Wronkiewicz`_

   - Fix bug in mne.io.edf where the channel offsets were ommitted in the voltage calculations by `Teon Brooks`_

   - Decouple section ordering in command-line from python interface for mne-report by `Mainak Jas`_

   - Fix bug with ICA resetting by `Denis Engemann`_

API
~~~

   - apply_inverse functions have a new boolean parameter ``prepared`` which saves computation time by calling ``prepare_inverse_operator`` only if it is False

   - find_events and read_events functions have a new parameter ``mask`` to set some bits to a don't care state by `Teon Brooks`_

   - New channels module including layouts, electrode montages, and neighbor definitions of sensors which deprecates ``mne.layouts`` by `Denis Engemann`_

   - ``read_raw_brainvision``, ``read_raw_edf``, ``read_raw_egi`` all use a standard montage import by `Teon Brooks`_

   - Fix missing calibration factors for ``mne.io.egi.read_raw_egi`` by `Denis Engemann`_ and `Federico Raimondo`_

   - Allow multiple filename patterns as a list (e.g., \*raw.fif and \*-eve.fif) to be parsed by mne report in ``Report.parse_folder()`` by `Mainak Jas`_

   - ``read_hsp``, ``read_elp``, and ``write_hsp``, ``write_mrk`` were removed and made private by `Teon Brooks`_

   - When computing the noise covariance or MNE inverse solutions, the rank is estimated empirically using more sensitive thresholds, which stabilizes results by `Denis Engemann`_ and `Eric Larson`_ and `Alex Gramfort`_

   - Raw FIFF files can be preloaded after class instantiation using ``raw.preload_data()``

   - Add ``label`` parameter to ``apply_inverse`` by `Teon Brooks`_

   - Deprecated ``label_time_courses`` for ``in_label`` method in `SourceEstimate` by `Teon Brooks`_

   - Deprecated ``as_data_frame`` for ``to_data_frame`` by `Chris Holdgraf`_

   - Add ``transform``, ``unit`` parameters to ``read_montage`` by `Teon Brooks`_

   - Deprecated ``fmin, fmid, fmax`` in stc.plot and added ``clim`` by `Mark Wronkiewicz`_

   - Use ``scipy.signal.welch`` instead of matplotlib.psd inside ``compute_raw_psd`` and ``compute_epochs_psd`` by `Yousra Bekhti`_ `Eric Larson`_ and `Denis Engemann`_. As a consquence, ``Raw.plot_raw_psds`` has been deprecated.

   - ``Raw`` instances returned by ``mne.forward.apply_forward_raw`` now always have times starting from
     zero to be consistent with all other ``Raw`` instances. To get the former ``start`` and ``stop`` times,
     use ``raw.first_samp / raw.info['sfreq']`` and ``raw.last_samp / raw.info['sfreq']``.

   - ``pick_types_evoked`` has been deprecated in favor of ``evoked.pick_types``.

   - Deprecated changing the sensor type of channels in ``rename_channels`` by `Teon Brooks`_

   - CUDA is no longer initialized at module import, but only when first used.

   - ``add_figs_to_section`` and ``add_images_to_section`` now have a ``textbox`` parameter to add comments to the image by `Teon Brooks`_

   - Deprecated ``iir_filter_raw`` for ``fit_iir_model_raw``.

   - Add ``montage`` parameter to the ``create_info`` function to create the info using montages by `Teon Brooks`_

Authors
~~~~~~~

The committer list for this release is the following (preceded by number of commits):

   515  Eric Larson
   343  Denis A. Engemann
   304  Alexandre Gramfort
   300  Teon Brooks
   142  Mainak Jas
   119  Jean-Remi King
    77  Alan Leggitt
    75  Marijn van Vliet
    63  Chris Holdgraf
    57  Yousra Bekhti
    49  Mark Wronkiewicz
    44  Christian Brodbeck
    30  Jona Sassenhagen
    29  Hari Bharadwaj
    27  Clément Moutard
    24  Ingoo Lee
    18  Marmaduke Woodman
    16  Martin Luessi
    10  Jaakko Leppakangas
     9  Andrew Dykstra
     9  Daniel Strohmeier
     7  kjs
     6  Dan G. Wakeman
     5  Federico Raimondo
     3  Basile Pinsard
     3  Christoph Dinh
     3  Hafeza Anevar
     2  Martin Billinger
     2  Roan LaPlante
     1  Manoj Kumar
     1  Matt Tucker
     1  Romain Trachel
     1  mads jensen
     1  sviter

.. _changes_0_8:

Version 0.8
-----------

Changelog
~~~~~~~~~

   - Add Python3 support by `Nick Ward`_, `Alex Gramfort`_, `Denis Engemann`_, and `Eric Larson`_

   - Add ``get_peak`` method for evoked and stc objects by  `Denis Engemann`_

   - Add ``iter_topography`` function for radically simplified custom sensor topography plotting by `Denis Engemann`_

   - Add field line interpolation by `Eric Larson`_

   - Add full provenance tacking for epochs and improve ``drop_log`` by `Tal Linzen`_, `Alex Gramfort`_ and `Denis Engemann`_

   - Add systematic contains method to ``Raw``, ``Epochs`` and ``Evoked`` for channel type membership testing by `Denis Engemann`_

   - Add fiff unicode writing and reading support by `Denis Engemann`_

   - Add 3D MEG/EEG field plotting function and evoked method by `Denis Engemann`_ and  `Alex Gramfort`_

   - Add consistent channel-dropping methods to ``Raw``, ``Epochs`` and ``Evoked`` by `Denis Engemann`_ and  `Alex Gramfort`_

   - Add ``equalize_channnels`` function to set common channels for a list of ``Raw``, ``Epochs``, or ``Evoked`` objects by `Denis Engemann`_

   - Add ``plot_events`` function to visually display paradigm by `Alex Gramfort`_

   - Improved connectivity circle plot by `Martin Luessi`_

   - Add ability to anonymize measurement info by `Eric Larson`_

   - Add callback to connectivity circle plot to isolate connections to clicked nodes `Roan LaPlante`_

   - Add ability to add patch information to source spaces by `Eric Larson`_

   - Add ``split_label`` function to divide labels into multiple parts by `Christian Brodbeck`_

   - Add ``color`` attribute to ``Label`` objects by `Christian Brodbeck`_

   - Add ``max`` mode for ``extract_label_time_course`` by `Mads Jensen`_

   - Add ``rename_channels`` function to change channel names and types in info object by `Dan Wakeman`_ and `Denis Engemann`_

   - Add  ``compute_ems`` function to extract the time course of experimental effects by `Denis Engemann`_, `Sébastien Marti`_ and `Alex Gramfort`_

   - Add option to expand Labels defined in a source space to the original surface (``Label.fill()``) by `Christian Brodbeck`_

   - GUIs can be invoked form the command line using `$ mne coreg` and `$ mne kit2fiff` by `Christian Brodbeck`_

   - Add ``add_channels_epochs`` function to combine different recordings at the Epochs level by `Christian Brodbeck`_ and `Denis Engemann`_

   - Add support for EGI Netstation simple binary files by `Denis Engemann`_

   - Add support for treating arbitrary data (numpy ndarray) as a Raw instance by `Eric Larson`_

   - Support for parsing the EDF+ annotation channel by `Martin Billinger`_

   - Add EpochsArray constructor for creating epochs from numpy arrays by `Denis Engemann`_ and `Federico Raimondo`_

   - Add connector to FieldTrip realtime client by `Mainak Jas`_

   - Add color and event_id with legend options in plot_events in viz.py by `Cathy Nangini`_

   - Add ``events_list`` parameter to ``mne.concatenate_raws`` to concatenate events corresponding to runs by `Denis Engemann`_

   - Add ``read_ch_connectivity`` function to read FieldTrip neighbor template .mat files and obtain sensor adjacency matrices by `Denis Engemann`_

   - Add display of head in helmet from -trans.fif file to check coregistration quality by `Mainak Jas`_

   - Add ``raw.add_events`` to allow adding events to a raw file by `Eric Larson`_

   - Add ``plot_image`` method to Evoked object to display data as images by `Jean-Remi King`_ and `Alex Gramfort`_ and `Denis Engemann`_

   - Add BCI demo with CSP on motor imagery by `Martin Billinger`_

   - New ICA API with unified methods for processing ``Raw``, ``Epochs`` and ``Evoked`` objects by `Denis Engemann`_

   - Apply ICA at the evoked stage by `Denis Engemann`_

   - New ICA methods for visualizing unmixing quality, artifact detection and rejection by `Denis Engemann`_

   - Add ``pick_channels`` and ``drop_channels`` mixin class to pick and drop channels from ``Raw``, ``Epochs``, and ``Evoked`` objects by `Andrew Dykstra`_ and `Denis Engemann`_

   - Add ``EvokedArray`` class to create an Evoked object from an array by `Andrew Dykstra`_

   - Add ``plot_bem`` method to visualize BEM contours on MRI anatomical images by `Mainak Jas`_ and `Alex Gramfort`_

   - Add automated ECG detection using cross-trial phase statistics by `Denis Engemann`_ and `Juergen Dammers`_

   - Add Forward class to succintly display gain matrix info by `Andrew Dykstra`_

   - Add reading and writing of split raw files by `Martin Luessi`_

   - Add OLS regression function by `Tal Linzen`_, `Teon Brooks`_ and `Denis Engemann`_

   - Add computation of point spread and cross-talk functions for MNE type solutions by `Alex Gramfort`_ and `Olaf Hauk`_

   - Add mask parameter to `plot_evoked_topomap` and ``evoked.plot_topomap`` by `Denis Engemann`_ and `Alex Gramfort`_

   - Add infomax and extended infomax ICA by `Denis Engemann`_, `Juergen Dammers`_ and `Lukas Breuer`_ and `Federico Raimondo`_

   - Aesthetically redesign interpolated topography plots by `Denis Engemann`_ and `Alex Gramfort`_

   - Simplify sensor space time-frequency analysis API with ``tfr_morlet`` function by `Alex Gramfort`_ and `Denis Engemann`_

   - Add new somatosensory MEG dataset with nice time-frequency content by `Alex Gramfort`_

   - Add HDF5 write/read support for SourceEstimates by `Eric Larson`_

   - Add InverseOperator class to display inverse operator info by `Mainak Jas`_

   - Add `$ mne report` command to generate html reports of MEG/EEG data analysis pipelines by `Mainak Jas`_, `Alex Gramfort`_ and `Denis Engemann`_

   - Improve ICA verbosity with regard to rank reduction by `Denis Engemann`_

BUG
~~~

   - Fix incorrect ``times`` attribute when stc was computed using ``apply_inverse`` after decimation at epochs stage for certain, arbitrary sample frequencies by `Denis Engemann`_

   - Fix corner case error for step-down-in-jumps permutation test (when step-down threshold was high enough to include all clusters) by `Eric Larson`_

   - Fix selection of total number of components via float when picking ICA sources by `Denis Engemann`_ and `Qunxi Dong`_

   - Fix writing and reading transforms after modification in measurment info by `Denis Engemann`_ and `Martin Luessi`_ and `Eric Larson`_

   - Fix pre-whitening / rescaling when estimating ICA on multiple channels without covariance by `Denis Engemann`_

   - Fix ICA pre-whitening, avoid recomputation when applying ICA to new data by `Denis Engemann`_

API
~~~

   - The minimum numpy version has been increased to 1.6 from 1.4.

   - Epochs object now has a selection attribute to track provenance of selected Epochs. The length of the drop_log attribute is now the same as the length of the original events passed to Epochs. In earlier versions it had the length of the events filtered by event_id. Epochs has also now a plot_drop_log method.

   - Deprecate Epochs.drop_picks in favor of a new method called drop_channels

   - Deprecate ``labels_from_parc`` and ``parc_from_labels`` in favor of ``read_labels_from_annot`` and ``write_labels_to_annot``

   - The default of the new add_dist option of ``setup_source_space`` to add patch information will change from False to True in MNE-Python 0.9

   - Deprecate ``read_evoked`` and ``write_evoked`` in favor of ``read_evokeds`` and ``write_evokeds``. read_evokeds will return all `Evoked` instances in a file by default.

   - Deprecate ``setno`` in favor of ``condition`` in the initialization of an Evoked instance. This affects ``mne.fiff.Evoked`` and ``read_evokeds``, but not ``read_evoked``.

   - Deprecate ``mne.fiff`` module, use ``mne.io`` instead e.g. ``mne.io.Raw`` instead of ``mne.fiff.Raw``.

   - Pick functions (e.g., ``pick_types``) are now in the mne namespace (e.g. use ``mne.pick_types``).

   - Deprecated ICA methods specific to one container type. Use ICA.fit, ICA.get_sources ICA.apply and ICA.plot_XXX for processing Raw, Epochs and Evoked objects.

   - The default smoothing method for ``mne.stc_to_label`` will change in v0.9, and the old method is deprecated.

   - As default, for ICA the maximum number of PCA components equals the number of channels passed. The number of PCA components used to reconstruct the sensor space signals now defaults to the maximum number of PCA components estimated.

Authors
~~~~~~~

The committer list for this release is the following (preceded by number of commits):

   * 418  Denis A. Engemann
   * 284  Alexandre Gramfort
   * 242  Eric Larson
   * 155  Christian Brodbeck
   * 144  Mainak Jas
   * 49  Martin Billinger
   * 49  Andrew Dykstra
   * 44  Tal Linzen
   * 37  Dan G. Wakeman
   * 36  Martin Luessi
   * 26  Teon Brooks
   * 20  Cathy Nangini
   * 15  Hari Bharadwaj
   * 15  Roman Goj
   * 10  Ross Maddox
   * 9  Marmaduke Woodman
   * 8  Praveen Sripad
   * 8  Tanay
   * 8  Roan LaPlante
   * 5  Saket Choudhary
   * 4  Nick Ward
   * 4  Mads Jensen
   * 3  Olaf Hauk
   * 3  Brad Buran
   * 2  Daniel Strohmeier
   * 2  Federico Raimondo
   * 2  Alan Leggitt
   * 1  Jean-Remi King
   * 1  Matti Hamalainen


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

   - ICA objects now expose the measurment info of the object fitted.

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

   - Add generator support for lcmv_epochs by `Denis Engemann`_

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
~~~~~~~

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
~~~~~~~

The committer list for this release is the following (preceded by number
of commits):

   * 33  Alexandre Gramfort
   * 12  Martin Luessi
   *  2  Yaroslav Halchenko
   *  1  Manfred Kitzbichler

.. _Alex Gramfort: http://alexandre.gramfort.net

.. _Martin Luessi: https://www.martinos.org/user/8245

.. _Yaroslav Halchenko: http://www.onerussian.com/

.. _Daniel Strohmeier: http://www.tu-ilmenau.de/bmti/fachgebiete/biomedizinische-technik/dipl-ing-daniel-strohmeier/

.. _Eric Larson: http://larsoner.com

.. _Denis Engemann: https://github.com/dengemann

.. _Christian Brodbeck: https://github.com/christianbrodbeck

.. _Simon Kornblith: http://simonster.com

.. _Teon Brooks: http://sites.google.com/a/nyu.edu/teon/

.. _Mainak Jas: http://ltl.tkk.fi/wiki/Mainak_Jas

.. _Roman Goj: http://romanmne.blogspot.co.uk

.. _Andrew Dykstra: https://github.com/adykstra

.. _Romain Trachel: http://www.lscp.net/braware/trachelBr.html

.. _Christopher Dinh: https://github.com/chdinh

.. _Nick Ward: http://www.ucl.ac.uk/ion/departments/sobell/Research/NWard

.. _Tal Linzen: http://tallinzen.net/

.. _Roan LaPlante: https://github.com/aestrivex

.. _Mads Jensen: https://github.com/MadsJensen

.. _Dan Wakeman: https://github.com/dgwakeman

.. _Qunxi Dong: https://github.com/dongqunxi

.. _Martin Billinger: https://github.com/kazemakase

.. _Federico Raimondo: https://github.com/fraimondo

.. _Cathy Nangini: https://github.com/KatiRG

.. _Jean-Remi King: https://github.com/kingjr

.. _Juergen Dammers: https://github.com/jdammers

.. _Olaf Hauk: http://www.neuroscience.cam.ac.uk/directory/profile.php?olafhauk

.. _Lukas Breuer: http://www.researchgate.net/profile/Lukas_Breuer

.. _Federico Raimondo: https://github.com/fraimondo

.. _Alan Leggitt: https://github.com/leggitta

.. _Marijn van Vliet: https://github.com/wmvanvliet

.. _Marmaduke Woodman: https://github.com/maedoc

.. _Jona Sassenhagen: https://github.com/jona-sassenhagen

.. _Hari Bharadwaj: http://www.haribharadwaj.com

.. _Chris Holdgraf: http://chrisholdgraf.com

.. _Jaakko Leppakangas: https://github.com/jaeilepp

.. _Yousra Bekhti: https://www.linkedin.com/pub/yousra-bekhti/56/886/421

.. _Mark Wronkiewicz: http://ilabs.washington.edu/graduate-students/bio/i-labs-mark-wronkiewicz

.. _Sébastien Marti: http://www.researchgate.net/profile/Sebastien_Marti

.. _Chris Bailey: https://github.com/cjayb

.. _Ross Maddox: http://faculty.washington.edu/rkmaddox/

.. _Alexandre Barachant: http://alexandre.barachant.org

.. _Daniel McCloy: http://dan.mccloy.info

.. _Jair Montoya Martinez: https://github.com/jmontoyam

.. _Samu Taulu: http://ilabs.washington.edu/institute-faculty/bio/i-labs-samu-taulu-dsc

.. _Lorenzo Desantis: https://github.com/lorenzo-desantis/

.. _Jukka Nenonen: https://www.linkedin.com/pub/jukka-nenonen/28/b5a/684

.. _Jussi Nurminen: https://scholar.google.fi/citations?user=R6CQz5wAAAAJ&hl=en

.. _Clemens Brunner: https://github.com/cle1109
