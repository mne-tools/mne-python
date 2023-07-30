.. NOTE: we use cross-references to highlight new functions and classes.
   Please follow the examples below like :func:`mne.stats.f_mway_rm`, so the
   whats_new page will have a link to the function/class documentation.

.. NOTE: there are 3 separate sections for changes, based on type:
   - "Enhancements" for new features
   - "Bugs" for bug fixes
   - "API changes" for backward-incompatible changes

.. NOTE: changes from first-time contributors should be added to the TOP of
   the relevant section (Enhancements / Bugs / API changes), and should look
   like this (where xxxx is the pull request number):

       - description of enhancement/bugfix/API change (:gh:`xxxx` by
         :newcontrib:`Firstname Lastname`)

   Also add a corresponding entry for yourself in doc/changes/names.inc

.. _current:

Current (1.5.dev0)
------------------

Enhancements
~~~~~~~~~~~~
- Add reader for NSx files (Blackrock Microsystems Data Acquisition Systems) (:gh:`11432` by :newcontrib:`Proloy Das`)
- Add ``cmap`` argument for the :func:`mne.viz.plot_sensors` (:gh:`11720` by :newcontrib:`Gennadiy Belonosov`)
- Return unmodified instance if new sampling frequency is identical to the original in :meth:`mne.io.Raw.resample`, :meth:`mne.Epochs.resample`, :meth:`mne.Evoked.resample` and :meth:`mne.SourceEstimate.resample` (:gh:`11736` by :newcontrib:`Gennadiy Belonosov`)
- When failing to locate a file, we now print the full path in quotation marks to help spot accidentally added trailing spaces (:gh:`11718` by `Richard Höchenberger`_)
- Add standard montage lookup table for ``easycap-M43`` (:gh:`11744` by :newcontrib:`Diptyajit Das`)
- Added :class:`mne.preprocessing.eyetracking.Calibration` to store eye-tracking calibration info, and :func:`mne.preprocessing.eyetracking.read_eyelink_calibration` to read calibration data from EyeLink systems (:gh:`11719` by `Scott Huberty`_)
- Ocular :class:`mne.Annotations` read in by :func:`mne.io.read_raw_eyelink` are now channel aware. This means if the left eye blinked, the associated annotation will store this in the ``'ch_names'`` key. (:gh:`11746` by `Scott Huberty`_)
- Added :func:`mne.preprocessing.eyetracking.interpolate_blinks` to linear interpolate eyetrack signals during blink periods. (:gh:`11740` by `Scott Huberty`_)
- Added a section for combining eye-tracking and EEG data to the preprocessing tutorial "working with eye tracker data in MNE-Python" (:gh:`11770` by `Scott Huberty`_)
- Added a ``show_bytes`` option to :ref:`mne show_fiff` to show byte offsets (:gh:`11800` by `Eric Larson`_)
- Add :meth:`mne.Annotations.count` and :func:`mne.count_annotations` to count unique annotations (:gh:`11796` by `Clemens Brunner`_)

Bugs
~~~~
- Fix bug where epochs ``event_id`` was not kept by :func:`mne.channels.combine_channels` (:gh:`11786` by :newcontrib:`Samuel Louviot`)
- Fix bug where user-provided codec was not used to read annotations when loading EEGLAB ``.set`` files (:gh:`11773` by :newcontrib:`Yiping Zuo`)
- Fix bug that required curv.*h files to create Brain object (:gh:`11704` by :newcontrib:`Aaron Earle-Richardson`)
- Extended test to highlight bug in :func:`mne.stats.permutation_t_test` (:gh:`11575` by :newcontrib:`Joshua Calder-Travis`)
- Fix bug that used wrong indices for line/label styles (sometimes causing an ``IndexError``) in :meth:`mne.preprocessing.ICA.plot_sources` under certain conditions (:gh:`11808` by :newcontrib:`Joshua Calder-Travis`)
- Fix bug with :func:`~mne.io.read_raw_snirf` to handle files with measurement time containing milliseconds (:gh:`11804` by :newcontrib:`Daniel Tse`)
- Fix bug where :func:`mne.io.read_raw_cnt` imports unnecessary durations (:gh:`11828` by `Jacob Woessner`_)
- Fix bug where :meth:`mne.viz.Brain.add_volume_labels` used an incorrect orientation (:gh:`11730` by `Alex Rockhill`_)
- Fix bug with :func:`mne.forward.restrict_forward_to_label` where cortical patch information was not adjusted (:gh:`11694` by `Eric Larson`_)
- Fix bug with PySide6 compatibility (:gh:`11721` by `Eric Larson`_)
- Fix hanging interpreter with matplotlib figures using ``mne/viz/_mpl_figure.py`` in spyder console and jupyter notebooks (:gh:`11696` by `Mathieu Scheltienne`_)
- Fix bug with overlapping text for :meth:`mne.Evoked.plot` (:gh:`11698` by `Alex Rockhill`_)
- For :func:`mne.io.read_raw_eyelink`, the default value of the ``gap_description`` parameter is now ``'BAD_ACQ_SKIP'``, following MNE convention (:gh:`11719` by `Scott Huberty`_)
- Fix bug with :func:`mne.io.read_raw_fil` where datasets without sensor positions would not import (:gh:`11733` by `George O'Neill`_)
- Fix bug with :func:`mne.chpi.compute_chpi_snr` where cHPI being off for part of the recording or bad channels being defined led to an error or incorrect behavior (:gh:`11754`, :gh:`11755` by `Eric Larson`_)
- Allow int-like for the argument ``id`` of `~mne.make_fixed_length_events` (:gh:`11748` by `Mathieu Scheltienne`_)
- Fix bug where :func:`mne.io.read_raw_egi` did not properly set the EEG reference location for the reference channel itself (:gh:`11822` by `Eric Larson`_)
- Fix bug with legacy :meth:`~mne.io.Raw.plot_psd` method where passed axes were not used (:gh:`11778` by `Daniel McCloy`_)
- blink :class:`mne.Annotations` read in by :func:`mne.io.read_raw_eyelink` now begin with ``'BAD_'``, i.e. ``'BAD_blink'``, because ocular data are missing during blinks. (:gh:`11746` by `Scott Huberty`_)
- Fix bug where :ref:`mne show_fiff` could fail with an ambiguous error if the file is corrupt (:gh:`11800` by `Eric Larson`_)
- Fix bug where annotation FIF files lacked an end block tag (:gh:`11800` by `Eric Larson`_)
- Fix display of :class:`~mne.Annotations` in `mne.preprocessing.ICA.plot_sources` when the ``raw`` has ``raw.first_samp != 0`` and doesn't have a measurement date (:gh:`11766` by `Mathieu Scheltienne`_)
- Fix bug in read_raw_eyelink, where STATUS information of samples was always assumed to be in the file. Performance and memory improvements were also made. (:gh:`11823` by `Scott Huberty`_)
- Fix bug in convert_flash_mris, when flash30 and no flash5_reg.mgz file regenrated, add missing_ok=True as it is done in line 2074. (:gh:`11842` by `Hamza Abdelhedi`_)

API changes
~~~~~~~~~~~
- The ``baseline`` argument can now be array-like (e.g. ``list``, ``tuple``, ``np.ndarray``, ...) instead of only a ``tuple`` (:gh:`11713` by `Clemens Brunner`_)
- The ``events`` and ``event_id`` parameters of `:meth:`Epochs.plot() <mne.Epochs.plot>` now accept boolean values; see docstring for details (:gh:`11445` by `Daniel McCloy`_ and `Clemens Brunner`_)
- Deprecated ``gap_description`` keyword argument of :func:`mne.io.read_raw_eyelink`, which will be removed in mne version 1.6, in favor of using :meth:`mne.Annotations.rename` (:gh:`11719` by `Scott Huberty`_)
