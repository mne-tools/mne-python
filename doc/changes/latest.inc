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

Current (1.2.dev0)
------------------

Enhancements
~~~~~~~~~~~~
- Add ``vmin`` and ``vmax`` parameters to :meth:`mne.Evoked.animate_topomap` (:gh:`11073` by :newcontrib:`Mats van Es`)
- Enable common average reference projection for ECoG, sEEG, and DBS data (optionally jointly with ``joint=True``) in :meth:`~mne.io.Raw.set_eeg_reference` and related functions/methods (:gh:`10656` by :newcontrib:`Moritz Gerster` and `Eric Larson`_)
- Add support for computing forward solutions using OpenMEEG_ (:gh:`11011` by `Alex Gramfort`_ and `Eric Larson`_)
- EEGLAB files (saved as MAT versions less than v7.3) can now be imported with :func:`mne.io.read_raw_eeglab` without the optional dependency ``pymatreader`` (:gh:`11006` by `Clemens Brunner`_)
- Add eight source estimate comparison metrics to new submodule :mod:`mne.simulation.metrics` (:gh:`6233` by `Kostiantyn Maksymenko`_ and `Alex Gramfort`_)
- Add :func:`mne.time_frequency.csd_tfr` to compute cross-spectral density from :class:`mne.time_frequency.EpochsTFR` (:gh:`10986` by `Alex Rockhill`_)
- Improve ``repr()`` for :class:`mne.minimum_norm.InverseOperator` when loose orientation is used (:gh:`11048` by `Eric Larson`_)
- Add support for vector and volume source estimates in :func:`mne.minimum_norm.get_point_spread` and :func:`mne.minimum_norm.get_cross_talk` (:gh:`8639`, :gh:`11083` by `Olaf Hauk`_, `Marijn van Vliet`_, `Alex Gramfort`_, and `Eric Larson`_)
- :meth:`mne.Epochs.plot_psd_topomap` now suppresses redundant colorbars when ``vlim='joint'`` (:gh:`11051` by `Daniel McCloy`_)
- Add ``starting_affine`` keyword argument to :func:`mne.transforms.compute_volume_registration` to initialize an alignment with an affine (:gh:`11020` by `Alex Rockhill`_)
- The ``trans`` parameter in :func:`mne.make_field_map` now accepts a :class:`~pathlib.Path` object, and uses standardised loading logic (:gh:`10784` by :newcontrib:`Andrew Quinn`)
- Add HTML representation for `~mne.Evoked` in Jupyter Notebooks (:gh:`11075` by `Valerii Chirkov`_ and `Andrew Quinn`_)
- Add support for ``temperature`` and ``gsr`` (galvanic skin response, i.e., electrodermal activity) channel types (:gh:`11090`, :gh:`11108` by `Eric Larson`_ and `Richard Höchenberger`_)
- Allow :func:`mne.beamformer.make_dics` to take ``pick_ori='vector'`` to compute vector source estimates (:gh:`19080` by `Alex Rockhill`_)
- Add ``units`` parameter to :func:`mne.io.read_raw_edf` in case units are missing from the file (:gh:`11099` by `Alex Gramfort`_)
- Add ``on_missing`` functionality to all of our classes that have a ``drop_channels`` method, to control what happens when channel names are not in the object (:gh:`11077` by `Andrew Quinn`_)
- Improve interpolation of bridged electrodes with `~mne.preprocessing.interpolate_bridged_electrodes` (:gh:`11094` by `Mathieu Scheltienne`_)
- Add :func:`mne.minimum_norm.apply_inverse_tfr_epochs` to apply inverse methods to time-frequency resolved epochs (:gh:`11095` by `Alex Rockhill`_)
- Add support for multiple probes via multiple CSV files passed to :func:`mne.io.read_raw_hitachi` (:gh:`11186` by `Eric Larson`_)
- Add :func:`mne.chpi.get_active_chpi` to retrieve the number of active hpi coils for each time point (:gh:`11122` by `Eduard Ort`_)
- Add example of how to obtain time-frequency decomposition using narrow bandpass Hilbert transforms to :ref:`ex-tfr-comparison` (:gh:`11116` by `Alex Rockhill`_)
- Add ``==`` and ``!=`` comparison between `mne.Projection` objects (:gh:`11147` by `Mathieu Scheltienne`_)
- Parse automatically temperature channel with :func:`mne.io.read_raw_edf` (:gh:`11150` by `Eric Larson`_ and `Alex Gramfort`_)
- Add ``unscale`` option to :func:`mne.head_to_mri` to facilitate working with surrogate MRI data (:gh:`11185` by `Eric Larson`_)
- Add ``encoding`` parameter to :func:`mne.io.read_raw_edf` and :func:`mne.io.read_raw_bdf` to support custom (non-UTF8) annotation channel encodings (:gh:`11154` by `Clemens Brunner`_)
- :class:`mne.preprocessing.ICA` gained a new method, :meth:`~mne.preprocessing.ICA.get_explained_variance_ratio`, that allows the retrieval of the proportion of variance explained by ICA components (:gh:`11141` by `Richard Höchenberger`_)
- Add config option ``MNE_REPR_HTML`` to disable HTML repr in notebook environments (:gh:`11159` by `Clemens Brunner`_)

Bugs
~~~~
- Fix bug in :func:`mne.io.read_raw_eeglab` where unlabeled fiducials causde reading errors (:gh:`11074` by :newcontrib:`Sebastiaan Mathot`)
- Fix bug in :func:`mne.time_frequency.read_csd` that returned ``projs`` as a list of dict instead of :class:`mne.Projection` (:gh:`11072` by :newcontrib:`Chetan Gohil`)
- Fix bug in :func:`mne.decoding.TimeFrequency` that prevented cloning if constructor arguments were modified (:gh:`11004` by :newcontrib:`Daniel Carlström Schad`)
- Fix bug in ``repr`` of decoding estimator classes (e.g., :class:`mne.decoding.SlidingEstimator`) where the parameters where not stored in the result string but instead just printed to ``sys.stdout`` (:gh:`11190` by :newcontrib:`Dominik Wetzel`_)
- Fix bug in :class:`mne.viz.Brain` constructor where the first argument was named ``subject_id`` instead of ``subject`` (:gh:`11049` by `Eric Larson`_)
- Fix bug in :ref:`mne coreg` where the MEG helmet position was not updated during ICP fitting (:gh:`11084` by `Eric Larson`_)
- Fix bug in :func:`~mne.io.read_raw_curry`, and :func:`~mne.io.read_raw_cnt` where digitization points were not read properly (:gh:`11145` by `Eric Larson`_)
- Fix bug in :func:`mne.minimum_norm.apply_inverse_epochs` where the average EEG projector was not checked properly (:gh:`11182` by `Eric Larson`_)
- Document ``height`` and ``weight`` keys of  ``subject_info`` entry in :class:`mne.Info` (:gh:`11019` by :newcontrib:`Sena Er`)
- Fix bug in :func:`mne.viz.plot_filter` when plotting filters created using ``output='ba'`` mode with ``compensation`` turned on. (:gh:`11040` by `Marian Dovgialo`_)
- Fix bugs in documentation of surface :class:`~mne.SourceSpaces` (:gh:`11171` by `Eric Larson`_)
- Fix bug in :func:`mne.io.read_raw_bti` where EEG, EMG, and H/VEOG channels were not detected properly, and many non-ECG channels were called ECG. The logic has been improved, and any channels of unknown type are now labeled as ``misc`` (:gh:`11102` by `Eric Larson`_)
- Fix bug in :func:`mne.viz.plot_topomap` when providing ``sphere="eeglab"`` (:gh:`11081` by `Mathieu Scheltienne`_)
- Fix bug in :meth:`mne.Dipole.to_mri` where MRI RAS rather than MRI surface RAS was returned (:gh:`11185` by `Eric Larson`_)
- The string and HTML representation of :class:`mne.preprocessing.ICA` reported incorrect values for the explained variance. This information has been removed from the representations, and should instead be retrieved via the new :meth:`~mne.preprocessing.ICA.get_explained_variance_ratio` method (:gh:`11141` by `Richard Höchenberger`_)
- Fix bug in :meth:`mne.Evoked.plot` and related methods where a ``np.nan`` location value in any channel causes spatial colours to fail (:gh:`6870` by `Simeon Wong`_)

API changes
~~~~~~~~~~~
- Starting with this release we now follow the Python convention of using ``FutureWarning`` instead of ``DeprecationWarning`` to signal user-facing changes to our API (:gh:`11120` by `Daniel McCloy`_)
- The ``bands`` parameter of :meth:`mne.Epochs.plot_psd_topomap` now accepts :class:`dict` input; legacy :class:`tuple` input is supported, but discouraged for new code (:gh:`11050` by `Daniel McCloy`_)
- The :func:`mne.head_to_mri` new function parameter ``kind`` default will change from ``'ras'`` to ``'mri'`` (:gh:`11185` by `Eric Larson`_)
- The ``show_toolbar`` argument to :class:`mne.viz.Brain` is being removed by deprecation (:gh:`11049` by `Eric Larson`_)
- New classes :class:`~mne.time_frequency.Spectrum` and :class:`~mne.time_frequency.EpochsSpectrum`, created via new methods :meth:`Raw.compute_psd()<mne.io.Raw.compute_psd>`, :meth:`Epochs.compute_psd()<mne.Epochs.compute_psd>`, and :meth:`Evoked.compute_psd()<mne.Evoked.compute_psd>` (:gh:`10184` by `Daniel McCloy`_)
- The ``mne.epochs.add_channels_epochs`` function has been deprecated in favor of :meth:`epochs.add_channels <mne.Epochs.add_channels>` (:gh:`11180` by `Eric Larson`_)
- The PSD functions that operate on Raw/Epochs/Evoked instances (``mne.time_frequency.psd_welch`` and ``mne.time_frequency.psd_multitaper``) are deprecated; for equivalent functionality create :class:`~mne.time_frequency.Spectrum` or :class:`~mne.time_frequency.EpochsSpectrum` objects instead and then run ``spectrum.get_data(return_freqs=True)`` (:gh:`10184` by `Daniel McCloy`_)
