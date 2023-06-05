.. _changes_1_4_1:

Version 1.4.1 (2023-06-05)
--------------------------

Bugs
~~~~

- Fix hanging interpreter with matplotlib figures using ``mne/viz/_mpl_figure.py`` in spyder console and jupyter notebooks`(:gh:`11696` by `Mathieu Scheltienne`_)

.. _changes_1_4_0:

Version 1.4.0 (2023-05-10)
--------------------------

Enhancements
~~~~~~~~~~~~
- Add functionality for reading CNT spans/annotations marked bad to :func:`mne.io.read_raw_cnt` (:gh:`11631` by :newcontrib:`Jacob Woessner`)
- Add ``:unit:`` Sphinx directive to enable use of uniform non-breaking spaces throughout the documentation (:gh:`11469` by :newcontrib:`Sawradip Saha`)
- Adjusted the algorithm used in :class:`mne.decoding.SSD` to support non-full rank data (:gh:`11458` by :newcontrib:`Thomas Binns`)
- Changed suggested type for ``ch_groups``` in `mne.viz.plot_sensors` from array to list of list(s) (arrays are still supported). (:gh:`11465` by `Hyonyoung Shin`_)
- Add support for UCL/FIL OPM data using :func:`mne.io.read_raw_fil` (:gh:`11366` by :newcontrib:`George O'Neill` and `Robert Seymour`_)
- Add harmonic field correction (HFC) for OPM sensors in :func:`mne.preprocessing.compute_proj_hfc` (:gh:`11536` by :newcontrib:`George O'Neill` and `Eric Larson`_)
- Forward argument ``axes`` from `mne.viz.plot_sensors` to `mne.channels.DigMontage.plot` (:gh:`11470` by :newcontrib:`Jan Ebert` and `Mathieu Scheltienne`_)
- Add forward IIR filtering, using parameters ``method='iir', phase='forward'`` (:gh:`11078` by :newcontrib:`Quentin Barthélemy`)
- Added ability to read stimulus durations from SNIRF files when using :func:`mne.io.read_raw_snirf` (:gh:`11397` by `Robert Luke`_)
- Add :meth:`mne.Info.save` to save an :class:`mne.Info` object to a fif file (:gh:`11401` by `Alex Rockhill`_)
- Improved error message when downloads are corrupted for :func:`mne.datasets.sample.data_path` and related functions (:gh:`11407` by `Eric Larson`_)
- Improved :func:`mne.get_config("") <mne.get_config>` behavior to return a dict of key-value pairs (config key and its description) rather than just a tuple of config keys (:gh:`11166` by `Clemens Brunner`_, `Daniel McCloy`_, and `Eric Larson`_)
- Add support for ``skip_by_annotation`` in :func:`mne.io.Raw.notch_filter` (:gh:`11388` by `Mainak Jas`_)
- Add support for ``output='complex'`` to :func:`mne.time_frequency.psd_array_welch` and when using ``method='welch'`` with :meth:`mne.Epochs.compute_psd` (:gh:`11556` by `Eric Larson`_)
- Slightly adjusted the window title for :func:`mne.Epochs.plot` (:gh:`11419` by `Richard Höchenberger`_ and `Daniel McCloy`_)
- Add :func:`mne.count_events` to count unique event types in a given event array (:gh:`11430` by `Clemens Brunner`_)
- Add a video to :ref:`tut-freesurfer-mne` of a brain inflating from the pial surface to aid in understanding the inflated brain (:gh:`11440` by `Alex Rockhill`_)
- Add automatic projection of sEEG contact onto the inflated surface for :meth:`mne.viz.Brain.add_sensors` (:gh:`11436` by `Alex Rockhill`_)
- Allow an image with intracranial electrode contacts (e.g. computed tomography) to be used without the freesurfer recon-all surfaces to locate contacts so that it doesn't have to be downsampled to freesurfer dimensions (for microelectrodes) and show an example :ref:`ex-ieeg-micro` with :func:`mne.transforms.apply_volume_registration_points` added to aid this transform (:gh:`11567` by `Alex Rockhill`_)
- Use new :meth:`dipy.align.imwarp.DiffeomorphicMap.transform_points` to transform a montage of intracranial contacts more efficiently (:gh:`11572` by `Alex Rockhill`_)
- Improve performance of raw data browsing with many annotations (:gh:`11614` by `Eric Larson`_)
- Add support for :func:`mne.preprocessing.maxwell_filter` with gradient-compensated CTF data, e.g., for tSSS-only mode (:gh:`10554` by `Eric Larson`_)
- Add support for eyetracking data using :func:`mne.io.read_raw_eyelink` (:gh:`11152` by `Dominik Welke`_ and `Scott Huberty`_)
- :func:`mne.channels.make_1020_channel_selections` gained a new parameter, ``return_ch_names``, to allow for easy retrieval of EEG channel names corresponding to the left, right, and midline portions of the montage (:gh:`11632` by `Richard Höchenberger`_)
- Methods for setting the sensor types of channels (e.g., for raw data, :meth:`mne.io.Raw.set_channel_types`) gained a new parameter, ``on_unit_change``, to control behavior (raise an exception, emit a warning, or do nothing) in case the measurement unit is adjusted automatically (:gh:`11668` by `Richard Höchenberger`_)
- :func:`mne.beamformer.trap_music` implements the TRAP-MUSIC localisation algorithm with the same signature as :func:`mne.beamformer.rap_music` (:gh:`11679` by `Théodore Papadopoulo`_)

Bugs
~~~~
- Improving compatibility with Qt6 by removing the use of deprecated ``AA_UseHighDpiPixmaps`` attribute for this Qt version (:gh:`11662` by :newcontrib:`Florin Pop`)
- Fix :func:`mne.time_frequency.psd_array_multitaper` docstring where argument ``bandwidth`` incorrectly reported argument as half-bandwidth and gave wrong explanation of default value (:gh:`11479` by :newcontrib:`Tom Stone`)
- Fix bug where installation of a package depending on ``mne`` will error when done in an environment where ``setuptools`` is not present (:gh:`11454` by :newcontrib:`Arne Pelzer`)
- Fix bug in :meth:`mne.Annotations.rename` where replacements were not done correctly (:gh:`11666` by :newcontrib:`Timur Sokhin` and `Eric Larson`_)
- Fix :meth:`mne.time_frequency.Spectrum.to_data_frame`'s docstring to reflect the correct name for the appended frequencies column (:gh:`11457` by :newcontrib:`Zvi Baratz`)
- Fix bug with :func:`mne.io.read_raw_egi` where events outside the recording window could break data reading (:gh:`11505` by :newcontrib:`Niels Focke`)
- Fix bug where :func:`mne.preprocessing.regress_artifact` and :class:`mne.preprocessing.EOGRegression` incorrectly tracked ``picks`` (:gh:`11366` by `Eric Larson`_)
- Fix bug where channel names were not properly sanitized in :func:`mne.write_evokeds` and related functions (:gh:`11399` by `Eric Larson`_)
- Fix bug where splash screen would not always disappear (:gh:`11398` by `Eric Larson`_)
- Fix bug where having a different combination of volumes loaded into ``freeview`` caused different affines to be returned by :func:`mne.read_lta` for the same Linear Transform Array (LTA) (:gh:`11402` by `Alex Rockhill`_)
- Fix how :class:`mne.channels.DigMontage` is set when using ``mne.gui.locate_ieeg`` so that :func:`mne.Info.get_montage` works and does not return ``None`` (:gh:`11421` by `Alex Rockhill`_)
- Fix :func:`mne.io.read_raw_edf` when reading EDF data with different sampling rates and a mix of data channels when using ``infer_types=True`` (:gh:`11427` by `Alex Gramfort`_)
- Fix how :class:`mne.channels.DigMontage` is set when using :func:`mne.preprocessing.ieeg.project_sensors_onto_brain` so that :func:`mne.Info.get_montage` works and does not return ``None`` (:gh:`11436` by `Alex Rockhill`_)
- Fix configuration folder discovery on Windows, which would fail in certain edge cases; and produce a helpful error message if discovery still fails (:gh:`11441` by `Richard Höchenberger`_)
- Make :class:`~mne.decoding.SlidingEstimator` and :class:`~mne.decoding.GeneralizingEstimator` respect the ``verbose`` argument. Now with ``verbose=False``, the progress bar is not shown during fitting, scoring, etc. (:gh:`11450` by `Mikołaj Magnuski`_)
- Fix bug with ``mne.gui.locate_ieeg`` where Freesurfer ``?h.pial.T1`` was not recognized and suppress excess logging (:gh:`11489` by `Alex Rockhill`_)
- All functions accepting paths can now correctly handle :class:`~pathlib.Path` as input. Historically, we expected strings (instead of "proper" path objects), and only added :class:`~pathlib.Path` support in a few select places, leading to inconsistent behavior. (:gh:`11473` and :gh:`11499` by `Mathieu Scheltienne`_)
- Fix visualization dialog compatibility with matplotlib 3.7 (:gh:`11409` by `Daniel McCloy`_ and `Eric Larson`_)
- Expand tilde (user directory) in config keys (:gh:`11537` by `Clemens Brunner`_)
- Fix bug in :func:`mne.preprocessing.compute_maxwell_basis` where using ``int_order=0`` would raise an error (:gh:`11562` by `Eric Larson`_)
- In the legacy function :func:`mne.viz.plot_epochs_psd_topomap`, the parameter ``names`` now works again (:gh:`11563` by `Daniel McCloy`_)
- Fix :func:`mne.io.read_raw` for file names containing multiple dots (:gh:`11521` by `Clemens Brunner`_)
- Fix bug in :func:`mne.export.export_raw` when exporting to EDF with a physical range set smaller than the data range (:gh:`11569` by `Mathieu Scheltienne`_)
- Fix bug in :func:`mne.concatenate_raws` where two raws could not be merged if the order of the bad channel lists did not match (:gh:`11502` by `Moritz Gerster`_)
- Fix bug where :meth:`mne.Evoked.plot_topomap` opened an extra figure (:gh:`11607` by `Alex Rockhill`_)
- Fix bug where :func:`mne.transforms.apply_volume_registration_points` modified info in place (:gh:`11612` by `Alex Rockhill`_)
- Fix bug where Maxwell-filtered data rank was not handled properly in :func:`mne.beamformer.make_lcmv` (:gh:`11664` by `Eric Larson`_)
- In :class:`~mne.Report`, custom figures now show up correctly when ``image_format='svg'`` is requested (:gh:`11623` by `Richard Höchenberger`_)
- Fix bug where providing ``axes`` in `mne.preprocessing.ICA.plot_components` would fail (:gh:`11654` by `Mathieu Scheltienne`_)
- Fix deprecation of ``title``, ``vmin`` and ``vmax`` in `~mne.viz.plot_tfr_topomap` called from interactive `mne.time_frequency.AverageTFR.plot_topo` (:gh:`11683` by `Mathieu Scheltienne`_)

API changes
~~~~~~~~~~~
- Deprecate ``mne.gui.locate_ieeg`` in favor of :func:`mne-gui-addons:mne_gui_addons.locate_ieeg` (:gh:`11616` by `Alex Rockhill`_)
- Deprecate arguments ``kind`` and ``path`` from :func:`mne.channels.read_layout` in favor of a common argument ``fname`` (:gh:`11500` by `Mathieu Scheltienne`_)
- Change ``aligned_ct`` positional argument in ``mne.gui.locate_ieeg`` to ``base_image`` to reflect that this can now be used with unaligned images (:gh:`11567` by `Alex Rockhill`_)
- ``mne.warp_montage_volume`` was deprecated in favor of :func:`mne.preprocessing.ieeg.warp_montage` (acts directly on points instead of using an intermediate volume) and :func:`mne.preprocessing.ieeg.make_montage_volume` (which makes a volume of ieeg contact locations which can still be useful) (:gh:`11572` by `Alex Rockhill`_)
- Deprecate ``mne.pick_channels_evoked`` in favor of ``evoked.copy().pick(...)`` (:gh:`11665` by `Eric Larson`_)
- Set instance methods ``inst.pick_types`` and ``inst.pick_channels`` as legacy in favor of ``inst.pick(...)`` (:gh:`11665` by `Eric Larson`_)
- The default of ``inst.pick_channels(..., ordered=False)`` will change to ``ordered=True`` in 1.5 to avoid silent bugs (:gh:`11665` by `Eric Larson`_)

Authors
~~~~~~~

* Alex Rockhill
* Alexandre Gramfort
* Arne Pelzer+
* Britta Westner
* Carlos de la Torre
* Clemens Brunner
* Daniel Hasegan
* Daniel McCloy
* Denis A. Engemann
* Dimitri Papadopoulos Orfanos
* Dinara Issagaliyeva
* Dominik Welke
* Enzo Altamiranda
* Eric Brayet
* Eric Larson
* Erkka Heinila
* Felix Klotzsche
* Florin Pop+
* George O'Neill+
* Giulio
* Hakimeh Aslsardroud
* Hyonyoung Shin+
* Jacob Woessner+
* Jan Ebert+
* Jennifer Behnke
* Jevri Hanna
* Lukas Hecker
* Mainak Jas
* Mark Alexander Henney
* Mathieu Scheltienne
* Mauricio Cespedes
* Michiru Kaneda
* Mikołaj Magnuski
* Moritz Gerster
* Niels Focke+
* Omer Shubi
* Pavel Navratil
* Quentin Barthélemy+
* Richard Höchenberger
* Rob Luke
* Robert Seymour+
* Santeri Ruuskanen
* Sawradip Saha+
* Scott Huberty
* Stefan Appelhoff
* Theodore Papadopoulo
* Thomas Samuel Binns+
* Timon Merk
* Timur Sokhin+
* Tom Ma
* Tom Stone+
* Toomas Erik Anijärv
* Zvi Baratz+
