.. NOTE: we are now using links to highlight new functions and classes.
   Please follow the examples below like :func:`mne.stats.f_mway_rm`, so the
   whats_new page will have a link to the function/class documentation.

.. NOTE: there are 3 separate sections for changes, based on type:
   - "Enhancements" for new features
   - "Bugs" for bug fixes
   - "API changes" for backward-incompatible changes

.. NOTE: changes from first-time contributors should be added to the TOP of
   the relevant section (Enhancements / Bugs / API changes), and should look
   like this (where xxxx is the pull request number):
   - description of enhancement/bugfix/API change (:gh:`xxxx` **by new contributor** |Firstname Lastname|_)

.. _current:

Current (0.24.dev0)
-------------------

.. |New Contributor| replace:: **New Contributor**

.. |David Julien| replace:: **David Julien**

.. |Romain Derollepot| replace:: **Romain Derollepot**

.. |Jan Sosulski| replace:: **Jan Sosulski**

.. |Xiaokai Xia| replace:: **Xiaokai Xia**

.. |Marian Dovgialo| replace:: **Marian Dovgialo**

.. |Pierre-Antoine Bannier| replace:: **Pierre-Antoine Bannier**

Enhancements
~~~~~~~~~~~~
.. - Add something cool (:gh:`9192` **by new contributor** |New Contributor|_)

- Add support for NIRSport and NIRSport2 devices to `mne.io.read_raw_nirx` (:gh:`9348` and :gh:`9401` **by new contributor** |David Julien|_, **new contributor** |Romain Derollepot|_, `Robert Luke`_, and `Eric Larson`_)

- New function :func:`mne.label.find_pos_in_annot` to get atlas label for MRI coordinates.  (:gh:`9376` by **by new contributor** |Marian Dovgialo|_)

- Add support for SURE parameter selection in :func:`mne.inverse_sparse.mixed_norm` and make ``alpha`` parameter now default to ``'sure'`` (:gh:`9430` by **new contributor** |Pierre-Antoine Bannier|_ and `Alex Gramfort`_)

- Speed up BCD solver in :func:`mne.inverse_sparse.mixed_norm` by adding Anderson acceleration (:gh:`9481` by **new contributor** |Pierre-Antoine Bannier|_ and `Alex Gramfort`_)

- New function :func:`mne.chpi.get_chpi_info` to retrieve basic information about the cHPI system used when recording MEG data (:gh:`9369` by `Richard Höchenberger`_)

- New namespace `mne.export` created to contain functions (such as `mne.export.export_raw` and `mne.export.export_epochs`) for exporting data to non-FIF formats (:gh:`9427` by `Eric Larson`_)

- Add support for Hitachi fNIRS devices in `mne.io.read_raw_hitachi` (:gh:`9391` by `Eric Larson`_)

- Add support for ``picks`` in :func:`mne.stc_near_sensors` (:gh:`9396` by `Eric Larson`_)

- Add projections when printing a :class:`mne.Info` in the notebook (:gh:`9403` by `Alex Gramfort`_)

- Add new function :func:`mne.bem.make_scalp_surfaces` to compute scalp surfaces for coregistration (:gh:`9463` by `Alex Gramfort`_)

- Add support for interpolating oxy and deoxyhaemoglobin data types (:gh:`9431` by `Robert Luke`_)

- Add support for SNIRF files in :class:`mne.Report` (:gh:`9443` by `Robert Luke`_)

- Add support for exporting MFF evoked files using `mne.export.export_evokeds` and `mne.export.export_evokeds_mff` (:gh:`9406` by `Evan Hathaway`_)

- :func:`mne.concatenate_raws`, :func:`mne.concatenate_epochs`, and :func:`mne.write_evokeds` gained a new parameter ``on_mismatch``, which controls behavior in case not all of the supplied instances share the same device-to-head transformation (:gh:`9438` by `Richard Höchenberger`_)

- Add support for multiple datablocks (acquistions with pauses) in :func:`mne.io.read_raw_nihon` (:gh:`9437` by `Federico Raimondo`_)

- Add new function :func:`mne.preprocessing.annotate_break` to automatically detect and mark "break" periods without any marked experimental events in the continuous data (:gh:`9445` by `Richard Höchenberger`_)

- Add "time_format" to :meth:`mne.io.Raw.plot` and :meth:`mne.preprocessing.ICA.plot_sources` to display acquisition time on x-axis (:gh:`9419` by `Martin Schulz`_)

- Speed up :meth:`mne.decoding.TimeDelayingRidge.predict` by switching to FFT-based convolution (:gh:`9458` by `Ross Maddox`_)

- Show multiple colors and linestyles for excluded components with :class:`mne.Evoked` in :meth:`mne.preprocessing.ICA.plot_sources` (:gh:`9444` by `Martin Schulz`_)

- Add functions for aligning MRI and CT data `mne.transforms.compute_volume_registration` and `mne.transforms.apply_volume_registration` (:gh:`9503` by `Alex Rockhill`_ and `Eric Larson`_)

- Add tutorial for how to processes image (CT and MR) files in order to localize electrode contacts for intracranial recordings :ref:`tut-ieeg-localize` (:gh`9484` by `Alex Rockhill`_)

- Add support for colormap normalization in :func:`mne.viz.plot_topomap` (:gh:`9468` by `Clemens Brunner`_)

- Update :func:`mne.preprocessing.realign_raw` with Numpy-recommended polynomial fitting method (:gh:`9514` by `Erica Peterson`_)

Bugs
~~~~
- Fix bug with :meth:`mne.Epochs.crop` and :meth:`mne.Evoked.crop` when ``include_tmax=False``, where the last sample was always cut off, even when ``tmax > epo.times[-1]`` (:gh:`9378` **by new contributor** |Jan Sosulski|_)

- Fix bug with `mne.io.read_raw_curry` to allow reading Curry 8 event files with '.cdt.ceo' extension (:gh:`9381` by **new contributor** |Xiaokai Xia|_ and `Daniel McCloy`_)

- Fix bug with :func:`mne.io.read_raw_nihon` where latin-1 annotations could not be read (:gh:`9384` by `Alex Gramfort`_)

- Fix bug when printing a :class:`mne.io.RawArray` in the notebook (:gh:`9404` by `Alex Gramfort`_)

- Fix bug when computing rank from info for SSS data with only gradiometers or magnetometers (:gh:`9435` by `Alex Gramfort`_)

- Fix bug with `mne.io.Raw.set_montage` and related functions where the channel coordinate frame was not properly set to head (:gh:`9447` by `Eric Larson`_)

- Fix bug with `mne.io.read_raw_fieldtrip` and `mne.read_epochs_fieldtrip` where channel positions were not set properly (:gh:`9447` by `Eric Larson`_)

- Fix bug with `mne.preprocessing.nirs.optical_density` where protection against zero values was not guaranteed (:gh:`9522` by `Eric Larson`_)

- :func:`mne.concatenate_raws` now raises an exception if ``raw.info['dev_head_t']`` differs between files. This behavior can be controlled using the new ``on_mismatch`` parameter (:gh:`9438` by `Richard Höchenberger`_)

- Fixed bug in :meth:`mne.Epochs.drop_bad` where subsequent rejections failed if they only specified thresholds for a subset of the channel types used in a previous rejection (:gh:`9485` by `Richard Höchenberger`_).

- Fix bug with `mne.simulation.simulate_evoked`, `mne.apply_forward`, and `mne.apply_forward_raw` where systems with EEG channels that come before MEG channels would have them mixed up in the output evoked or raw object (:gh:`#9513` by `Eric Larson`_)

- In :func:`mne.viz.plot_ica_scores` and :meth:`mne.preprocessing.ICA.plot_scores`, the figure and axis titles no longer overlap when plotting only a single EOG or ECG channel (:gh:`9489` by `Richard Höchenberger`_).

- Ensure `mne.io.Raw.get_montage` works with SNIRF data (:gh:`9524` by `Robert Luke`_)

- Fix bug in :func:`mne.viz.plot_topomap` (and related methods like :meth:`mne.Evoked.plot_topomap`) where large distances between electrodes (higher than head radius) would lead to an error (:gh:`9528` by `Mikołaj Magnuski`_).

API changes
~~~~~~~~~~~
- In `mne.compute_source_morph`, the ``niter_affine`` and ``niter_sdr`` parameters have been replaced by ``niter`` and ``pipeline`` parameters for more consistent and finer-grained control of registration/warping steps and iteration (:gh:`9505` by `Alex Rockhill`_ and `Eric Larson`_)
