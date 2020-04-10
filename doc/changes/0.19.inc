.. NOTE: we are now using links to highlight new functions and classes.
   Please follow the examples below like :func:`mne.stats.f_mway_rm`, so the
   whats_new page will have a link to the function/class documentation.

.. NOTE: there are 3 separate sections for changes, based on type:
   - "Changelog" for new features
   - "Bug" for bug fixes
   - "API" for backward-incompatible changes

.. _changes_0_19:

Version 0.19
------------

Changelog
~~~~~~~~~

- Add :func:`mne.cuda.set_cuda_device` and config variable ``MNE_CUDA_DEVICE`` to select among multiple GPUs (by numeric device ID) by `Daniel McCloy`_.

- Add :func:`mne.channels.make_standard_montage` to create :class:`mne.channels.DigMontage` from templates by `Joan Massich`_ and `Alex Gramfort`_.

- Add :func:`mne.channels.compute_dev_head_t` to compute Device-to-Head transformation from a montage by `Joan Massich`_ and `Alex Gramfort`_.

- Add :func:`mne.channels.read_dig_fif` to read digitization coordinates from ``.fif`` files by `Joan Massich`_ and `Alex Gramfort`_.

- Add :func:`mne.channels.read_dig_egi` to read digitization coordinates from EGI ``.xml`` files by `Joan Massich`_ and `Alex Gramfort`_.

- Add :func:`mne.channels.read_dig_polhemus_isotrak` and :func:`mne.channels.read_polhemus_fastscan` to read Polhemus data by `Joan Massich`_

- Add :func:`mne.channels.read_dig_captrack` to read BrainVision CapTrak (BVCT) digitization coordinate files by `Stefan Appelhoff`_ and `Joan Massich`_

- Add :func:`mne.channels.make_dig_montage` to create :class:`mne.channels.DigMontage` objects out of np.arrays by `Joan Massich`_

- Add :func:`mne.channels.read_custom_montage` to read various EEG electrode locations files by `Joan Massich`_ and `Alex Gramfort`_.

- Add support for making epochs with duplicated events, by allowing three policies: "error" (default), "drop", or "merge" in :class:`mne.Epochs` by `Stefan Appelhoff`_

- Allow :meth:`mne.Annotations.crop` to support negative ``tmin`` and ``tmax`` by `Joan Massich`_

- Unknown events code in GDF are now visible in the ``event_id`` by `Theodore Papadopoulo`_

- Now :func:`mne.io.read_raw_ctf` populates ``raw.annotations`` with the markers in ``MarkerFile.mrk`` if any by `Joan Massich`_

- Add options for controlling the use of the ``-T1`` flag and the location of the brainmask output in :ref:`mne watershed_bem <gen_mne_watershed_bem>` by `Eric Larson`_

- Add support to :func:`mne.read_annotations` to read CTF marker files by `Joan Massich`_

- Do not convert effective number of averages (``nave`` attribute of :class:`mne.Evoked`) to integer except when saving to FIFF file by `Daniel McCloy`_.

- Add automatic fiducial position estimation in :ref:`mne coreg <gen_mne_coreg>` using MNI Talairach fiducial locations in :func:`mne.coreg.get_mni_fiducials` by `Jon Houck`_ and `Eric Larson`_

- Add support for :ref:`mne coreg <gen_mne_coreg>` scaling surrogate subjects without surface reconstructions, such as those created for volumetric analyses only (e.g., with ``recon-all -autorecon1``) by `Eric Larson`_

- Add reader for Curry data in :func:`mne.io.read_raw_curry` by `Dirk Gütlin`_

- Butterfly channel plots now possible for :meth:`mne.Epochs.plot_psd` with ``average=False``. Infrastructure for this function now shared with analogous Raw function, found in ``mne.viz.utils`` by `Jeff Hanna` _

- Add option not to orthogonalize power envelopes with ``orthogonalize=False`` in :func:`mne.connectivity.envelope_correlation` by `Denis Engemann`_

- Accept filenames of raw .fif files that end in ``_meg.fif`` to enable complicance with the Brain Imaging Data Structure by `Stefan Appelhoff`_

- Add function to check the type of a FIF file using :func:`mne.what` and `mne what <gen_mne_what>` by `Eric Larson`_

- Add support for specifying the initial time and/or position and providing a :class:`mne.SourceMorph` instead of :class:`mne.SourceSpaces` in :func:`mne.viz.plot_volume_source_estimates` by `Eric Larson`_

- Speed up morph map generation in :func:`mne.read_morph_map` by ~5-10x by using :func:`numba.jit` by `Eric Larson`_

- Speed up :func:`mne.setup_volume_source_space`, especially when ``volume_label is not None`` by `Eric Larson`_

- Speed up :ref:`mne coreg <gen_mne_coreg>` interactive and automated (ICP) alignment by using nearest-neighbor calculations in the MRI coordinate frame, by `Eric Larson`_

- Add :func:`mne.dig_mri_distances` to compute the distances between digitized head points and the MRI head surface by `Alex Gramfort`_ and `Eric Larson`_

- Add scale bars for data channels in :func:`mne.io.Raw.plot` by `Eric Larson`_

- Add :func:`mne.viz.plot_brain_colorbar` to plot a colorbar appropriately matched to a :func:`mne.viz.plot_source_estimates` plot by `Eric Larson`_

- Add support for showing head surface (to visualize digitization fit) while showing a single-layer BEM to :func:`mne.viz.plot_alignment` by `Eric Larson`_

- Add option ``include_tmax=True`` to cropping methods :meth:`mne.io.Raw.crop`, :meth:`mne.Epochs.crop`, :meth:`mne.Evoked.crop`, :meth:`mne.SourceEstimate.crop`, :meth:`mne.Dipole.crop`, and :meth:`mne.time_frequency.AverageTFR.crop` by `Eric Larson`_

- Change the behavior of :meth:`mne.io.Raw.plot` for ``scalings='auto'`` and ``remove_dc=True`` to compute the scalings on the data with DC removed by `Clemens Brunner`_

- Allow creating annotations within existing annotations in :func:`mne.io.Raw.plot` by default (the old snapping behavior can be toggled by pressing 'p') by `Clemens Brunner`_

- Add plotting of rank estimates in :func:`mne.viz.plot_cov` by `Eric Larson`_

- Add a new :func:`mne.viz.plot_sensors_connectivity` function to visualize the sensor connectivity in 3D by `Guillaume Favelier`_ and `Alex Gramfort`_

- Add support for ``info['utc_offset']``, ``info['device_info']``, and ``info['helium_info']`` components of :class:`mne.Info` by `Eric Larson`_

- Add control over dipole colors in :func:`mne.viz.plot_dipole_locations` when using orthoview mode by `Eric Larson`_

- Use second-order-sections filtering in :meth:`mne.io.Raw.plot` and :ref:`mne browse_raw <gen_mne_browse_raw>` by `Eric Larson`_

- Add re-referencing functionality for ecog and seeg channel types in :func:`mne.set_eeg_reference` by `Keith Doelling`_

- Add support for median averaging and for returning unaggregated segments in :func:`mne.time_frequency.psd_welch` by `Richard Höchenberger`_

- :func:`io.read_raw_kit`: add support for NYU New York 2019 system update, by `Christian Brodbeck`_


Bug
~~~

- Fix setting montage eeg ref position for captrack by `Joan Massich`_

- Fix saving raw read from BDF file using ``tmin`` and ``tmax`` using ``preload=False`` by `Alex Gramfort`_

- Fix :func:`mne.grand_average` to use equal sum-to-one weights (like it used to, before changes to underlying :func:`mne.combine_evoked`) by `Daniel McCloy`_

- Fix :meth:`mne.io.Raw.filter` to deal with instances with no data channels properly by `Eric Larson`_

- Fix one-sample baseline issue in :class:`mne.BaseEpochs` when using `tmin=0` by `Milan Rybář`_

- Fix bug in :func:`mne.viz.plot_volume_source_estimates` where ``'glass_brain'`` MRIs were not transformed to MNI space, by `Eric Larson`_

- Fix bug in :func:`mne.viz.plot_volume_source_estimates` where MRIs with voxels not in RAS orientation could not be browsed properly, by `Eric Larson`_

- Fix bug in :meth:`mne.SourceMorph.apply` where output STCs had ``stc.vertices`` defined improperly, by `Eric Larson`_

- Fix bug in :meth:`mne.SourceMorph.apply` where the default was errantly ``mri_space=False`` instead of ``mri_space=None`` (as documented), by `Eric Larson`_

- Fix :meth:`mne.VolVectorSourceEstimate.normal` and :func:`mne.minimum_norm.apply_inverse` to only allow normal extraction/orientation for surface and discrete source spaces by `Eric Larson`_

- Fix :meth:`mne.io.Raw.set_annotations` for ``meas_date`` previous to 1970 by `Joan Massich`_

- Fix horizontal spacing issues in :meth:`mne.io.Raw.plot_psd` by `Jeff Hanna`_

- Fix reading of dates in BrainVision files if no "New Segment" marker is specified, no date is given, or data is missing, by `Stefan Appelhoff`_

- Fix bug with reading one-channel GDF files by `Abram Hindle`_

- Fix bug with y-axis labeling in :meth:`mne.io.Raw.plot_psd` by `Eric Larson`_

- Fix side-effect where :func:`mne.viz.plot_ica_sources` and :meth:`mne.preprocessing.ICA.plot_sources` changed the ``ICA.exclude`` attribute even when users didn't interact with the plot by `Daniel McCloy`_.

- Fix scaling of sources in :meth:`ica.plot_sources(epochs) <mne.preprocessing.ICA.plot_sources>` by `Eric Larson`_

- Fix wrong assumptions about units in BrainVision montages and add test asserting units in "mm" or "auto", by `Stefan Appelhoff`_

- Fix scaling issue with signals in mV in EDF files read with :func:`mne.io.read_raw_edf` by `Alex Gramfort`_

- Fix :func:`mne.extract_label_time_course` behavior when using STCs generated with ``apply_inverse(..., label=label)`` or ``stc.in_label`` by `Eric Larson`_

- Fix bug in :func:`mne.io.read_raw_brainvision` so that recording date timestamps are also recognized if channel reference data is negative, by `Stefan Appelhoff`_

- Fix order of ``info['dig']`` that was alphabetical based on channel names and not following the channel order when using :meth:`mne.io.Raw.set_montage` and a ``mne.channels.Montage`` object by `Joan Massich`_ and `Alex Gramfort`_.

- Fix reading CNT files larger than 2Gb by `Joan Massich`_

- Fix reading of 4D/BTi data with different channel names and onsets in :func:`mne.io.read_raw_bti` by `Joshua Bear`_ and `Eberhard Eich`_

- Fix formula for effective number of averages in :func:`mne.combine_evoked` when ``weights='equal'`` by `Daniel McCloy`_.

- Fix bug in :func:`mne.simulation.simulate_stc` to avoid empty stc if label vertices and source space do not intersect, by `Kostiantyn Maksymenko`_

- Fix ``event_id='auto'`` in :func:`mne.events_from_annotations` to recover Brainvision markers after saving it in ``.fif`` by `Joan Massich`_

- Fix :func:`mne.read_epochs_eeglab` when epochs are stored as float. By `Thomas Radman`_

- Fix :func:`mne.Evoked.resample` and :func:`mne.Epochs.resample` not setting ``inst.info['lowpass']`` properly by `Eric Larson`_

- Fix checks when constructing volumetric and surface source spaces with :func:`mne.setup_volume_source_space` and :func:`mne.setup_source_space`, respectively, by `Eric Larson`_

- Fix bug in handling of :class:`mne.Evoked` types that were not produced by MNE-Python (e.g., alternating average) by `Eric Larson`_

- Fix bug in :func:`mne.read_source_estimate` where vector volumetric source estimates could not be read by `Eric Larson`_

- Fix bug in :func:`mne.inverse_sparse.mixed_norm` and :func:`mne.inverse_sparse.tf_mixed_norm` where ``weights`` was supplied but ``weights_min`` was not, by `Eric Larson`_

- Fix bug in :func:`mne.set_eeg_reference` where non-EEG channels could be re-referenced by default if there were no EEG channels present, by `Eric Larson`_

- Fix bug in :func:`mne.io.Raw.plot` when using HiDPI displays and the MacOSX backend of matplotlib by `Eric Larson`_

- Fix bug in :func:`mne.viz.plot_compare_evokeds` when using Neuromag 122 system by `Eric Larson`_

- Fix bug in :func:`mne.Epochs.plot_psd` when some channels had zero/infinite ``psd`` values causing erroneous error messages by `Luke Bloy`_

- Fix :func:`mne.Evoked.decimate` not setting ``inst.first`` and ``inst.last`` properly by `Marijn van Vliet`_

- Fix :func:`mne.io.read_raw_brainvision` not handling ``Event`` markers created by PyCorder correctly by `Richard Höchenberger`_

- Fix support for string-like objects (such as :class:`python:pathlib.Path`) by `Eric Larson`_

- Fix :class:`mne.Report` silently suppressing exceptions when used as a context manager by `Marijn van Vliet`_

API
~~~

- Deprecate ``mne.channels.Montage`` class, ``mne.channels.read_montage`` and ``mne.channels.read_dig_montage`` function by `Joan Massich`_.

- Deprecate passing ``Montage``, ``str`` as montage parameter in :meth:`mne.io.Raw.set_montage` by `Joan Massich`_.

- Deprecate ``set_dig`` parameter in :meth:`mne.io.Raw.set_montage` and ``update_ch_names`` in ``mne.io.RawEEGLAB.set_montage`` when using :class:`mne.channels.DigMontage` as  by `Joan Massich`_.

- Now :meth:`mne.io.Raw.set_montage` raises an error when :class:`mne.channels.DigMontage` contains only a subset of the channels in ``raw.info``. It also adds ``raise_if_subset`` parameter to ensure backward compatibility (defaults to False in 0.19, to True in 0.20, and will be removed in 0.21) by `Joan Massich`_.

- Minimum dependency versions for the following libraries have been bumped up (by `Eric Larson`_):

  - NumPy: 1.12.1
  - SciPy: 0.18.1
  - matplotlib: 2.0.2
  - scikit-learn: 0.18.2
  - pandas 0.19.2

- New boolean parameter ``show_scrollbars`` for :meth:`mne.io.Raw.plot`, :meth:`mne.Epochs.plot`, and :meth:`mne.preprocessing.ICA.plot_sources` (and associated functions) that allows hiding the scrollbars and buttons for a "zen mode" data browsing experience. When the plot window has focus, zen mode can be toggled by pressing :kbd:`z`, by `Daniel McCloy`_.

- Deprecate ``mne.evoked.grand_average`` in favor of :func:`mne.grand_average` (which works on both :class:`~mne.Evoked` and :class:`~mne.time_frequency.AverageTFR`) by `Daniel McCloy`_

- Deprecate ``exclude`` parameter in :func:`mne.viz.plot_ica_sources` and :meth:`mne.preprocessing.ICA.plot_sources`, instead always use the ``exclude`` attribute of the ICA object by `Daniel McCloy`_.

- Deprecate ``montage`` parameter in favor of the ``set_montage`` method in all EEG data readers :func:`mne.io.read_raw_cnt`, :func:`mne.io.read_raw_egi`, :func:`mne.io.read_raw_edf`, :func:`mne.io.read_raw_gdf`, :func:`mne.io.read_raw_nicolet`, :func:`mne.io.read_raw_eeglab` and :func:`mne.read_epochs_eeglab` by `Alex Gramfort`_

- New parameter ``clear`` in :func:`mne.viz.plot_epochs_image` for clearing pre-existing axes before plotting into them by `Daniel McCloy`_

- :func:`mne.viz.plot_epochs_image` no longer supports ``group_by='type'`` — combining by channel type is now the default when ``picks`` is a channel type string; to get individual plots for each channel, pass ``picks`` as a list of channel names or indices by `Daniel McCloy`_

- New parameter ``combine`` in :func:`mne.viz.plot_compare_evokeds` for specifying method to combine information across channels by `Daniel McCloy`_

- FIFF constants related to SmartShield (``*_SMSH_*`` and ``*_SMARTSHIELD``) have been renamed to ``IAS`` for consistency with MEGIN, by `Eric Larson`_

- The ``gfp`` parameter of :func:`mne.viz.plot_compare_evokeds` is deprecated; use ``combine='gfp'`` instead by `Daniel McCloy`_

- The ``truncate_yaxis='max_ticks'`` parameter of :func:`mne.viz.plot_compare_evokeds` is deprecated; use ``truncate_yaxis='auto'`` instead by `Daniel McCloy`_

- The ``truncate_xaxis`` and ``truncate_yaxis`` parameters of :func:`mne.viz.plot_compare_evokeds` now perform one-sided truncation unless both are ``True`` by `Daniel McCloy`_

- The ``show_legend`` parameter of :func:`mne.viz.plot_compare_evokeds` is renamed to ``legend`` by `Daniel McCloy`_

- :func:`mne.viz.plot_compare_evokeds` always returns a list of figures even when a single figure is generated by `Daniel McCloy`_

- Deprecate ``average=True`` and ``spatial_colors=False`` for :func:`mne.Epochs.plot_psd` by `Jeff Hanna`_

- :func:`mne.io.read_raw_brainvision` no longer raises an error when there are inconsistencies between ``info['chs']`` and ``montage`` but warns instead by `Joan Massich`_

- Add ``update_ch_names`` parameter to ``mne.io.RawEEGLAB.set_montage`` to allow updating the channel names based on the montage by `Joan Massich`_

- Reading annotations contained in GDF files with :func:`mne.io.read_raw_gdf` now returns numeric event codes as descriptions (instead of textual descriptions) due to restrictive licensing of the GDF event code table from BioSig by `Clemens Brunner`_

- :func:`channels.find_ch_connectivity` now returns pre-built neighbor maps for KIT systems when available, by `Christian Brodbeck`_
