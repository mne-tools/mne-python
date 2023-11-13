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
- EEGLAB files (saved as MAT versions less than v7.3) can now be imported with :func:`mne.io.read_raw_eeglab` without the optional dependency ``pymatreader`` (:gh:`11006` by `Clemens Brunner`_)
- Add eight source estimate comparison metrics to new submodule :mod:`mne.simulation.metrics` (:gh:`6233` by `Kostiantyn Maksymenko`_ and `Alex Gramfort`_)
- Add :func:`mne.time_frequency.csd_tfr` to compute cross-spectral density from :class:`mne.time_frequency.EpochsTFR` (:gh:`10986` by `Alex Rockhill`_)
- Improve ``repr()`` for :class:`mne.minimum_norm.InverseOperator` when loose orientation is used (:gh:`11048` by `Eric Larson`_)
- Add support for vector source estimates in :func:`mne.minimum_norm.get_point_spread` and :func:`mne.minimum_norm.get_cross_talk` (:gh:`8639` by `Olaf Hauk`_, `Marijn van Vliet`_, `Alex Gramfort`_, and `Eric Larson`_)
- :meth:`mne.Epochs.plot_psd_topomap` now suppresses redundant colorbars when ``vlim='joint'`` (:gh:`11051` by `Daniel McCloy`_)
- Add ``starting_affine`` keyword argument to :func:`mne.transforms.compute_volume_registration` to initialize an alignment with an affine (:gh:`11020` by `Alex Rockhill`_)
- The ``trans`` parameter in :func:`mne.make_field_map` now accepts a :class:`~pathlib.Path` object, and uses standardised loading logic (:gh:`10784` by :newcontrib:`Andrew Quinn`)
- Add HTML representation for `~mne.Evoked` in Jupyter Notebooks (:gh:`11075` by `Valerii Chirkov`_ and `Andrew Quinn`_)
- Allow :func:`mne.beamformer.make_dics` to take ``pick_ori='vector'`` to compute vector source estimates (:gh:`19080` by `Alex Rockhill`_)
- Add ``on_missing`` functionality to all of our classes that have a ``drop_channels`` method, to control what happens when channel names are not in the object (:gh:`11077` by `Andrew Quinn`_)

- Added noisy point-spread functions to mne.minimum_norm.make_inverse_resolution_matrix (:gh:`11058` by `Olaf Hauk`)

Bugs
~~~~
- Fix bug in :func:`mne.io.read_raw_eeglab` where unlabeled fiducials causde reading errors (:gh:`11074` by :newcontrib:`Sebastiaan Mathot`)
- Fix bug in :func:`mne.time_frequency.read_csd` that returned ``projs`` as a list of dict instead of :class:`mne.Projection` (:gh:`11072` by :newcontrib:`Chetan Gohil`)
- Fix bug in :func:`mne.decoding.TimeFrequency` that prevented cloning if constructor arguments were modified (:gh:`11004` by :newcontrib:`Daniel Carlström Schad`)
- Fix bug in :class:`mne.viz.Brain` constructor where the first argument was named ``subject_id`` instead of ``subject`` (:gh:`11049` by `Eric Larson`_)
- Document ``height`` and ``weight`` keys of  ``subject_info`` entry in :class:`mne.Info` (:gh:`11019` by :newcontrib:`Sena Er`)
- Fix bug in :func:`mne.viz.plot_filter` when plotting filters created using ``output='ba'`` mode with ``compensation`` turned on. (:gh:`11040` by `Marian Dovgialo`_)
- Fix bug in :func:`mne.viz.plot_topomap` when providing ``sphere="eeglab"`` (:gh:`11081` by `Mathieu Scheltienne`_)

API changes
~~~~~~~~~~~
- The ``bands`` parameter of :meth:`mne.Epochs.plot_psd_topomap` now accepts :class:`dict` input; legacy :class:`tuple` input is supported, but discouraged for new code (:gh:`11050` by `Daniel McCloy`_)
- The ``show_toolbar`` argument to :class:`mne.viz.Brain` is being removed by deprecation (:gh:`11049` by `Eric Larson`_)
