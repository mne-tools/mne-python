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

Version 1.6.dev0 (development)
------------------------------

Enhancements
~~~~~~~~~~~~
- Improve tests for saving splits with `Epochs` (:gh:`11884` by `Dmitrii Altukhov`_)
- Added functionality for linking interactive figures together, such that changing one figure will affect another, see :ref:`tut-ui-events` and :mod:`mne.viz.ui_events`. Current figures implementing UI events are :func:`mne.viz.plot_topomap` and :func:`mne.viz.plot_source_estimates` (:gh:`11685` :gh:`11891` by `Marijn van Vliet`_)
- HTML anchors for :class:`mne.Report` now reflect the ``section-title`` of the report items rather than using a global incrementor ``global-N`` (:gh:`11890` by `Eric Larson`_)
- Added public :func:`mne.io.write_info` to complement :func:`mne.io.read_info` (:gh:`11918` by `Eric Larson`_)
- Added option ``remove_dc`` to to :meth:`Raw.compute_psd() <mne.io.Raw.compute_psd>`, :meth:`Epochs.compute_psd() <mne.Epochs.compute_psd>`, and :meth:`Evoked.compute_psd() <mne.Evoked.compute_psd>`, to allow skipping DC removal when computing Welch or multitaper spectra (:gh:`11769` by `Nikolai Chapochnikov`_)
- Add the possibility to provide a float between 0 and 1 as ``n_grad``, ``n_mag`` and ``n_eeg`` in `~mne.compute_proj_raw`, `~mne.compute_proj_epochs` and `~mne.compute_proj_evoked` to select the number of vectors based on the cumulative explained variance (:gh:`11919` by `Mathieu Scheltienne`_)
- Add helpful error messages when using methods on empty :class:`mne.Epochs`-objects (:gh:`11306` by `Martin Schulz`_)
- Add :class:`~mne.time_frequency.EpochsSpectrumArray` and :class:`~mne.time_frequency.SpectrumArray` to support creating power spectra from :class:`NumPy array <numpy.ndarray>` data (:gh:`11803` by `Alex Rockhill`_)

Bugs
~~~~
- Fix bugs with saving splits for :class:`~mne.Epochs` (:gh:`11876` by `Dmitrii Altukhov`_)
- Fix bug with multi-plot 3D rendering where only one plot was updated (:gh:`11896` by `Eric Larson`_)
- Fix bug where subject birthdays were not correctly read by :func:`mne.io.read_raw_snirf` (:gh:`11912` by `Eric Larson`_)
- Fix bug with :func:`mne.chpi.compute_head_pos` for CTF data where digitization points were modified in-place, producing an incorrect result during a save-load round-trip (:gh:`11934` by `Eric Larson`_)
- Fix bug where non-compliant stimulus data streams were not ignored by :func:`mne.io.read_raw_snirf` (:gh:`11915` by `Johann Benerradi`_)
- Fix bug with ``pca=False`` in :func:`mne.minimum_norm.compute_source_psd` (:gh:`11927` by `Alex Gramfort`_)
- Removed preload parameter from :func:`mne.io.read_raw_eyelink`, because data are always preloaded no matter what preload is set to (:gh:`11910` by `Scott Huberty`_)
- Fix bug with :meth:`~mne.viz.Brain.add_annotation` when reading an annotation from a file with both hemispheres shown (:gh:`11946` by `Marijn van Vliet`_)

API changes
~~~~~~~~~~~
- ``mne.preprocessing.apply_maxfilter`` and ``mne maxfilter`` have been deprecated and will be removed in 1.7. Use :func:`mne.preprocessing.maxwell_filter` (see :ref:`this tutorial <tut-artifact-sss>`) in Python or the command-line utility from MEGIN ``maxfilter`` and :func:`mne.bem.fit_sphere_to_headshape` instead (:gh:`11938` by `Eric Larson`_)
- :func:`mne.io.kit.read_mrk` reading pickled files is deprecated using something like ``np.savetxt(fid, pts, delimiter="\t", newline="\n")`` to save your points instead (:gh:`11937` by `Eric Larson`_)
