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

Current (1.3.dev0)
------------------

Enhancements
~~~~~~~~~~~~
- Add the argument ``max_iter`` to :func:`~mne.time_frequency.psd_array_multitaper` and to :func:`~mne.time_frequency.csd_array_multitaper` to increase the maximum number of iterations to reach convergence when using ``adaptive=True`` (:gh:`11237` by `Mathieu Scheltienne`_)
- Mixed, cortical + discrete source spaces with fixed orientations are now allowed. (:gh:`11241` by `Jevri Hanna`_)
- Add :func:`mne.beamformer.apply_dics_tfr_epochs` to apply a DICS beamformer to time-frequency resolved epochs (:gh:`11096` by `Alex Rockhill`_)
- Check whether head radius (estimated from channel positions) is correct when reading EEGLAB data with :func:`~mne.io.read_raw_eeglab` and :func:`~mne.read_epochs_eeglab`. If head radius is not within likely values, warn informing about possible units mismatch and the new ``montage_units`` argument (:gh:`11283` by `Mikołaj Magnuski`_).

Bugs
~~~~
- Fix bug to be able to customize legends when using :func:`mne.viz._brain.add_volume_labels` (:gh:`11289` by :newcontrib:`Mauricio Cespedes Tenorio`)
- Fix bug in documentation of :func:`mne.channels.make_dig_montage` (:gh:`11235` by :newcontrib:`Daniel Hasegan`)
- Add support for bad channel handling in :func:`mne.set_bipolar_reference` (:gh:`11245` by `Moritz Gerster`_, :newcontrib:`Dinara Issagaliyeva`, :newcontrib:`Jennifer Behnke`, :newcontrib:`Hakimeh Aslsardroud`, and :newcontrib:`Pavel Navratil`)
- Fix X coordinates of ear_left and ear_right in head outlines dict (:gh:`11255` by :newcontrib:`Tom Ma`)
- Add type checking for ``tmin`` and ``tmax`` arguments for ``crop`` (:gh:`11263` by :newcontrib:`Carlos de la Torre-Ortiz`).
- Fix bug where ``ica.reject_`` was not saved to disk, and the ``ica.reject_`` property was not inherited from ``Epochs`` when doing ``ICA.fit(epochs)`` (:gh:`11244` by `Eric Larson`_)
- Fix bug in automatic MESA detection for disabling advanced 3D options (:gh:`11271` by `Eric Larson`_)
- Fix bug in :meth:`set_montage('mgh60') <mne.io.Raw.set_montage>` and related functions where the ``'mgh60'`` sensor positions were incorrectly ordered (:gh:`11290` by `Eric Larson`_)
- Fix bug in the ``.compute_psd()`` methods where the number of unaggregated Welch segments was wrongly computed for some inputs, leading to an assertion error when computing the PSD (:gh:`11248` by `Daniel McCloy`_)
- Fix bug in the :func:`~mne.viz.plot_evoked_topo` and :meth:`~mne.Evoked.plot_topo`, where legend colors where shown incorrectly on newer matplotlib versions (:gh:`11258` by `Erkka Heinila`_)
- Fix bug where EEGLAB channel positions were read as meters, while they are commonly in millimeters, leading to head outlies of the size of one channel when plotting topomaps. Now ``montage_units`` argument has been added to :func:`~mne.io.read_raw_eeglab` and :func:`~mne.read_epochs_eeglab` to control in what units EEGLAB channel positions are read. The default is millimeters, ``'mm'`` (:gh:`11283` by `Mikołaj Magnuski`_)

API changes
~~~~~~~~~~~
- None yet
