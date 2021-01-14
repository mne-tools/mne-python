.. NOTE: we are now using links to highlight new functions and classes.
   Please follow the examples below like :func:`mne.stats.f_mway_rm`, so the
   whats_new page will have a link to the function/class documentation.

.. NOTE: there are 3 separate sections for changes, based on type:
   - "Enhancements" for new features
   - "Bugs" for bug fixes
   - "API changes" for backward-incompatible changes

.. _current:

Current (0.23.dev0)
-------------------

.. |New Contributor| replace:: **New Contributor**


Enhancements
~~~~~~~~~~~~
- Update the ``notebook`` 3d backend to use ``ipyvtk_simple`` for a better integration within ``Jupyter`` (:gh:`8503` by `Guillaume Favelier`_)

- Add toggle-all button to :class:`mne.Report` HTML and ``width`` argument to :meth:`mne.Report.add_bem_to_section` (:gh:`8723` by `Eric Larson`_)

- Speed up :func:`mne.inverse_sparse.tf_mixed_norm` using STFT/ISTFT linearity (:gh:`8697` by `Eric Larson`_)

Bugs
~~~~
- Fix zen mode and scalebar toggling for :meth:`raw.plot() <mne.io.Raw.plot>` when using the ``macosx`` matplotlib backend (:gh:`8688` by `Daniel McCloy`_)

- Fix bug with :func:`mne.preprocessing.maxwell_filter` where the eSSS basis had to exactly match the good channels instead of being a superset (:gh:`8675` by `Eric Larson`_)

- Fix bug with :meth:`mne.Report.add_bem_to_section` where ``n_jobs != 1`` would cause ``n_jobs`` subsets of MRI images in some orientations to be flipped (:gh:`8713` by `Eric Larson`_)

- Fix bug with :meth:`raw.plot() <mne.io.Raw.plot>` where annotations didn't immediately appear when changing window duration (:gh:`8689` by `Daniel McCloy`_)

- Fix bug with :func:`mne.io.read_raw_nicolet` where header type values such as num_sample and duration_in_sec where not parsed properly (:gh:`8712` by `Alex Gramfort`_)

- Fix bug with ``replace`` argument of :meth:`mne.Report.add_bem_to_section` and :meth:`mne.Report.add_slider_to_section` (:gh:`8723` by `Eric Larson`_)

- Allow sEEG channel types in :meth:`mne.Evoked.plot_joint` (:gh:`8736` by `Daniel McCloy`_)

- Function :func:`mne.set_bipolar_reference` was not working when passing ``Epochs`` constructed with some ``picks`` (:gh:`8728` by `Alex Gramfort`_)

- Fix anonymization issue of FIF files after IO round trip (:gh:`8731` by `Alex Gramfort`_)


API changes
~~~~~~~~~~~

- None yet
