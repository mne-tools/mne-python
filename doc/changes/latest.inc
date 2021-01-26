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

.. |Richard Koehler| replace:: **Richard Koehler**

Enhancements
~~~~~~~~~~~~
- Add dbs as new channel type for deep brain stimulation (DBS) recordings (:gh:`8739` **by new contributor** |Richard Koehler|_)

- Update the ``notebook`` 3d backend to use ``ipyvtk_simple`` for a better integration within ``Jupyter`` (:gh:`8503` by `Guillaume Favelier`_)

- Add toggle-all button to :class:`mne.Report` HTML and ``width`` argument to :meth:`mne.Report.add_bem_to_section` (:gh:`8723` by `Eric Larson`_)

- Add infant template MRI dataset downloader :func:`mne.datasets.fetch_infant_template` (:gh:`8738` by `Eric Larson`_ and `Christian O'Reilly`_)

- Speed up :func:`mne.inverse_sparse.tf_mixed_norm` using STFT/ISTFT linearity (:gh:`8697` by `Eric Larson`_)

- `mne.Report.parse_folder` now processes supported non-FIFF files by default, too (:gh:`8744` by `Richard Höchenberger`_)

- `mne.Report` has gained a new method `~mne.Report.add_custom_css` for adding user-defined styles (:gh:`8762` by `Richard Höchenberger`_)

- Add option to control appearance of opaque inside surface of the head to :ref:`mne coreg` (:gh:`8793` by `Eric Larson`_)

- Add :func:`mne.io.read_raw_nedf` for reading StarStim / enobio NEDF files (:gh:`8734` by `Tristan Stenner`_)

- Add :meth:`raw.describe() <mne.io.Raw.describe>` to display (or return) descriptive statistics for each channel (:gh:`8760` by `Clemens Brunner`_)

- Add :meth:`annotations.to_data_frame() <mne.Annotations.to_data_frame>` to return annotations as a pandas dataframe (:gh:`8783` by `Robert Luke`_)


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

- Fix title not shown in :func:`mne.viz.plot_montage` (:gh:`8752` by `Clemens Brunner`_)

- `mne.io.read_raw_egi` now correctly handles `pathlib.Path` filenames (:gh:`8759` by `Richard Höchenberger`_)

API changes
~~~~~~~~~~~

- None yet
