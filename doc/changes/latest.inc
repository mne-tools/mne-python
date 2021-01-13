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
- Add dbs as new channel type for deep brain stimulation (DBS) recordings  (:gh:`8739` **by new contributor** |Richard Koehler|_)

- Update the ``notebook`` 3d backend to use ``ipyvtk_simple`` for a better integration within ``Jupyter`` (:gh:`8503` by `Guillaume Favelier`_)

- Speed up :func:`mne.inverse_sparse.tf_mixed_norm` using STFT/ISTFT linearity by `Eric Larson`_

Bugs
~~~~
- Fix zen mode and scalebar toggling for :meth:`raw.plot() <mne.io.Raw.plot>` when using the ``macosx`` matplotlib backend (:gh:`8688` by `Daniel McCloy`_)

- Fix bug with :func:`mne.preprocessing.maxwell_filter` where the eSSS basis had to exactly match the good channels instead of being a superset (:gh:`8675` by `Eric Larson`_)

- Fix bug with :meth:`raw.plot() <mne.io.Raw.plot>` where annotations didn't immediately appear when changing window duration (:gh:`8689` by `Daniel McCloy`_)

API changes
~~~~~~~~~~~

- None yet
