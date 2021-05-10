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

Enhancements
~~~~~~~~~~~~
.. - Add something cool (:gh:`9192` **by new contributor** |New Contributor|_)

- New function :func:`mne.chpi.get_chpi_info` to retrieve basic information about the cHPI system used when recording MEG data (:gh:`9369` by `Richard Höchenberger`_)

- Add support for NIRSport devices to `mne.io.read_raw_nirx` (:gh:`9348` **by new contributor** |David Julien|_, **new contributor** |Romain Derollepot|_, `Robert Luke`_, and `Eric Larson`_)


Bugs
~~~~
- Fix bug with :meth:`mne.Epochs.crop` and :meth:`mne.Evoked.crop` when ``include_tmax=False``, where the last sample was always cut off, even when ``tmax > epo.times[-1]`` (:gh:`9378` **by new contributor** |Jan Sosulski|_)

- Fix bug with `mne.io.read_raw_curry` when Curry 8 generate event file end with '.cdt.ceo', the function cannot read it (:gh:`9381` **by new contributor** |Xiaokai Xia|_)

API changes
~~~~~~~~~~~
- Nothing yet
