Ten things about MNE
====================


.. _picking_channels:

Picking channels
----------------
If we are interested in analyzing only a part of the data,
e.g. if we have EEG and MEG data but we want to consider only EEG channels,
we can use the :func:`pick_types <mne.pick_types>` function.
We then give the resulting variable to every step of the analysis::

    >>> # We select only EEG channels
    >>> picks = mne.pick_types(raw.info, meg=False, eeg=True, eog=False,
                               stim=False, exclude='bads')
    >>> # and for example we fit the ICA with only those picked channels
    >>> ica.fit(raw, picks=picks)

Please note the `exclude='bads'` option, which excludes the channels we previously
marked as bad (see :ref:`marking_bad_channels`).


.. topic:: See also:

    * :ref:`sphx_glr_auto_examples_preprocessing_plot_virtual_evoked.py`
