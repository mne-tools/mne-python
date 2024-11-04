Fix a bug in :func:`mne.epochs.make_metadata`, where missing values in the columns
generated for ``keep_first`` and ``keep_last`` events were depicted by empty strings,
while it should have been ``NA`` values. This issue existed since MNE-Python 1.7,
by `Richard Höchenberger`_.
