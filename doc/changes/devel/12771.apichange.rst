:func:`mne.datasets.fetch_fsaverage` now returns a :class:`python:pathlib.Path` object
rather than a string. Support for string concatenation with plus (``+``) is thus
deprecated and will be removed in 1.9, use the forward-slash ``/`` operator instead,
by `Eric Larson`_.
