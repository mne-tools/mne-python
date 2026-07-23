When creating :class:`~mne.Evoked` by averaging :class:`mne.Epochs` via the :meth:`~mne.Epochs.average`
method, static analysis tools like Pylance will now correctly infer whether a list of :class:`~mne.EvokedArray`
or a single :class:`~mne.EvokedArray` is returned that a `pathlib.Path`, enabling better editor support like
automated code completions on the returned object, by `Richard Höchenberger`_.