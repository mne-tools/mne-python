When indexing :class:`~mne.Epochs` (e.g. by doing ``epochs[0]``), static code analysis tools like Pylance
should now be able to infer that the returned object is an epoch, too, and provide editor support
like automated code completions, by `Richard Höchenberger`_.