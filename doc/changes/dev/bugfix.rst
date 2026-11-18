Fix bug where :meth:`mne.io.Raw.get_montage` could return the position of the wrong fNIRS optode when the montage has 10 or more sources or detectors (e.g., ``D1`` and ``D13``), by `Kalle Mäkelä`_.
