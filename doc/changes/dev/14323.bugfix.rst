Fixed a bug where :meth:`mne.Epochs.interpolate_bads` raised ``IndexError`` for fNIRS data, because the nearest-neighbour donor assignment assumed 2-D (channels, times) data, by `Kalle Mäkelä`_.
