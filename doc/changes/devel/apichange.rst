:attr:`mne.Info.ch_names` will now return an empty list instead of raising a :py:`KeyError` if no channels
are present, by `Richard Höchenberger`_.