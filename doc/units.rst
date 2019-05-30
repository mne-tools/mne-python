:orphan:

.. _units:

Internal representation (units)
===============================

When importing data, MNE-Python will always convert measurements to the same
standard units. Thus the in-memory representation of data are always in:

- Volts (eeg, eog, seeg, emg, ecg, bio, ecog)
- Teslas (magnetometers)
- Teslas/meter (gradiometers)
- Amperes*meter (dipole fits)
- Molar (aka mol/L) (fNIRS data: oxyhemoglobin (hbo), deoxyhemoglobin (hbr))
- Arbitrary units (various derived unitless quantities)

Note, however, that most MNE-Python plotting functions will scale the data when
plotted to yield nice-looking axis annotations in a sensible range; for
example, :meth:`mne.io.Raw.plot_psd` will convert teslas to femtoteslas (fT)
and volts to microvolts (μV) when plotting MEG and EEG data.

The units used in internal data representation are particularly important to
remember when extracting data from MNE-Python objects and manipulating it
outside MNE-Python (e.g., when using methods like :meth:`~mne.io.Raw.get_data`
or :meth:`~mne.Epochs.to_data_frame` to convert data to :class:`NumPy arrays
<numpy.ndarray>` or :class:`Pandas DataFrames <pandas.DataFrame>` for analysis
or plotting with other Python modules).
