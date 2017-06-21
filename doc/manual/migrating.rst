.. _migrating:

Migrating from EEGLAB
=====================

To read in data exported from EEGLAB, MNE offers an EDF reader :func:`mne.io.read_raw_edf` and a ``set`` file reader.
To read in `set` files containing ``raw`` data, use :func:`mne.io.read_raw_eeglab` and to read in ``set`` files containing
``epochs`` data, use :func:`mne.read_epochs_eeglab`.

Here is a cheatsheet to help users migrate painlessly from EEGLAB. For the sake of clarity, let us assume
that the following are already defined or known: the file name ``fname``, time interval of the epochs ``tmin`` and ``tmax``,
and the conditions ``cond1`` and ``cond2``. The variables ``l_freq`` and ``h_freq`` are the frequencies (in Hz) below which
and above which to filter out data.

+-------------------+--------------------------------------------------------------+----------------------------------------------------------------------------------------------+
| Processing step   | EEGLAB function                                              | MNE                                                                                          |
+===================+==============================================================+==============================================================================================+
| Get started       | | addpath(...);                                              | | import mne                                                                                 |
|                   | | eeglab;                                                    | | from mne import io,     :class:`Epochs <mne.Epochs>`                                       |
|                   |                                                              | | from mne.preprocessing import     :class:`ICA <mne.preprocessing.ICA>`                     |
+-------------------+--------------------------------------------------------------+----------------------------------------------------------------------------------------------+
| Import data       | EEG = pop_fileio(fname);                                     | | :func:`raw = io.read_raw_fif(fname) <mne.io.Raw>`                                          |
|                   |                                                              | | :func:`raw = io.read_raw_edf(fname) <mne.io.read_raw_edf>`                                 |
|                   |                                                              | | :func:`raw = io.read_raw_eeglab(fname) <mne.io.read_raw_eeglab>`                           |
+-------------------+--------------------------------------------------------------+----------------------------------------------------------------------------------------------+
| Filter data       | EEG = pop_eegfiltnew(EEG, l_freq, h_freq);                   | :func:`raw.filter(l_freq, h_freq) <mne.io.Raw.filter>`                                       |
+-------------------+--------------------------------------------------------------+----------------------------------------------------------------------------------------------+
| Run ICA           | EEG = pop_runica(EEG);                                       | :func:`ica.fit(raw) <mne.preprocessing.ICA.fit>`                                             |
+-------------------+--------------------------------------------------------------+----------------------------------------------------------------------------------------------+
| Epoch data        | | event_id = {'cond1', 'cond2'};                             | | :func:`events = mne.find_events(raw) <mne.find_events>`                                    |
|                   | | Epochs = pop_epochs(EEG, event_id, [tmin, tmax]);          | | :py:class:`event_id = dict(cond1=32, cond2=64) <dict>`                                     |
|                   | |                                                            | | :class:`epochs = Epochs(raw, events, event_id, tmin, tmax) <mne.Epochs>`                   |
+-------------------+--------------------------------------------------------------+----------------------------------------------------------------------------------------------+
| Selecting epochs  | Epochs = pop_epochs(EEG_epochs, {cond2});                    | :class:`epochs[cond2] <mne.Epochs>`                                                          |
+-------------------+--------------------------------------------------------------+----------------------------------------------------------------------------------------------+
| ERP butterfly plot| pop_timtopo(EEG_epochs, ...);                                | :func:`evoked.plot() <mne.Evoked.plot>`                                                      |
+-------------------+--------------------------------------------------------------+----------------------------------------------------------------------------------------------+
| Contrast ERPs     | pop_compareerps(EEG_epochs1, EEG_epochs2);                   | :func:`mne.combine_evoked([evoked1, -evoked2], weights='equal').plot() <mne.combine_evoked>` |
+-------------------+--------------------------------------------------------------+----------------------------------------------------------------------------------------------+
| Save data         | EEG = pop_saveset(EEG, fname);                               | | :func:`raw.save(fname) <mne.io.Raw.save>`                                                  |
|                   |                                                              | | :func:`epochs.save(fname) <mne.Epochs.save>`                                               |
|                   |                                                              | | :func:`evoked.save(fname) <mne.Evoked.save>`                                               |
+-------------------+--------------------------------------------------------------+----------------------------------------------------------------------------------------------+

Note that MNE has functions to read a variety of file formats, not just :func:`mne.io.Raw`. The interested user is directed to the :ref:`IO documentation <ch_convert>`.

Pitfalls
--------

* Python methods often operate in-place. This means the object the method is called on is modified in-place (e.g., see the filter example above).
  This can be confusing to new users migrating from Matlab. However, it is also possible to ask MNE functions not to modify the input.
  To do this, call the ``copy`` method of the object (.e.g, use  :func:`raw_filtered = raw.copy().filter(l_freq, h_freq) <mne.io.Raw.copy>`).
* The concept of channel types is critical in MNE because it supports analysis of multimodal data (e.g., EEG, MEG, EOG, Stim channel)
  whereas most EEGLAB functions assume the same channel type (EEG).
