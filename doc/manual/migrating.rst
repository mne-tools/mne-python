.. _migrating

Migrating from EEGLAB
=====================

To read in data exported from EEGLAB, MNE offers :ref:`EDF and set file readers <ch_convert>`.

Here is a cheatsheet to help users migrate painlessly from EEGLAB. For the sake of clarity, let us assume
that the following are already defined or known: the file name ``fname``, low-pass cut-off frequency ``lfreq``,
time interval of the epochs ``tmin`` and ``tmax``, and the conditions ``cond1`` and ``cond2``.

+------------------+-------------------------------------------------------------+----------------------------------------------------+
|Processing step   |EEGLAB function                                              |MNE                                                 |
+==================+=============================================================+====================================================+
|Import data       |EEG = pop_fileio(fname);                                     | raw = io.Raw(fname)                                |
+------------------+-------------------------------------------------------------+----------------------------------------------------+
|Filter data       |EEG = pop_eegfiltnew(EEG, lfreq, hfreq);                     | raw.filter(lfreq, hfreq)                           |
+------------------+-------------------------------------------------------------+----------------------------------------------------+
|Run ICA           |EEG = pop_runica(EEG);                                       | ica.fit(raw)                                       |
+------------------+-------------------------------------------------------------+----------------------------------------------------+
|Epoching data     || event_id = {'cond1', 'cond2'};                             || events = mne.find_events(raw)                     |
|                  || Epochs = pop_epochs(EEG, event_id, [tmin, tmax]) ;         || event_id = dict(cond1=32, cond2=64)               |
|                  ||                                                            || epochs = Epochs(raw, events, event_id, tmin, tmax)|
+------------------+-------------------------------------------------------------+----------------------------------------------------+
|Selecting epochs  |Epochs = pop_epochs(EEG_epochs, {cond2});                    | epochs[cond2]                                      |
+------------------+-------------------------------------------------------------+----------------------------------------------------+
|ERP butterfly plot|pop_timtopo(EEG_epochs, ...);                                | evoked.plot()                                      |
+------------------+-------------------------------------------------------------+----------------------------------------------------+
|Contrast ERPs     |pop_compareerps(EEG_epochs1, EEG_epochs2);                   | (evoked1 - evoked2).plot()                         |
+------------------+-------------------------------------------------------------+----------------------------------------------------+
||                 ||                                                            || raw.save(fname)                                   |
|| Save data       || EEG = pop_saveset(EEG, fname);                             || epochs.save(fname)                                |
||                 ||                                                            || evoked.save(fname)                                |
+------------------+-------------------------------------------------------------+----------------------------------------------------+

Pitfalls
--------

* Python operations are natively in place. To explicitly ask for a copy like in Matlab, use the boolean argument ``copy`` provided in MNE functions. 
* The concept of channel types is critical in MNE because it supports analysis of multimodal data (e.g., EEG, MEG, EOG) whereas most EEGLAB 
  functions assume the same channel type to be EEG.
