.. _migrating:

Migrating from other analysis software
======================================

Here we offer some tips on how to migrate from other analysis software.

EEGLAB
^^^^^^

To read in data exported from EEGLAB, MNE-Python includes an :file:`.edf`
reader :func:`mne.io.read_raw_edf` and a ``set`` file reader. To read in
``set`` files containing ``raw`` data, use :func:`mne.io.read_raw_eeglab` and
to read in ``set`` files containing ``epochs`` data, use
:func:`mne.read_epochs_eeglab`.

This table summarizes equivalent EEGLAB and MNE-Python code for some of the
most common analysis tasks. For the sake of clarity, the table below assumes
the following variables exist: the file name ``fname``, time interval of the
epochs ``tmin`` and ``tmax``, and the experimental conditions ``cond1`` and
``cond2``. The variables ``l_freq`` and ``h_freq`` are the frequencies (in Hz)
below which and above which to filter out data.

.. cssclass:: table-bordered
.. rst-class:: midvalign

+---------------------+----------------------------------------------------------+--------------------------------------------------------------------------------------------------+
| Processing step     | EEGLAB function                                          | MNE-Python                                                                                       |
+=====================+==========================================================+==================================================================================================+
| Get started         | | ``addpath(...);``                                      | | :mod:`import mne <mne>`                                                                        |
|                     | | ``eeglab;``                                            | | :mod:`from mne import io, <mne.io>` :class:`~mne.Epochs`                                       |
|                     |                                                          | | :mod:`from mne.preprocessing <mne.preprocessing>` :class:`import ICA <mne.preprocessing.ICA>`  |
+---------------------+----------------------------------------------------------+--------------------------------------------------------------------------------------------------+
| Import data         | ``EEG = pop_fileio(fname);``                             | | :func:`raw = io.read_raw_fif(fname) <mne.io.read_raw_fif>`                                     |
|                     |                                                          | | :func:`raw = io.read_raw_edf(fname) <mne.io.read_raw_edf>`                                     |
|                     |                                                          | | :func:`raw = io.read_raw_eeglab(fname) <mne.io.read_raw_eeglab>`                               |
+---------------------+----------------------------------------------------------+--------------------------------------------------------------------------------------------------+
| Filter data         | ``EEG = pop_eegfiltnew(EEG, l_freq, h_freq);``           | :func:`raw.filter(l_freq, h_freq) <mne.io.Raw.filter>`                                           |
+---------------------+----------------------------------------------------------+--------------------------------------------------------------------------------------------------+
| Run ICA             | ``EEG = pop_runica(EEG, 'pca', n);``                     | | :class:`ica = ICA(max_pca_components=n) <mne.preprocessing.ICA>`                               |
|                     |                                                          | | :func:`ica.fit(raw) <mne.preprocessing.ICA.fit>`                                               |
+---------------------+----------------------------------------------------------+--------------------------------------------------------------------------------------------------+
| Epoch data          | | ``event_id = {'cond1', 'cond2'};``                     | | :func:`events = mne.find_events(raw) <mne.find_events>`                                        |
|                     | | ``Epochs = pop_epochs(EEG, event_id, [tmin, tmax]);``  | | :py:class:`event_id = dict(cond1=32, cond2=64) <dict>`                                         |
|                     |                                                          | | :class:`epochs = Epochs(raw, events, event_id, tmin, tmax) <mne.Epochs>`                       |
+---------------------+----------------------------------------------------------+--------------------------------------------------------------------------------------------------+
| Selecting epochs    | ``Epochs = pop_epochs(EEG_epochs, {cond2});``            | :class:`epochs[cond2] <mne.Epochs>`                                                              |
+---------------------+----------------------------------------------------------+--------------------------------------------------------------------------------------------------+
| ERP butterfly plot  | ``pop_timtopo(EEG_epochs, ...);``                        | | :meth:`evoked = epochs[cond2].average() <mne.Epochs.average>`                                  |
|                     |                                                          | | :func:`evoked.plot() <mne.Evoked.plot>`                                                        |
|                     |                                                          | | :func:`evoked.plot_joint() <mne.Evoked.plot_joint>`                                            |
+---------------------+----------------------------------------------------------+--------------------------------------------------------------------------------------------------+
| Contrast ERPs       | ``pop_compareerps(EEG_epochs1, EEG_epochs2);``           | | :func:`mne.combine_evoked([evoked1, -evoked2], weights='equal').plot() <mne.combine_evoked>`   |
|                     |                                                          | | :func:`mne.viz.plot_compare_evokeds([evoked1, evoked2]) <mne.viz.plot_compare_evokeds>`        |
+---------------------+----------------------------------------------------------+--------------------------------------------------------------------------------------------------+
| Save data           | ``EEG = pop_saveset(EEG, fname);``                       | | :func:`raw.save(fname) <mne.io.Raw.save>`                                                      |
|                     |                                                          | | :func:`epochs.save(fname) <mne.Epochs.save>`                                                   |
|                     |                                                          | | :func:`evoked.save(fname) <mne.Evoked.save>`                                                   |
+---------------------+----------------------------------------------------------+--------------------------------------------------------------------------------------------------+

Potential pitfalls
~~~~~~~~~~~~~~~~~~

- Many of the MNE-Python objects have methods that operate in-place to save
  memory (i.e., the data in the :class:`~mne.io.Raw` object is changed when you
  call :meth:`raw.filter(lfreq, hfreq) <mne.io.Raw.filter>`). If you do not
  want this, it is always possible to first call the object's
  :meth:`~mne.io.Raw.copy` method (e.g., ``filtered_raw =
  raw.copy().filter(lfreq, hfreq)``). In addition, some MNE-Python functions
  have a boolean ``copy`` parameter that achieves the same purpose.

- The concept of channel types is critical in MNE because it supports analysis
  of multimodal data (e.g., EEG, MEG, EOG, Stim channel, etc) whereas most
  EEGLAB functions assume all channels are of the same type (EEG). To restrict
  channels to a single type, see :func:`mne.pick_types`, :meth:`raw.pick_types
  <mne.io.Raw.pick_types>`, :meth:`epochs.pick_types <mne.Epochs.pick_types>`,
  :meth:`evoked.pick_types <mne.Evoked.pick_types>`, etc.
