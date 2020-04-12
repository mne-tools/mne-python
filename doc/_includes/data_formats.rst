:orphan:

Supported data formats
======================

.. NOTE: part of this file is included in doc/overview/implementation.rst.
   Changes here are reflected there. If you want to link to this content,
   link to :ref:`data-formats`. The next line is
   a target for :start-after: so we can omit the title above:
   data-formats-begin-content

When MNE-Python loads sensor data, the data are stored in a Python object of
type :class:`mne.io.Raw`. Specialized loading functions are provided for the
raw data file formats from a variety of equipment manufacturers. All raw data
input/output functions in MNE-Python are found in :mod:`mne.io` and start
with :samp:`read_raw_{*}`; see the documentation for each reader function for
more info on reading specific file types.

As seen in the table below, there are also a few formats defined by other
neuroimaging analysis software packages that are supported (EEGLAB,
FieldTrip). Like the equipment-specific loading functions, these will also
return an object of class :class:`~mne.io.Raw`; additional functions are
available for reading data that has already been epoched or averaged (see
table).

.. NOTE: To include only the table, here's a different target for :start-after:
   data-formats-begin-table

.. cssclass:: table-bordered
.. rst-class:: midvalign

============  ============================================  =========  ===================================
Data type     File format                                   Extension  MNE-Python function
============  ============================================  =========  ===================================
MEG           :ref:`Artemis123 <import-artemis>`            .bin       :func:`mne.io.read_raw_artemis123`

MEG           :ref:`4-D Neuroimaging / BTi <import-bti>`    <dir>      :func:`mne.io.read_raw_bti`

MEG           :ref:`CTF <import-ctf>`                       <dir>      :func:`mne.io.read_raw_ctf`

MEG and EEG   :ref:`Elekta Neuromag <import-neuromag>`      .fif       :func:`mne.io.read_raw_fif`

MEG           :ref:`KIT <import-kit>`                       .sqd       :func:`mne.io.read_raw_kit`,
                                                                       :func:`mne.read_epochs_kit`

MEG and EEG   :ref:`FieldTrip <import-fieldtrip>`           .mat       :func:`mne.io.read_raw_fieldtrip`,
                                                                       :func:`mne.read_epochs_fieldtrip`,
                                                                       :func:`mne.read_evoked_fieldtrip`

EEG           :ref:`Brainvision <import-bv>`                .vhdr      :func:`mne.io.read_raw_brainvision`

EEG           :ref:`Biosemi data format <import-biosemi>`   .bdf       :func:`mne.io.read_raw_bdf`

EEG           :ref:`Neuroscan CNT <import-cnt>`             .cnt       :func:`mne.io.read_raw_cnt`

EEG           :ref:`European data format <import-edf>`      .edf       :func:`mne.io.read_raw_edf`

EEG           :ref:`EEGLAB <import-set>`                    .set       :func:`mne.io.read_raw_eeglab`,
                                                                       :func:`mne.read_epochs_eeglab`

EEG           :ref:`EGI simple binary <import-egi>`         .egi       :func:`mne.io.read_raw_egi`

EEG           :ref:`EGI MFF format <import-mff>`            .mff       :func:`mne.io.read_raw_egi`

EEG           :ref:`eXimia <import-nxe>`                    .nxe       :func:`mne.io.read_raw_eximia`

EEG           :ref:`General data format <import-gdf>`       .gdf       :func:`mne.io.read_raw_gdf`

EEG           :ref:`Nicolet <import-nicolet>`               .data      :func:`mne.io.read_raw_nicolet`

NIRS          :ref:`NIRx <import-nirx>`                     directory  :func:`mne.io.read_raw_nirx`
============  ============================================  =========  ===================================

More details are provided in the tutorials in the :ref:`tut-data-formats`
section.
