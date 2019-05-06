:orphan:

.. _data-formats:

Supported data formats
======================

When MNE-Python loads sensor data, the data are stored in a Python object of
type :class:`mne.io.Raw`. Specialized loading functions are provided for the
raw data file formats from a variety of equipment manufacturers. All raw data
input/output functions in MNE-Python are found in :mod:`mne.io` and start
with :samp:`read_raw_{*}`; see the documentation for each reader function for
more info on reading specific file types.

As seen in the table below, there are also a few formats defined by other
neuroimaging analysis software packages that are supported (EEGLAB,
FieldTrip). Like the equipment-specific loading functions, these will also
return an object of class :class:`~mne.io.Raw`.

============  =============  =========  ===================================
Data type     File format    Extension  MNE-Python function
============  =============  =========  ===================================
MEG           Artemis123     .bin       :func:`mne.io.read_raw_artemis123`

MEG           4-D            .dir       :func:`mne.io.read_raw_bti`
              Neuroimaging
              / BTI

MEG           CTF            .dir       :func:`mne.io.read_raw_ctf`

MEG           Elekta         .fif       :func:`mne.io.read_raw_fif`
              Neuromag

MEG           KIT            .sqd       :func:`mne.io.read_raw_kit`

MEG and EEG   FieldTrip      .mat       :func:`mne.io.read_raw_fieldtrip`

EEG           Brainvision    .vhdr      :func:`mne.io.read_raw_brainvision`

EEG           Biosemi data   .bdf       :func:`mne.io.read_raw_bdf`
              format

EEG           Neuroscan CNT  .cnt       :func:`mne.io.read_raw_cnt`

EEG           European data  .edf       :func:`mne.io.read_raw_edf`
              format

EEG           EEGLAB         .set       :func:`mne.io.read_raw_eeglab`

EEG           EGI simple     .egi       :func:`mne.io.read_raw_egi`
              binary

EEG           EGI MFF        .mff       :func:`mne.io.read_raw_egi`
              format

EEG           eXimia         .nxe       :func:`mne.io.read_raw_eximia`

EEG           General data   .gdf       :func:`mne.io.read_raw_gdf`
              format

EEG           Nicolet        .data      :func:`mne.io.read_raw_nicolet`
============  =============  =========  ===================================
