:orphan:

.. include:: ../links.inc

.. _mne_matlab:

========================
MNE-MATLAB documentation
========================

.. contents:: Page contents
   :local:
   :depth: 2

.. note:: The MNE MATLAB Toolbox is compatible with Matlab versions 7.0 or later.

Overview
########

The MNE software contains a collection Matlab ``.m``-files to
facilitate interfacing with binary file formats of the MNE software.
The toolbox is located at ``$MNE_ROOT/share/matlab`` . The
names of the MNE Matlab toolbox functions begin either with `mne_` or
with `fiff_` . When you source the ``mne_setup`` script
as described in :ref:`user_environment`, one of the following actions
takes place:

- If you do not have the Matlab startup.m
  file, it will be created and lines allowing access to the MNE Matlab
  toolbox are added.

- If you have startup.m and it does not have the standard MNE
  Matlab toolbox setup lines, you will be instructed to add them manually.

- If you have startup.m and the standard MNE Matlab toolbox
  setup lines are there, nothing happens.

A summary of the available routines is provided in the `MNE-C manual`_. The
toolbox also contains a set of examples which may be useful starting points
for your own development. The names of these functions start with ``mne_ex``.

.. note::

   The MATLAB function ``fiff_setup_read_raw`` has a significant change. The
   sample numbers now take into account possible initial skip in the file,
   *i.e.*, the time between the start of the data acquisition and the start of
   saving the data to disk. The ``first_samp`` member of the returned structure
   indicates the initial skip in samples. If you want your own routines, which
   assume that initial skip has been removed, perform identically with the
   previous version, subtract ``first_samp`` from the sample numbers you
   specify to ``fiff_read_raw_segment``. Furthermore, ``fiff_setup_read_raw``
   has an optional argument to allow reading of unprocessed MaxShield data
   acquired with the Elekta MEG systems.

.. tabularcolumns:: |p{0.3\linewidth}|p{0.6\linewidth}|
.. _BGBCGHAG:
.. table:: High-level reading routines.

    +--------------------------------+--------------------------------------------------------------+
    | Function                       | Purpose                                                      |
    +================================+==============================================================+
    | fiff_find_evoked               | Find all evoked data sets from a file.                       |
    +--------------------------------+--------------------------------------------------------------+
    | fiff_read_bad_channels         | Read the bad channel list.                                   |
    +--------------------------------+--------------------------------------------------------------+
    | fiff_read_ctf_comp             | Read CTF software gradient compensation data.                |
    +--------------------------------+--------------------------------------------------------------+
    | fiff_read_evoked               | Read evoked-response data.                                   |
    +--------------------------------+--------------------------------------------------------------+
    | fiff_read_evoked_all           | Read all evoked-response data from a file.                   |
    +--------------------------------+--------------------------------------------------------------+
    | fiff_read_meas_info            | Read measurement information.                                |
    +--------------------------------+--------------------------------------------------------------+
    | fiff_read_mri                  | Read an MRI description file.                                |
    +--------------------------------+--------------------------------------------------------------+
    | fiff_read_proj                 | Read signal-space projection data.                           |
    +--------------------------------+--------------------------------------------------------------+
    | fiff_read_raw_segment          | Read a segment of raw data with time limits are specified    |
    |                                | in samples.                                                  |
    +--------------------------------+--------------------------------------------------------------+
    | fiff_read_raw_segment_times    | Read a segment of raw data with time limits specified        |
    |                                | in seconds.                                                  |
    +--------------------------------+--------------------------------------------------------------+
    | fiff_setup_read_raw            | Set up data structures before using fiff_read_raw_segment    |
    |                                | or fiff_read_raw_segment_times.                              |
    +--------------------------------+--------------------------------------------------------------+


.. tabularcolumns:: |p{0.3\linewidth}|p{0.6\linewidth}|
.. table:: Channel selection utilities.

    +--------------------------------+--------------------------------------------------------------+
    | Function                       | Purpose                                                      |
    +================================+==============================================================+
    | fiff_pick_channels             | Create a selector to pick desired channels from data         |
    |                                | according to include and exclude lists.                      |
    +--------------------------------+--------------------------------------------------------------+
    | fiff_pick_channels_evoked      | Pick desired channels from evoked-response data according    |
    |                                | to include and exclude lists.                                |
    +--------------------------------+--------------------------------------------------------------+
    | fiff_pick_info                 | Modify measurement info to include only selected channels.   |
    +--------------------------------+--------------------------------------------------------------+
    | fiff_pick_types                | Create a selector to pick desired channels from data         |
    |                                | according to channel types (MEG, EEG, STIM) in combination   |
    |                                | with include and exclude lists.                              |
    +--------------------------------+--------------------------------------------------------------+
    | fiff_pick_types_evoked         | Pick desired channels from evoked-response data according    |
    |                                | to channel types (MEG, EEG, STIM) in combination with        |
    |                                | include and exclude lists.                                   |
    +--------------------------------+--------------------------------------------------------------+


.. tabularcolumns:: |p{0.3\linewidth}|p{0.6\linewidth}|
.. table:: Coordinate transformation utilities.

    +--------------------------------+--------------------------------------------------------------+
    | Function                       | Purpose                                                      |
    +================================+==============================================================+
    | fiff_invert_transform          | Invert a coordinate transformation structure.                |
    +--------------------------------+--------------------------------------------------------------+
    | fiff_reset_ch_pos              | Reset channel position transformation to the default values  |
    |                                | present in the file.                                         |
    +--------------------------------+--------------------------------------------------------------+
    | fiff_transform_eeg_chs         | Transform electrode positions to another coordinate frame.   |
    +--------------------------------+--------------------------------------------------------------+
    | fiff_transform_meg_chs         | Apply a coordinate transformation to the sensor location     |
    |                                | data to bring the integration points to another coordinate   |
    |                                | frame.                                                       |
    +--------------------------------+--------------------------------------------------------------+


.. tabularcolumns:: |p{0.3\linewidth}|p{0.6\linewidth}|
.. table:: Basic reading routines.

    +--------------------------------+--------------------------------------------------------------+
    | Function                       | Purpose                                                      |
    +================================+==============================================================+
    | fiff_define_constants          | Define a structure which contains the constant relevant      |
    |                                | to fif files.                                                |
    +--------------------------------+--------------------------------------------------------------+
    | fiff_dir_tree_find             | Find nodes of a given type in a directory tree structure.    |
    +--------------------------------+--------------------------------------------------------------+
    | fiff_list_dir_tree             | List a directory tree structure.                             |
    +--------------------------------+--------------------------------------------------------------+
    | fiff_make_dir_tree             | Create a directory tree structure.                           |
    +--------------------------------+--------------------------------------------------------------+
    | fiff_open                      | Open a fif file and create the directory tree structure.     |
    +--------------------------------+--------------------------------------------------------------+
    | fiff_read_named_matrix         | Read a named matrix from a fif file.                         |
    +--------------------------------+--------------------------------------------------------------+
    | fiff_read_tag                  | Read one tag from a fif file.                                |
    +--------------------------------+--------------------------------------------------------------+
    | fiff_read_tag_info             | Read the info of one tag from a fif file.                    |
    +--------------------------------+--------------------------------------------------------------+
    | fiff_split_name_list           | Split a colon-separated list of names into a cell array      |
    |                                | of strings.                                                  |
    +--------------------------------+--------------------------------------------------------------+


.. tabularcolumns:: |p{0.3\linewidth}|p{0.6\linewidth}|
.. table:: Writing routines.

    +--------------------------------+--------------------------------------------------------------+
    | Function                       | Purpose                                                      |
    +================================+==============================================================+
    | fiff_end_block                 | Write a FIFF_END_BLOCK tag.                                  |
    +--------------------------------+--------------------------------------------------------------+
    | fiff_end_file                  | Write the standard closing.                                  |
    +--------------------------------+--------------------------------------------------------------+
    | fiff_start_block               | Write a FIFF_START_BLOCK tag.                                |
    +--------------------------------+--------------------------------------------------------------+
    | fiff_start_file                | Write the appropriate beginning of a file.                   |
    +--------------------------------+--------------------------------------------------------------+
    | fiff_write_ch_info             | Write a channel information structure.                       |
    +--------------------------------+--------------------------------------------------------------+
    | fiff_write_coord_trans         | Write a coordinate transformation structure.                 |
    +--------------------------------+--------------------------------------------------------------+
    | fiff_write_ctf_comp            | Write CTF compensation data.                                 |
    +--------------------------------+--------------------------------------------------------------+
    | fiff_write_dig_point           | Write one digitizer data point.                              |
    +--------------------------------+--------------------------------------------------------------+
    | fiff_write_complex             | Write single-precision complex numbers.                      |
    +--------------------------------+--------------------------------------------------------------+
    | fiff_write_complex_matrix      | Write a single-precision complex matrix.                     |
    +--------------------------------+--------------------------------------------------------------+
    | fiff_write_double              | Write double-precision floats.                               |
    +--------------------------------+--------------------------------------------------------------+
    | fiff_write_double_complex      | Write double-precision complex numbers.                      |
    +--------------------------------+--------------------------------------------------------------+
    |fiff_write_double_complex_matrix| Write a double-precision complex matrix.                     |
    +--------------------------------+--------------------------------------------------------------+
    | fiff_write_double_matrix       | Write a double-precision matrix.                             |
    +--------------------------------+--------------------------------------------------------------+
    | fiff_write_evoked              | Write an evoked-reponse data file.                           |
    +--------------------------------+--------------------------------------------------------------+
    | fiff_write_float               | Write single-precision floats.                               |
    +--------------------------------+--------------------------------------------------------------+
    | fiff_write_float_matrix        | Write a single-precision matrix.                             |
    +--------------------------------+--------------------------------------------------------------+
    | fiff_write_id                  | Write an id tag.                                             |
    +--------------------------------+--------------------------------------------------------------+
    | fiff_write_int                 | Write 32-bit integers.                                       |
    +--------------------------------+--------------------------------------------------------------+
    | fiff_write_int_matrix          | Write a matrix of 32-bit integers.                           |
    +--------------------------------+--------------------------------------------------------------+
    | fiff_write_name_list           | Write a name list.                                           |
    +--------------------------------+--------------------------------------------------------------+
    | fiff_write_named_matrix        | Write a named matrix.                                        |
    +--------------------------------+--------------------------------------------------------------+
    | fiff_write_proj                | Write SSP data.                                              |
    +--------------------------------+--------------------------------------------------------------+
    | fiff_write_short               | Write 16-bit integers.                                       |
    +--------------------------------+--------------------------------------------------------------+
    | fiff_write_string              | Write a string.                                              |
    +--------------------------------+--------------------------------------------------------------+


.. tabularcolumns:: |p{0.3\linewidth}|p{0.6\linewidth}|
.. table:: High-level data writing routines.

    +--------------------------------+--------------------------------------------------------------+
    | Function                       | Purpose                                                      |
    +================================+==============================================================+
    | fiff_write_evoked              | Write an evoked-response data file.                          |
    +--------------------------------+--------------------------------------------------------------+
    | fiff_finish_writing_raw        | Write the closing tags to a raw data file.                   |
    +--------------------------------+--------------------------------------------------------------+
    | fiff_start_writing_raw         | Start writing raw data file, *i.e.*, write the measurement   |
    |                                | information.                                                 |
    +--------------------------------+--------------------------------------------------------------+
    | fiff_write_dig_file            | Write a fif file containing digitization data.               |
    +--------------------------------+--------------------------------------------------------------+
    | fiff_write_raw_buffer          | Write one raw data buffer. This is used after a call to      |
    |                                | fiff_start_writing_raw.                                      |
    +--------------------------------+--------------------------------------------------------------+


.. tabularcolumns:: |p{0.3\linewidth}|p{0.6\linewidth}|
.. table:: Coil definition utilities.

    +--------------------------------+--------------------------------------------------------------+
    | Function                       | Purpose                                                      |
    +================================+==============================================================+
    | mne_add_coil_defs              | Add coil definitions to an array of channel information      |
    |                                | structures.                                                  |
    +--------------------------------+--------------------------------------------------------------+
    | mne_load_coil_def              | Load a coil definition file.                                 |
    +--------------------------------+--------------------------------------------------------------+

.. tabularcolumns:: |p{0.3\linewidth}|p{0.6\linewidth}|
.. table:: Routines for software gradient compensation and signal-space projection.

    +--------------------------------+--------------------------------------------------------------+
    | Function                       | Purpose                                                      |
    +================================+==============================================================+
    | mne_compensate_to              | Apply or remove CTF software gradient compensation from      |
    |                                | evoked-response data.                                        |
    +--------------------------------+--------------------------------------------------------------+
    | mne_get_current_comp           | Get the state of software gradient compensation from         |
    |                                | measurement info.                                            |
    +--------------------------------+--------------------------------------------------------------+
    | mne_make_compensator           | Make a compensation matrix which switches the status of      |
    |                                | CTF software gradient compensation from one state to another.|
    +--------------------------------+--------------------------------------------------------------+
    | mne_make_projector_info        | Create a signal-space projection operator with the           |
    |                                | projection item definitions and cell arrays of channel names |
    |                                | and bad channel names as input.                              |
    +--------------------------------+--------------------------------------------------------------+
    | mne_make_projector_info        | Like mne_make_projector but uses the measurement info        |
    |                                | structure as input.                                          |
    +--------------------------------+--------------------------------------------------------------+
    | mne_set_current_comp           | Change the information about the compensation status in      |
    |                                | measurement info.                                            |
    +--------------------------------+--------------------------------------------------------------+


.. tabularcolumns:: |p{0.3\linewidth}|p{0.6\linewidth}|
.. table:: High-level routines for reading MNE data files.

    +--------------------------------+--------------------------------------------------------------+
    | Function                       | Purpose                                                      |
    +================================+==============================================================+
    | mne_pick_channels_cov          | Pick desired channels from a sensor covariance matrix.       |
    +--------------------------------+--------------------------------------------------------------+
    | mne_pick_channels_forward      | Pick desired channels (rows) from a forward solution.        |
    +--------------------------------+--------------------------------------------------------------+
    | mne_read_bem_surfaces          | Read triangular tessellations of surfaces for                |
    |                                | boundary-element models.                                     |
    +--------------------------------+--------------------------------------------------------------+
    | mne_read_cov                   | Read a covariance matrix.                                    |
    +--------------------------------+--------------------------------------------------------------+
    | mne_read_epoch                 | Read an epoch of data from the output file of mne_epochs2mat.|
    +--------------------------------+--------------------------------------------------------------+
    | mne_read_events                | Read an event list from a fif file produced by               |
    |                                | mne_browse_raw or mne_process_raw.                           |
    +--------------------------------+--------------------------------------------------------------+
    | mne_read_forward_solution      | Read a forward solution from a fif file.                     |
    +--------------------------------+--------------------------------------------------------------+
    | mne_read_inverse_operator      | Read an inverse operator from a fif file.                    |
    +--------------------------------+--------------------------------------------------------------+
    | mne_read_morph_map             | Read an morphing map produced with mne_make_morph_maps.      |
    +--------------------------------+--------------------------------------------------------------+
    | mne_read_noise_cov             | Read a noise-covariance matrix from a fif file.              |
    +--------------------------------+--------------------------------------------------------------+
    | mne_read_source_spaces         | Read source space information from a fif file.               |
    +--------------------------------+--------------------------------------------------------------+


.. tabularcolumns:: |p{0.3\linewidth}|p{0.6\linewidth}|
.. table:: High-level routines for writing MNE data files.

    +--------------------------------+--------------------------------------------------------------+
    | Function                       | Purpose                                                      |
    +================================+==============================================================+
    | mne_write_cov                  | Write a covariance matrix to an open file.                   |
    +--------------------------------+--------------------------------------------------------------+
    | mne_write_cov_file             | Write a complete file containing just a covariance matrix.   |
    +--------------------------------+--------------------------------------------------------------+
    | mne_write_events               | Write a fif format event file compatible with mne_browse_raw |
    |                                | and mne_process_raw.                                         |
    +--------------------------------+--------------------------------------------------------------+
    | mne_write_inverse_sol_stc      | Write stc files containing an inverse solution or other      |
    |                                | dynamic data on the cortical surface.                        |
    +--------------------------------+--------------------------------------------------------------+
    | mne_write_inverse_sol_w        | Write w files containing an inverse solution or other static |
    |                                | data on the cortical surface.                                |
    +--------------------------------+--------------------------------------------------------------+


.. tabularcolumns:: |p{0.3\linewidth}|p{0.6\linewidth}|
.. _BABBDDAI:
.. table:: Routines related to stc, w, and label files.

    +--------------------------------+--------------------------------------------------------------+
    | Function                       | Purpose                                                      |
    +================================+==============================================================+
    | mne_read_stc_file              | Read data from one stc file. The vertex numbering in the     |
    |                                | returned structure will start from 0.                        |
    +--------------------------------+--------------------------------------------------------------+
    | mne_read_stc_file1             | Read data from one stc file. The vertex numbering in the     |
    |                                | returned structure will start from 1.                        |
    +--------------------------------+--------------------------------------------------------------+
    | mne_read_w_file                | Read data from one w file. The vertex numbering in the       |
    |                                | returned structure will start from 0.                        |
    +--------------------------------+--------------------------------------------------------------+
    | mne_read_w_file1               | Read data from one w file. The vertex numbering in the       |
    |                                | returned structure will start from 1.                        |
    +--------------------------------+--------------------------------------------------------------+
    | mne_write_stc_file             | Write a new stc file. It is assumed the the vertex numbering |
    |                                | in the input data structure containing the stc information   |
    |                                | starts from 0.                                               |
    +--------------------------------+--------------------------------------------------------------+
    | mne_write_stc_file1            | Write a new stc file. It is assumed the the vertex numbering |
    |                                | in the input data structure containing the stc information   |
    |                                | starts from 1.                                               |
    +--------------------------------+--------------------------------------------------------------+
    | mne_write_w_file               | Write a new w file. It is assumed the the vertex numbering   |
    |                                | in the input data structure containing the w file            |
    |                                | information starts from 0.                                   |
    +--------------------------------+--------------------------------------------------------------+
    | mne_write_w_file1              | Write a new w file. It is assumed the the vertex numbering   |
    |                                | in the input data structure containing the w file            |
    |                                | information starts from 1.                                   |
    +--------------------------------+--------------------------------------------------------------+
    | mne_read_label_file            | Read a label file (ROI).                                     |
    +--------------------------------+--------------------------------------------------------------+
    | mne_write_label_file           | Write a label file (ROI).                                    |
    +--------------------------------+--------------------------------------------------------------+
    | mne_label_time_courses         | Extract time courses corresponding to a label from an        |
    |                                | stc file.                                                    |
    +--------------------------------+--------------------------------------------------------------+


.. tabularcolumns:: |p{0.3\linewidth}|p{0.6\linewidth}|
.. table:: Routines for reading FreeSurfer surfaces.

    +--------------------------------+--------------------------------------------------------------+
    | Function                       | Purpose                                                      |
    +================================+==============================================================+
    | mne_read_curvature             | Read a curvature file.                                       |
    +--------------------------------+--------------------------------------------------------------+
    | mne_read_surface               | Read one surface, return the vertex locations and            |
    |                                | triangulation info.                                          |
    +--------------------------------+--------------------------------------------------------------+
    | mne_read_surfaces              | Read surfaces corresponding to one or both hemispheres.      |
    |                                | Optionally read curvature information and add derived        |
    |                                | surface data.                                                |
    +--------------------------------+--------------------------------------------------------------+
    | mne_reduce_surface             | Reduce the number of triangles on a surface using the        |
    |                                | reducepatch Matlab function.                                 |
    +--------------------------------+--------------------------------------------------------------+
    | mne_write_surface              | Write a FreeSurfer surface file.                             |
    +--------------------------------+--------------------------------------------------------------+


.. tabularcolumns:: |p{0.3\linewidth}|p{0.6\linewidth}|
.. _BGBEGFBD:
.. table:: Utility functions.

    +--------------------------------+--------------------------------------------------------------+
    | Function                       | Purpose                                                      |
    +================================+==============================================================+
    | mne_block_diag                 | Create a sparse block-diagonal matrix out of a vector.       |
    +--------------------------------+--------------------------------------------------------------+
    | mne_combine_xyz                | Calculate the square sum of the three Cartesian components   |
    |                                | of several vectors listed in one row or column vector.       |
    +--------------------------------+--------------------------------------------------------------+
    | mne_file_name                  | Compose a file name relative to $MNE_ROOT.                   |
    +--------------------------------+--------------------------------------------------------------+
    | mne_find_channel               | Find a channel by name from measurement info.                |
    +--------------------------------+--------------------------------------------------------------+
    | mne_find_source_space_hemi     | Determine whether a given source space belongs to the left   |
    |                                | or right hemisphere.                                         |
    +--------------------------------+--------------------------------------------------------------+
    | mne_fread3                     | Read a three-byte integer.                                   |
    +--------------------------------+--------------------------------------------------------------+
    | mne_fwrite3                    | Write a three-byte integer.                                  |
    +--------------------------------+--------------------------------------------------------------+
    | mne_make_combined_event_file   | Combine data from several trigger channels into one event    |
    |                                | file.                                                        |
    +--------------------------------+--------------------------------------------------------------+
    | mne_omit_first_line            | Omit first line from a multi-line message. This routine is   |
    |                                | useful for formatting error messages.                        |
    +--------------------------------+--------------------------------------------------------------+
    | mne_prepare_inverse_operator   | Prepare inverse operator data for calculating L2             |
    |                                | minimum-norm solutions and dSPM.                             |
    +--------------------------------+--------------------------------------------------------------+
    | mne_setup_toolbox              | Set up the MNE Matlab toolbox.                               |
    +--------------------------------+--------------------------------------------------------------+
    | mne_transform_coordinates      | Transform locations between different coordinate systems.    |
    |                                | This function uses the output file from                      |
    |                                | ``mne_collect_transforms``.                                  |
    +--------------------------------+--------------------------------------------------------------+
    | mne_transpose_named_matrix     | Create a transpose of a named matrix.                        |
    +--------------------------------+--------------------------------------------------------------+
    | mne_transform_source_space_to  | Transform source space data to another coordinate frame.     |
    +--------------------------------+--------------------------------------------------------------+


.. tabularcolumns:: |p{0.3\linewidth}|p{0.6\linewidth}|
.. _BGBEFADJ:
.. table:: Examples demonstrating the use of the toolbox.

    +--------------------------------+--------------------------------------------------------------+
    | Function                       | Purpose                                                      |
    +================================+==============================================================+
    | mne_ex_average_epochs          | Example of averaging epoch data produced by mne_epochs2mat.  |
    +--------------------------------+--------------------------------------------------------------+
    | mne_ex_cancel_noise            | Example of noise cancellation procedures.                    |
    +--------------------------------+--------------------------------------------------------------+
    | mne_ex_compute_inverse         | Example of computing a L2 minimum-norm estimate or a dSPM    |
    |                                | solution.                                                    |
    +--------------------------------+--------------------------------------------------------------+
    | mne_ex_data_sets               | Example of listing evoked-response data sets.                |
    +--------------------------------+--------------------------------------------------------------+
    | mne_ex_evoked_grad_amp         | Compute tangential gradient amplitudes from planar           |
    |                                | gradiometer data.                                            |
    +--------------------------------+--------------------------------------------------------------+
    | mne_ex_read_epochs             | Read epoch data from a raw data file.                        |
    +--------------------------------+--------------------------------------------------------------+
    | mne_ex_read_evoked             | Example of reading evoked-response data.                     |
    +--------------------------------+--------------------------------------------------------------+
    | mne_ex_read_raw                | Example of reading raw data.                                 |
    +--------------------------------+--------------------------------------------------------------+
    | mne_ex_read_write_raw          | Example of processing raw data (read and write).             |
    +--------------------------------+--------------------------------------------------------------+

.. note:: In order for the inverse operator calculation to work correctly with data processed with the Elekta-Neuromag Maxfilter (TM) software, the so-called *processing history* block must be included in data files. Previous versions of the MNE Matlab functions did not copy processing history to files saved. As of March 30, 2009, the Matlab toolbox routines fiff_start_writing_raw and fiff_write_evoked have been enhanced to include these data to the output file as appropriate. If you have older raw data files created in Matlab from input which has been processed Maxfilter, it is necessary to copy the *processing history* block from the original to modified raw data file using the ``mne_copy_processing_history`` utility. The raw data processing programs mne_browse_raw and mne_process_raw have handled copying of the processing history since revision 2.5 of the MNE software.

Some data structures
####################

The MNE Matlab toolbox relies heavily on structures to organize
the data. This section gives detailed information about fields in
the essential data structures employed in the MNE Matlab toolbox.
In the structure definitions, data types referring to other MNE
Matlab toolbox structures are shown in italics. In addition, :ref:`matlab_fif_constants`
lists the values of various FIFF constants defined by fiff_define_constants.m .
The documented structures are:

**tag**

    Contains one tag from the fif file, see :ref:`BGBGIIGD`.

**taginfo**

    Contains the information about one tag, see :ref:`BGBBJBJJ`.

**directory**

    Contains the tag directory as a tree structure, see :ref:`BGBEDHBG`.

**id**

    A fif ID, see :ref:`BGBDAHHJ`.

**named matrix**

    Contains a matrix with names for rows and/or columns, see :ref:`BGBBEDID`.
    A named matrix is used to store, *e.g.*, SSP vectors and forward solutions.

**trans**

    A 4 x 4 coordinate-transformation matrix operating on augmented column
    vectors. Indication of the coordinate frames to which this transformation
    relates is included, see :ref:`BGBDHBIF`.

**dig**

    A Polhemus digitizer data point, see :ref:`BGBHDEDG`.

**coildef**

    The coil definition structure useful for forward calculations and array
    visualization, see :ref:`BGBGBEBH`. For more detailed information on
    coil definitions, see :ref:`coil_geometry_information`.

**ch**

    Channel information structure, see :ref:`BGBIABGD`.

**proj**

    Signal-space projection data, see :ref:`BGBCJHJB`.

**comp**

    Software gradiometer compensation data, see :ref:`BGBJDIFD`.

**measurement info**

    Translation of the FIFFB_MEAS_INFO entity, see :ref:`BGBFHDIJ`. This
    data structure is returned by fiff_read_meas_info .

**surf**

    Used to represent triangulated surfaces and cortical source spaces, see :ref:`BGBEFJCB`.

**cov**

    Used for storing covariance matrices, see :ref:`BGBJJIED`.

**fwd**

    Forward solution data returned by mne_read_forward_solution ,
    see :ref:`BGBFJIBJ`.

**inv**

    Inverse operator decomposition data returned by mne_read_inverse_operator.
    For more information on inverse operator
    decomposition, see :ref:`minimum_norm_estimates`. For an example on how to
    compute inverse solution using this data, see the sample routine mne_ex_compute_inverse .

.. note:: The MNE Matlab toolbox tries it best to employ vertex numbering starting from 1 as opposed to 0 as recorded in the data files. There are, however, two exceptions where explicit attention to the vertex numbering convention is needed. First, the standard stc and w file reading and writing routines return and    assume zero-based vertex numbering. There are now versions with names ending with '1', which return and assume one-based vertex numbering, see :ref:`BABBDDAI`. Second, the logno field of the channel information in the data files produced by mne_compute_raw_inverse is the zero-based number of the vertex whose source space signal is contained on this channel.


.. tabularcolumns:: |p{0.38\linewidth}|p{0.06\linewidth}|p{0.46\linewidth}|
.. _matlab_fif_constants:
.. table:: FIFF constants.

    +-------------------------------+-------+----------------------------------------------------------+
    | Name                          | Value | Purpose                                                  |
    +-------------------------------+-------+----------------------------------------------------------+
    | FIFFV_MEG_CH                  | 1     | This is a MEG channel.                                   |
    +-------------------------------+-------+----------------------------------------------------------+
    | FIFFV_REF_MEG_CH              | 301   | This a reference MEG channel, located far away from the  |
    |                               |       | head.                                                    |
    +-------------------------------+-------+----------------------------------------------------------+
    | FIFFV_EEF_CH                  | 2     | This is an EEG channel.                                  |
    +-------------------------------+-------+----------------------------------------------------------+
    | FIFFV_MCG_CH                  | 201   | This a MCG channel.                                      |
    +-------------------------------+-------+----------------------------------------------------------+
    | FIFFV_STIM_CH                 | 3     | This is a digital trigger channel.                       |
    +-------------------------------+-------+----------------------------------------------------------+
    | FIFFV_EOG_CH                  | 202   | This is an EOG channel.                                  |
    +-------------------------------+-------+----------------------------------------------------------+
    | FIFFV_EMG_CH                  | 302   | This is an EMG channel.                                  |
    +-------------------------------+-------+----------------------------------------------------------+
    | FIFFV_ECG_CH                  | 402   | This is an ECG channel.                                  |
    +-------------------------------+-------+----------------------------------------------------------+
    | FIFFV_MISC_CH                 | 502   | This is a miscellaneous analog channel.                  |
    +-------------------------------+-------+----------------------------------------------------------+
    | FIFFV_RESP_CH                 | 602   | This channel contains respiration monitor output.        |
    +-------------------------------+-------+----------------------------------------------------------+
    | FIFFV_COORD_UNKNOWN           | 0     | Unknown coordinate frame.                                |
    +-------------------------------+-------+----------------------------------------------------------+
    | FIFFV_COORD_DEVICE            | 1     | The MEG device coordinate frame.                         |
    +-------------------------------+-------+----------------------------------------------------------+
    | FIFFV_COORD_ISOTRAK           | 2     | The Polhemus digitizer coordinate frame (does not appear |
    |                               |       | in data files).                                          |
    +-------------------------------+-------+----------------------------------------------------------+
    | FIFFV_COORD_HPI               | 3     | HPI coil coordinate frame (does not appear in data       |
    |                               |       | files).                                                  |
    +-------------------------------+-------+----------------------------------------------------------+
    | FIFFV_COORD_HEAD              | 4     | The MEG head coordinate frame (Neuromag convention).     |
    +-------------------------------+-------+----------------------------------------------------------+
    | FIFFV_COORD_MRI               | 5     | The MRI coordinate frame.                                |
    +-------------------------------+-------+----------------------------------------------------------+
    | FIFFV_COORD_MRI_SLICE         | 6     | The coordinate frame of a single MRI slice.              |
    +-------------------------------+-------+----------------------------------------------------------+
    | FIFFV_COORD_MRI_DISPLAY       | 7     | The preferred coordinate frame for displaying the MRIs   |
    |                               |       | (used by MRIlab).                                        |
    +-------------------------------+-------+----------------------------------------------------------+
    | FIFFV_COORD_DICOM_DEVICE      | 8     | The DICOM coordinate frame (does not appear in files).   |
    +-------------------------------+-------+----------------------------------------------------------+
    | FIFFV_COORD_IMAGING_DEVICE    | 9     | A generic imaging device coordinate frame (does not      |
    |                               |       | appear in files).                                        |
    +-------------------------------+-------+----------------------------------------------------------+
    | FIFFV_MNE_COORD_TUFTS_EEG     | 300   | The Tufts EEG data coordinate frame.                     |
    +-------------------------------+-------+----------------------------------------------------------+
    | FIFFV_MNE_COORD_CTF_DEVICE    | 1001  | The CTF device coordinate frame (does not appear in      |
    |                               |       | files).                                                  |
    +-------------------------------+-------+----------------------------------------------------------+
    | FIFFV_MNE_COORD_CTF_HEAD      | 1004  | The CTF/4D head coordinate frame.                        |
    +-------------------------------+-------+----------------------------------------------------------+
    | FIFFV_ASPECT_AVERAGE          | 100   | Data aspect: average.                                    |
    +-------------------------------+-------+----------------------------------------------------------+
    | FIFFV_ASPECT_STD_ERR          | 101   | Data aspect: standard error of mean.                     |
    +-------------------------------+-------+----------------------------------------------------------+
    | FIFFV_ASPECT_SINGLE           | 102   | Single epoch.                                            |
    +-------------------------------+-------+----------------------------------------------------------+
    | FIFFV_ASPECT_SUBAVERAGE       | 103   | One subaverage.                                          |
    +-------------------------------+-------+----------------------------------------------------------+
    | FIFFV_ASPECT_ALTAVERAGE       | 104   | One alternating (plus-minus) subaverage.                 |
    +-------------------------------+-------+----------------------------------------------------------+
    | FIFFV_ASPECT_SAMPLE           | 105   | A sample cut from raw data.                              |
    +-------------------------------+-------+----------------------------------------------------------+
    | FIFFV_ASPECT_POWER_DENSITY    | 106   | Power density spectrum.                                  |
    +-------------------------------+-------+----------------------------------------------------------+
    | FIFFV_ASPECT_DIPOLE_WAVE      | 200   | The time course of an equivalent current dipole.         |
    +-------------------------------+-------+----------------------------------------------------------+
    | FIFFV_BEM_SURF_ID_UNKNOWN     | -1    | Unknown BEM surface.                                     |
    +-------------------------------+-------+----------------------------------------------------------+
    | FIFFV_BEM_SURF_ID_BRAIN       | 1     | The inner skull surface                                  |
    +-------------------------------+-------+----------------------------------------------------------+
    | FIFFV_BEM_SURF_ID_SKULL       | 3     | The outer skull surface                                  |
    +-------------------------------+-------+----------------------------------------------------------+
    | FIFFV_BEM_SURF_ID_HEAD        | 4     | The scalp surface                                        |
    +-------------------------------+-------+----------------------------------------------------------+
    | FIFFV_MNE_SURF_LEFT_HEMI      | 101   | Left hemisphere cortical surface                         |
    +-------------------------------+-------+----------------------------------------------------------+
    | FIFFV_MNE_SURF_RIGHT_HEMI     | 102   | Right hemisphere cortical surface                        |
    +-------------------------------+-------+----------------------------------------------------------+
    | FIFFV_POINT_CARDINAL          | 1     | Digitization point which is a cardinal landmark a.k.a.   |
    |                               |       | fiducial point                                           |
    +-------------------------------+-------+----------------------------------------------------------+
    | FIFFV_POINT_HPI               | 2     | Digitized HPI coil location                              |
    +-------------------------------+-------+----------------------------------------------------------+
    | FIFFV_POINT_EEG               | 3     | Digitized EEG electrode location                         |
    +-------------------------------+-------+----------------------------------------------------------+
    | FIFFV_POINT_ECG               | 3     | Digitized ECG electrode location                         |
    +-------------------------------+-------+----------------------------------------------------------+
    | FIFFV_POINT_EXTRA             | 4     | Additional head surface point                            |
    +-------------------------------+-------+----------------------------------------------------------+
    | FIFFV_POINT_LPA               | 1     | Identifier for left auricular landmark                   |
    +-------------------------------+-------+----------------------------------------------------------+
    | FIFFV_POINT_NASION            | 2     | Identifier for nasion                                    |
    +-------------------------------+-------+----------------------------------------------------------+
    | FIFFV_POINT_RPA               | 3     | Identifier for right auricular landmark                  |
    +-------------------------------+-------+----------------------------------------------------------+
    | FIFFV_MNE_FIXED_ORI           | 1     | Fixed orientation constraint used in the computation of  |
    |                               |       | a forward solution.                                      |
    +-------------------------------+-------+----------------------------------------------------------+
    | FIFFV_MNE_FREE_ORI            | 2     | No orientation constraint used in the computation of     |
    |                               |       | a forward solution                                       |
    +-------------------------------+-------+----------------------------------------------------------+
    | FIFFV_MNE_MEG                 | 1     | Indicates an inverse operator based on MEG only          |
    +-------------------------------+-------+----------------------------------------------------------+
    | FIFFV_MNE_EEG                 | 2     | Indicates an inverse operator based on EEG only.         |
    +-------------------------------+-------+----------------------------------------------------------+
    | FIFFV_MNE_MEG_EEG             | 3     | Indicates an inverse operator based on both MEG and EEG. |
    +-------------------------------+-------+----------------------------------------------------------+
    | FIFFV_MNE_UNKNOWN_COV         | 0     | An unknown covariance matrix                             |
    +-------------------------------+-------+----------------------------------------------------------+
    | FIFFV_MNE_NOISE_COV           | 1     | Indicates a noise covariance matrix.                     |
    +-------------------------------+-------+----------------------------------------------------------+
    | FIFFV_MNE_SENSOR_COV          | 1     | Synonym for FIFFV_MNE_NOISE_COV                          |
    +-------------------------------+-------+----------------------------------------------------------+
    | FIFFV_MNE_SOURCE_COV          | 2     | Indicates a source covariance matrix                     |
    +-------------------------------+-------+----------------------------------------------------------+
    | FIFFV_MNE_FMRI_PRIOR_COV      | 3     | Indicates a covariance matrix associated with fMRI priors|
    +-------------------------------+-------+----------------------------------------------------------+
    | FIFFV_MNE_SIGNAL_COV          | 4     | Indicates the data (signal + noise) covariance matrix    |
    +-------------------------------+-------+----------------------------------------------------------+
    | FIFFV_MNE_DEPTH_PRIOR_COV     | 5     | Indicates the depth prior (depth weighting) covariance   |
    |                               |       | matrix                                                   |
    +-------------------------------+-------+----------------------------------------------------------+
    | FIFFV_MNE_ORIENT_PRIOR_COV    | 6     | Indicates the orientation (loose orientation constrain)  |
    |                               |       | prior covariance matrix                                  |
    +-------------------------------+-------+----------------------------------------------------------+
    | FIFFV_PROJ_ITEM_NONE          | 0     | The nature of this projection item is unknown            |
    +-------------------------------+-------+----------------------------------------------------------+
    | FIFFV_PROJ_ITEM_FIELD         | 1     | This is projection item is a generic field pattern or    |
    |                               |       | field patters.                                           |
    +-------------------------------+-------+----------------------------------------------------------+
    | FIFFV_PROJ_ITEM_DIP_FIX       | 2     | This projection item is the field of one dipole          |
    +-------------------------------+-------+----------------------------------------------------------+
    | FIFFV_PROJ_ITEM_DIP_ROT       | 3     | This projection item corresponds to the fields of three  |
    |                               |       | or two orthogonal dipoles at some location.              |
    +-------------------------------+-------+----------------------------------------------------------+
    | FIFFV_PROJ_ITEM_HOMOG_GRAD    | 4     | This projection item contains the homogeneous gradient   |
    |                               |       | fields as seen by the sensor array.                      |
    +-------------------------------+-------+----------------------------------------------------------+
    | FIFFV_PROJ_ITEM_HOMOG_FIELD   | 5     | This projection item contains the three homogeneous field|
    |                               |       | components as seen by the sensor array.                  |
    +-------------------------------+-------+----------------------------------------------------------+
    | FIFFV_PROJ_ITEM_EEG_AVREF     | 10    | This projection item corresponds to the average EEG      |
    |                               |       | reference.                                               |
    +-------------------------------+-------+----------------------------------------------------------+

.. _BGBGIIGD:

.. table:: The tag structure.

    =======  ===========  ============================================
    Field    Data type    Description
    =======  ===========  ============================================
    kind     int32        The kind of the data item.
    type     uint32       The data type used to represent the data.
    size     int32        Size of the data in bytes.
    next     int32        Byte offset of the next tag in the file.
    data     various      The data itself.
    =======  ===========  ============================================

.. _BGBBJBJJ:

.. table:: The taginfo structure.

    =======  ===========  ============================================
    Field    Data type    Description
    =======  ===========  ============================================
    kind     double       The kind of the data item.
    type     double       The data type used to represent the data.
    size     double       Size of the data in bytes.
    pos      double       Byte offset to this tag in the file.
    =======  ===========  ============================================

.. _BGBEDHBG:

.. table:: The directory structure.

    ============  ============  ================================================================
    Field         Data type     Description
    ============  ============  ================================================================
    block         double        The block id of this directory node.
    id            id            The unique identifier of this node.
    parent_id     id            The unique identifier of the node this node was derived from.
    nent          double        Number of entries in this node.
    nchild        double        Number of children to this node.
    dir           taginfo       Information about tags in this node.
    children      directory     The children of this node.
    ============  ============  ================================================================

.. _BGBDAHHJ:

.. table:: The id structure.

    ==========  ===========  ============================================================
    Field       Data type    Description
    ==========  ===========  ============================================================
    version     int32        The fif file version (major  < < 16 | minor).
    machid      int32(2)     Unique identifier of the computer this id was created on.
    secs        int32        Time since January 1, 1970 (seconds).
    usecs       int32        Time since January 1, 1970 (microseconds past secs ).
    ==========  ===========  ============================================================

.. _BGBBEDID:

.. table:: The named matrix structure.

    ============  ===========  ======================================================================
    Field         Data type    Description
    ============  ===========  ======================================================================
    nrow          int32        Number of rows.
    ncol          int32        Number of columns.
    row_names     cell(*)      The names of associated with the rows. This member may be empty.
    col_names     cell(*)      The names of associated with the columns. This member may be empty.
    data          various      The matrix data, usually of type single or double.
    ============  ===========  ======================================================================


.. tabularcolumns:: |p{0.2\linewidth}|p{0.2\linewidth}|p{0.55\linewidth}|
.. _BGBDHBIF:
.. table:: The trans structure.

    +---------------------------+-----------+----------------------------------------------------------+
    | Field                     | Data Type | Description                                              |
    +===========================+===========+==========================================================+
    | from                      | int32     | The source coordinate frame, see                         |
    |                           |           | :ref:`matlab_fif_constants`. Look                        |
    |                           |           | for entries starting with FIFFV_COORD or FIFFV_MNE_COORD.|
    +---------------------------+-----------+----------------------------------------------------------+
    | to                        | int32     | The destination coordinate frame.                        |
    +---------------------------+-----------+----------------------------------------------------------+
    | trans                     |double(4,4)| The 4-by-4 coordinate transformation matrix. This        |
    |                           |           | operates from augmented position column vectors given in |
    |                           |           | *from* coordinates to give results in *to* coordinates.  |
    +---------------------------+-----------+----------------------------------------------------------+


.. tabularcolumns:: |p{0.2\linewidth}|p{0.2\linewidth}|p{0.55\linewidth}|
.. _BGBHDEDG:
.. table:: The dig structure.

    +---------------------------+-----------+----------------------------------------------------------+
    | Field                     | Data Type | Description                                              |
    +===========================+===========+==========================================================+
    | kind                      | int32     | The type of digitizing point. Possible values are listed |
    |                           |           | in :ref:`matlab_fif_constants`. Look for entries         |
    |                           |           | starting with FIFF_POINT.                                |
    +---------------------------+-----------+----------------------------------------------------------+
    | ident                     | int32     | Identifier for this point.                               |
    +---------------------------+-----------+----------------------------------------------------------+
    | r                         | single(3) | The location of this point.                              |
    +---------------------------+-----------+----------------------------------------------------------+


.. tabularcolumns:: |p{0.2\linewidth}|p{0.2\linewidth}|p{0.55\linewidth}|
.. _BGBGBEBH:
.. table:: The coildef structure. For more detailed information, see :ref:`coil_geometry_information`.

    +-------------------+-------------------+----------------------------------------------------------+
    | Field             | Data Type         | Description                                              |
    +===================+===================+==========================================================+
    | class             | double            | The coil (or electrode) class.                           |
    +-------------------+-------------------+----------------------------------------------------------+
    | id                | double            | The coil (or electrode) id.                              |
    +-------------------+-------------------+----------------------------------------------------------+
    | accuracy          | double            | Representation accuracy.                                 |
    +-------------------+-------------------+----------------------------------------------------------+
    | num_points        | double            | Number of integration points.                            |
    +-------------------+-------------------+----------------------------------------------------------+
    | size              | double            | Coil size.                                               |
    +-------------------+-------------------+----------------------------------------------------------+
    | baseline          | double            | Coil baseline.                                           |
    +-------------------+-------------------+----------------------------------------------------------+
    | description       | char(*)           | Coil description.                                        |
    +-------------------+-------------------+----------------------------------------------------------+
    | coildefs          | double            | Each row contains the integration point weight, followed |
    |                   | (num_points,7)    | by location [m] and normal.                              |
    +-------------------+-------------------+----------------------------------------------------------+
    | FV                | struct            | Contains the faces and vertices which can be used to     |
    |                   |                   | draw the coil for visualization.                         |
    +-------------------+-------------------+----------------------------------------------------------+


.. tabularcolumns:: |p{0.2\linewidth}|p{0.2\linewidth}|p{0.55\linewidth}|
.. _BGBIABGD:
.. table:: The ch structure.

    +---------------------------+-----------+----------------------------------------------------------+
    | Field                     | Data Type | Description                                              |
    +===========================+===========+==========================================================+
    | scanno                    | int32     | Scanning order number, starting from 1.                  |
    +---------------------------+-----------+----------------------------------------------------------+
    | logno                     | int32     | Logical channel number, conventions in the usage of this |
    |                           |           | number vary.                                             |
    +---------------------------+-----------+----------------------------------------------------------+
    | kind                      | int32     | The channel type (FIFFV_MEG_CH, FIFF_EEG_CH, etc., see   |
    |                           |           | :ref:`matlab_fif_constants` ).                           |
    +---------------------------+-----------+----------------------------------------------------------+
    | range                     | double    | The hardware-oriented part of the calibration factor.    |
    |                           |           | This should be only applied to the continuous raw data.  |
    +---------------------------+-----------+----------------------------------------------------------+
    | cal                       | double    | The calibration factor to bring the channels to physical |
    |                           |           | units.                                                   |
    +---------------------------+-----------+----------------------------------------------------------+
    | loc                       | double(12)| The channel location. The first three numbers indicate   |
    |                           |           | the location [m], followed by the three unit vectors of  |
    |                           |           | the channel-specific coordinate frame. These data contain|
    |                           |           | the values saved in the fif file and should not be       |
    |                           |           | changed. The values are specified in device coordinates  |
    |                           |           | for MEG and in head coordinates for EEG channels,        |
    |                           |           | respectively.                                            |
    +---------------------------+-----------+----------------------------------------------------------+
    | coil_trans                |double(4,4)| Initially, transformation from the channel coordinates   |
    |                           |           | to device coordinates. This transformation is updated by |
    |                           |           | calls to fiff_transform_meg_chs and                      |
    |                           |           | fiff_transform_eeg_chs.                                  |
    +---------------------------+-----------+----------------------------------------------------------+
    | eeg_loc                   | double(6) | The location of the EEG electrode in coord_frame         |
    |                           |           | coordinates. The first three values contain the location |
    |                           |           | of the electrode [m]. If six values are present, the     |
    |                           |           | remaining ones indicate the location of the reference    |
    |                           |           | electrode for this channel.                              |
    +---------------------------+-----------+----------------------------------------------------------+
    | coord_frame               | int32     | Initially, the coordinate frame is FIFFV_COORD_DEVICE    |
    |                           |           | for MEG channels and FIFFV_COORD_HEAD for EEG channels.  |
    +---------------------------+-----------+----------------------------------------------------------+
    | unit                      | int32     | Unit of measurement. Relevant values are: 201 = T/m,     |
    |                           |           | 112 = T, 107 = V, and 202 = Am.                          |
    +---------------------------+-----------+----------------------------------------------------------+
    | unit_mul                  | int32     | The data are given in unit s multiplied by 10unit_mul.   |
    |                           |           | Presently, unit_mul is always zero.                      |
    +---------------------------+-----------+----------------------------------------------------------+
    | ch_name                   | char(*)   | Name of the channel.                                     |
    +---------------------------+-----------+----------------------------------------------------------+
    | coil_def                  | coildef   | The coil definition structure. This is present only if   |
    |                           |           | mne_add_coil_defs has been successfully called.          |
    +---------------------------+-----------+----------------------------------------------------------+


.. tabularcolumns:: |p{0.2\linewidth}|p{0.2\linewidth}|p{0.55\linewidth}|
.. _BGBCJHJB:
.. table:: The proj structure.

    +---------------------------+-----------+----------------------------------------------------------+
    | Field                     | Data Type | Description                                              |
    +===========================+===========+==========================================================+
    | kind                      | int32     | The type of the projection item. Possible values are     |
    |                           |           | listed in :ref:`matlab_fif_constants`. Look for entries  |
    |                           |           | starting with FIFFV_PROJ_ITEM or FIFFV_MNE_PROJ_ITEM.    |
    +---------------------------+-----------+----------------------------------------------------------+
    | active                    | int32     | Is this item active, i.e., applied or about to be        |
    |                           |           | applied to the data.                                     |
    +---------------------------+-----------+----------------------------------------------------------+
    | data                      | named     | The projection vectors. The column names indicate the    |
    |                           | matrix    | names of the channels associated to the elements of the  |
    |                           |           | vectors.                                                 |
    +---------------------------+-----------+----------------------------------------------------------+



.. tabularcolumns:: |p{0.2\linewidth}|p{0.2\linewidth}|p{0.55\linewidth}|
.. _BGBJDIFD:
.. table:: The comp structure.

    +---------------------------+-----------+----------------------------------------------------------+
    | Field                     | Data Type | Description                                              |
    +===========================+===========+==========================================================+
    | ctfkind                   | int32     | The kind of the compensation as stored in file.          |
    +---------------------------+-----------+----------------------------------------------------------+
    | kind                      | int32     | ctfkind mapped into small integer numbers.               |
    +---------------------------+-----------+----------------------------------------------------------+
    | save_calibrated           | logical   | Were the compensation data saved in calibrated form. If  |
    |                           |           | this field is false, the matrix will be decalibrated     |
    |                           |           | using the fields row_cals and col_cals when the          |
    |                           |           | compensation data are saved by the toolbox.              |
    +---------------------------+-----------+----------------------------------------------------------+
    | row_cals                  | double(*) | Calibration factors applied to the rows of the           |
    |                           |           | compensation data matrix when the data were read.        |
    +---------------------------+-----------+----------------------------------------------------------+
    | col_cals                  | double(*) | Calibration factors applied to the columns of the        |
    |                           |           | compensation data matrix when the data were read.        |
    +---------------------------+-----------+----------------------------------------------------------+
    | data                      | named     | The compensation data matrix. The row_names list the     |
    |                           | matrix    | names of the channels to which this compensation applies |
    |                           |           | and the col_names the compensation channels.             |
    +---------------------------+-----------+----------------------------------------------------------+


.. tabularcolumns:: |p{0.2\linewidth}|p{0.2\linewidth}|p{0.55\linewidth}|
.. _BGBFHDIJ:
.. table:: The meas info structure.

    +---------------------------+-----------+----------------------------------------------------------+
    | Field                     | Data Type | Description                                              |
    +===========================+===========+==========================================================+
    | file_id                   | id        | The fif ID of the measurement file.                      |
    +---------------------------+-----------+----------------------------------------------------------+
    | meas_id                   | id        | The ID assigned to this measurement by the acquisition   |
    |                           |           | system or during file conversion.                        |
    +---------------------------+-----------+----------------------------------------------------------+
    | nchan                     | int32     | Number of channels.                                      |
    +---------------------------+-----------+----------------------------------------------------------+
    | sfreq                     | double    | Sampling frequency.                                      |
    +---------------------------+-----------+----------------------------------------------------------+
    | highpass                  | double    | Highpass corner frequency [Hz]. Zero indicates a DC      |
    |                           |           | recording.                                               |
    +---------------------------+-----------+----------------------------------------------------------+
    | lowpass                   | double    | Lowpass corner frequency [Hz].                           |
    +---------------------------+-----------+----------------------------------------------------------+
    | chs                       | ch(nchan) | An array of channel information structures.              |
    +---------------------------+-----------+----------------------------------------------------------+
    | ch_names                  |cell(nchan)| Cell array of channel names.                             |
    +---------------------------+-----------+----------------------------------------------------------+
    | dev_head_t                | trans     | The device to head transformation.                       |
    +---------------------------+-----------+----------------------------------------------------------+
    | ctf_head_t                | trans     | The transformation from 4D/CTF head coordinates to       |
    |                           |           | Neuromag head coordinates. This is only present in       |
    |                           |           | 4D/CTF data.                                             |
    +---------------------------+-----------+----------------------------------------------------------+
    | dev_ctf_t                 | trans     | The transformation from device coordinates to 4D/CTF     |
    |                           |           | head coordinates. This is only present in 4D/CTF data.   |
    +---------------------------+-----------+----------------------------------------------------------+
    | dig                       | dig(*)    | The Polhemus digitization data in head coordinates.      |
    +---------------------------+-----------+----------------------------------------------------------+
    | bads                      | cell(*)   | Bad channel list.                                        |
    +---------------------------+-----------+----------------------------------------------------------+
    | projs                     | proj(*)   | SSP operator data.                                       |
    +---------------------------+-----------+----------------------------------------------------------+
    | comps                     | comp(*)   | Software gradient compensation data.                     |
    +---------------------------+-----------+----------------------------------------------------------+


.. tabularcolumns:: |p{0.2\linewidth}|p{0.2\linewidth}|p{0.55\linewidth}|
.. _BGBEFJCB:

.. table:: The surf structure.

    +---------------------------+-----------+----------------------------------------------------------+
    | Field                     | Data Type | Description                                              |
    +===========================+===========+==========================================================+
    | id                        | int32     | The surface ID.                                          |
    +---------------------------+-----------+----------------------------------------------------------+
    | sigma                     | double    | The electrical conductivity of the compartment bounded by|
    |                           |           | this surface. This field is present in BEM surfaces only.|
    +---------------------------+-----------+----------------------------------------------------------+
    | np                        | int32     | Number of vertices on the surface.                       |
    +---------------------------+-----------+----------------------------------------------------------+
    | ntri                      | int32     | Number of triangles on the surface.                      |
    +---------------------------+-----------+----------------------------------------------------------+
    | coord_frame               | int32     | Coordinate frame in which the locations and orientations |
    |                           |           | are expressed.                                           |
    +---------------------------+-----------+----------------------------------------------------------+
    | rr                        | double    | The vertex locations.                                    |
    |                           | (np,3)    |                                                          |
    +---------------------------+-----------+----------------------------------------------------------+
    | nn                        | double    | The vertex normals. If derived surface data was not      |
    |                           | (np,3)    | requested, this is empty.                                |
    +---------------------------+-----------+----------------------------------------------------------+
    | tris                      | int32     | Vertex numbers of the triangles in counterclockwise      |
    |                           | (ntri,3)  | order as seen from the outside.                          |
    +---------------------------+-----------+----------------------------------------------------------+
    | nuse                      | int32     | Number of active vertices, *i.e.*, vertices included in  |
    |                           |           | a decimated source space.                                |
    +---------------------------+-----------+----------------------------------------------------------+
    | inuse                     | int32(np) | Which vertices are in use.                               |
    +---------------------------+-----------+----------------------------------------------------------+
    | vertno                    |int32(nuse)| Indices of the vertices in use.                          |
    +---------------------------+-----------+----------------------------------------------------------+
    | curv                      | double(np)| Curvature values at the vertices. If curvature           |
    |                           |           | information was not requested, this field is empty or    |
    |                           |           | absent.                                                  |
    +---------------------------+-----------+----------------------------------------------------------+
    | tri_area                  | double    | The triangle areas in m2.If derived surface data was not |
    |                           | (ntri)    | requested, this field will be missing.                   |
    +---------------------------+-----------+----------------------------------------------------------+
    | tri_cent                  | double    | The triangle centroids. If derived surface data was not  |
    |                           | (ntri,3)  | requested, this field will be missing.                   |
    +---------------------------+-----------+----------------------------------------------------------+
    | tri_nn                    | double    | The triangle normals. If derived surface data was not    |
    |                           | (ntri,3)  | requested, this field will be missing.                   |
    +---------------------------+-----------+----------------------------------------------------------+
    | nuse_tri                  | int32     | Number of triangles in use. This is present only if the  |
    |                           |           | surface corresponds to a source space created with the   |
    |                           |           | ``--ico`` option.                                        |
    +---------------------------+-----------+----------------------------------------------------------+
    | use_tris                  | int32     | The vertices of the triangles in use in the complete     |
    |                           | (nuse_tri)| triangulation. This is present only if the surface       |
    |                           |           | corresponds to a source space created with the           |
    |                           |           | ``--ico`` option.                                        |
    +---------------------------+-----------+----------------------------------------------------------+
    | nearest                   | int32(np) | This field is present only if patch information has been |
    |                           |           | computed for a source space. For each vertex in the      |
    |                           |           | triangulation, these values indicate the nearest active  |
    |                           |           | source space vertex.                                     |
    +---------------------------+-----------+----------------------------------------------------------+
    | nearest_dist              | double(np)| This field is present only if patch information has been |
    |                           |           | computed for a source space. For each vertex in the      |
    |                           |           | triangulation, these values indicate the distance to the |
    |                           |           | nearest active source space vertex.                      |
    +---------------------------+-----------+----------------------------------------------------------+
    | dist                      | double    | Distances between vertices on this surface given as a    |
    |                           | (np,np)   | sparse matrix. A zero off-diagonal entry in this matrix  |
    |                           |           | indicates that the corresponding distance has not been   |
    |                           |           | calculated.                                              |
    +---------------------------+-----------+----------------------------------------------------------+
    | dist_limit                | double    | The value given to mne_add_patch_info with the ``--dist``|
    |                           |           | option. This value is presently                          |
    |                           |           | always negative, indicating that only distances between  |
    |                           |           | active source space vertices, as indicated by the vertno |
    |                           |           | field of this structure, have been calculated.           |
    +---------------------------+-----------+----------------------------------------------------------+


.. tabularcolumns:: |p{0.2\linewidth}|p{0.2\linewidth}|p{0.55\linewidth}|
.. _BGBJJIED:

.. table:: The cov structure.

    +---------------------------+-----------+----------------------------------------------------------+
    | Field                     | Data Type | Description                                              |
    +===========================+===========+==========================================================+
    | kind                      | double    | What kind of a covariance matrix (1 = noise covariance,  |
    |                           |           | 2 = source covariance).                                  |
    +---------------------------+-----------+----------------------------------------------------------+
    | diag                      | double    | Is this a diagonal matrix.                               |
    +---------------------------+-----------+----------------------------------------------------------+
    | dim                       | int32     | Dimension of the covariance matrix.                      |
    +---------------------------+-----------+----------------------------------------------------------+
    | names                     | cell(*)   | Names of the channels associated with the entries        |
    |                           |           | (may be empty).                                          |
    +---------------------------+-----------+----------------------------------------------------------+
    | data                      | double    | The covariance matrix. This a double(dim) vector for a   |
    |                           | (dim,dim) | diagonal covariance matrix.                              |
    +---------------------------+-----------+----------------------------------------------------------+
    | projs                     | proj(*)   | The SSP vectors applied to these data.                   |
    +---------------------------+-----------+----------------------------------------------------------+
    | bads                      | cell(*)   | Bad channel names.                                       |
    +---------------------------+-----------+----------------------------------------------------------+
    | nfree                     | int32     | Number of data points used to compute this matrix.       |
    +---------------------------+-----------+----------------------------------------------------------+
    | eig                       |double(dim)| The eigenvalues of the covariance matrix. This field may |
    |                           |           | be empty for a diagonal covariance matrix.               |
    +---------------------------+-----------+----------------------------------------------------------+
    | eigvec                    | double    | The eigenvectors of the covariance matrix.               |
    |                           | (dim,dim) |                                                          |
    +---------------------------+-----------+----------------------------------------------------------+


.. tabularcolumns:: |p{0.2\linewidth}|p{0.2\linewidth}|p{0.55\linewidth}|
.. _BGBFJIBJ:

.. table:: The fwd structure.

    +-------------------------+-------------+----------------------------------------------------------+
    | Field                   | Data Type   | Description                                              |
    +=========================+=============+==========================================================+
    | source_ori              | int32       | Has the solution been computed for the current component |
    |                         |             | normal to the cortex only (1) or all three source        |
    |                         |             | orientations (2).                                        |
    +-------------------------+-------------+----------------------------------------------------------+
    | coord_frame             | int32       | Coordinate frame in which the locations and orientations |
    |                         |             | are expressed.                                           |
    +-------------------------+-------------+----------------------------------------------------------+
    | nsource                 | int32       | Total number of source space points.                     |
    +-------------------------+-------------+----------------------------------------------------------+
    | nchan                   | int32       | Number of channels.                                      |
    +-------------------------+-------------+----------------------------------------------------------+
    | sol                     | named       | The forward solution matrix.                             |
    |                         | matrix      |                                                          |
    +-------------------------+-------------+----------------------------------------------------------+
    | sol_grad                | named       | The derivatives of the forward solution with respect to  |
    |                         | matrix      | the dipole location coordinates.                         |
    |                         |             | This field is present only if the forward solution was   |
    |                         |             | computed with the ``--grad`` option in MNE-C.            |
    +-------------------------+-------------+----------------------------------------------------------+
    | mri_head_t              | trans       | Transformation from the MRI coordinate frame to the      |
    |                         |             | (Neuromag) head coordinate frame.                        |
    +-------------------------+-------------+----------------------------------------------------------+
    | src                     | surf(:)     | The description of the source spaces.                    |
    +-------------------------+-------------+----------------------------------------------------------+
    | source_rr               | double      | The source locations.                                    |
    |                         | (nsource,3) |                                                          |
    +-------------------------+-------------+----------------------------------------------------------+
    | source_nn               | double(:,3) | The source orientations. Number of rows is either        |
    |                         |             | nsource (fixed source orientations) or 3*nsource         |
    |                         |             | (all source orientations).                               |
    +-------------------------+-------------+----------------------------------------------------------+


.. tabularcolumns:: |p{0.2\linewidth}|p{0.2\linewidth}|p{0.55\linewidth}|
.. _BGBIEIJE:

.. table:: The inv structure. Note: The fields proj, whitener, reginv, and noisenorm are filled in by the routine mne_prepare_inverse_operator.

    +---------------------+-------------+----------------------------------------------------------+
    | Field               | Data Type   | Description                                              |
    +=====================+=============+==========================================================+
    | methods             | int32       | Has the solution been computed using MEG data (1), EEG   |
    |                     |             | data (2), or both (3).                                   |
    +---------------------+-------------+----------------------------------------------------------+
    | source_ori          | int32       | Has the solution been computed for the current component |
    |                     |             | normal to the cortex only (1) or all three source        |
    |                     |             | orientations (2).                                        |
    +---------------------+-------------+----------------------------------------------------------+
    | nsource             | int32       | Total number of source space points.                     |
    +---------------------+-------------+----------------------------------------------------------+
    | nchan               | int32       | Number of channels.                                      |
    +---------------------+-------------+----------------------------------------------------------+
    | coord_frame         | int32       | Coordinate frame in which the locations and orientations |
    |                     |             | are expressed.                                           |
    +---------------------+-------------+----------------------------------------------------------+
    | source_nn           | double(:,3) | The source orientations. Number of rows is either        |
    |                     |             | nsource (fixed source orientations) or 3*nsource (all    |
    |                     |             | source orientations).                                    |
    +---------------------+-------------+----------------------------------------------------------+
    | sing                | double      | The singular values, *i.e.*, the diagonal values of      |
    |                     | (nchan)     | :math:`\Lambda`, see :ref:`mne_solution`.                |
    +---------------------+-------------+----------------------------------------------------------+
    | eigen_leads         | double      | The matrix :math:`V`, see :ref:`mne_solution`.           |
    |                     | (:,nchan)   |                                                          |
    +---------------------+-------------+----------------------------------------------------------+
    | eigen_fields        | double      | The matrix :math:`U^\top`, see                           |
    |                     | (nchan,     | :ref:`mne_solution`.                                     |
    |                     | nchan)      |                                                          |
    +---------------------+-------------+----------------------------------------------------------+
    | noise_cov           | cov         | The noise covariance matrix :math:`C`.                   |
    +---------------------+-------------+----------------------------------------------------------+
    | source_cov          | cov         | The source covariance matrix :math:`R`.                  |
    +---------------------+-------------+----------------------------------------------------------+
    | src                 | surf(:)     | The description of the source spaces.                    |
    +---------------------+-------------+----------------------------------------------------------+
    | mri_head_t          | trans       | Transformation from the MRI coordinate frame to the      |
    |                     |             | (Neuromag) head coordinate frame.                        |
    +---------------------+-------------+----------------------------------------------------------+
    | nave                | double      | The number of averages.                                  |
    +---------------------+-------------+----------------------------------------------------------+
    | projs               | proj(:)     | The SSP vectors which were active when the decomposition |
    |                     |             | was computed.                                            |
    +---------------------+-------------+----------------------------------------------------------+
    | proj                | double      | The projection operator computed using projs.            |
    |                     | (nchan)     |                                                          |
    +---------------------+-------------+----------------------------------------------------------+
    | whitener            |             | A sparse matrix containing the noise normalization       |
    |                     |             | factors. Dimension is either nsource (fixed source       |
    |                     |             | orientations) or 3*nsource (all source orientations).    |
    +---------------------+-------------+----------------------------------------------------------+
    | reginv              | double      | The diagonal matrix :math:`\Gamma`, see                  |
    |                     | (nchan)     | :ref:`mne_solution`.                                     |
    +---------------------+-------------+----------------------------------------------------------+
    | noisenorm           | double(:)   | A sparse matrix containing the noise normalization       |
    |                     |             | factors. Dimension is either nsource (fixed source       |
    |                     |             | orientations) or 3*nsource (all source orientations).    |
    +---------------------+-------------+----------------------------------------------------------+


On-line documentation for individual routines
#############################################

Each of the routines listed in Tables :ref:`BGBCGHAG` - :ref:`BGBEFADJ` has on-line documentation accessible by saying ``help`` <*routine name*> in Matlab.
