

.. _c_reference:

===============
C API Reference
===============

Note that most programs have the options ``--version`` and ``--help`` which
give the version information and usage information, respectively.

.. contents:: Contents
   :local:
   :depth: 2


List of components
##################

Software components
===================

.. tabularcolumns:: |p{0.3\linewidth}|p{0.65\linewidth}|
.. table::


    +----------------------------+--------------------------------------------+
    | Name                       |   Purpose                                  |
    +============================+============================================+
    | `mne_analyze`_             | An interactive analysis tool for computing |
    |                            | source estimates, see                      |
    |                            | :ref:`ch_interactive_analysis`.            |
    +----------------------------+--------------------------------------------+
    | `mne_average_estimates`_   | Average data across subjects.              |
    +----------------------------+--------------------------------------------+
    | `mne_browse_raw`_          | Interactive raw data browser. Includes     |
    |                            | filtering, offline averaging, and          |
    |                            | computation of covariance matrices,        |
    |                            | see :ref:`ch_browse`.                      |
    +----------------------------+--------------------------------------------+
    | `mne_compute_mne`_         | Computes the minimum-norm estimates,       |
    |                            | Most functionality is included in          |
    |                            | :ref:`mne_make_movie`.                     |
    +----------------------------+--------------------------------------------+
    | `mne_compute_raw_inverse`_ | Compute the inverse solution from raw data |
    |                            | see :ref:`computing_inverse`.              |
    +----------------------------+--------------------------------------------+
    | `mne_convert_mne_data`_    | Convert MNE data files to other file       |
    |                            | formats.                                   |
    +----------------------------+--------------------------------------------+
    | `mne_do_forward_solution`_ | Convenience script to calculate the forward|
    |                            | solution matrix, see :ref:`BABCHEJD`.      |
    +----------------------------+--------------------------------------------+
    | `mne_do_inverse_operator`_ | Convenience script for inverse operator    |
    |                            | decomposition, see :ref:`CIHCFJEI`.        |
    +----------------------------+--------------------------------------------+
    | `mne_forward_solution`_    | Calculate the forward solution matrix, see |
    |                            | :ref:`CHDDIBAH`.                           |
    +----------------------------+--------------------------------------------+
    | `mne_inverse_operator`_    | Compute the inverse operator decomposition |
    |                            | see :ref:`inverse_operator`.               |
    +----------------------------+--------------------------------------------+
    | `mne_make_movie`_          | Make movies in batch mode, see             |
    |                            | :ref:`movies_and_snapshots`.               |
    +----------------------------+--------------------------------------------+
    | `mne_make_source_space`_   | Create a *fif* source space description    |
    |                            | file, see :ref:`BEHCGJDD`.                 |
    +----------------------------+--------------------------------------------+
    | `mne_process_raw`_         | A batch-mode version of mne_browse_raw,    |
    |                            | see :ref:`ch_browse`.                      |
    +----------------------------+--------------------------------------------+
    | `mne_redo_file`_           | Many intermediate result files contain a   |
    |                            | description of their                       |
    |                            | 'production environment'. Such files can   |
    |                            | be recreated easily with this utility.     |
    |                            | This is convenient if, for example,        |
    |                            | the selection of bad channels is changed   |
    |                            | and the inverse operator decomposition has |
    |                            | to be recalculated.                        |
    +----------------------------+--------------------------------------------+
    | `mne_redo_file_nocwd`_     | Works like mne_redo_file but does not try  |
    |                            | to change in to the working directory      |
    |                            | specified in the 'production environment'. |
    +----------------------------+--------------------------------------------+
    | `mne_setup_forward_model`_ | Set up the BEM-related fif files,          |
    |                            | see :ref:`CIHDBFEG`.                       |
    +----------------------------+--------------------------------------------+
    | `mne_setup_mri`_           | A convenience script to create the fif     |
    |                            | files describing the anatomical MRI data,  |
    |                            | see :ref:`BABCCEHF`                        |
    +----------------------------+--------------------------------------------+
    | `mne_setup_source_space`_  | A convenience script to create source space|
    |                            | description file, see                      |
    |                            | :ref:`setting_up_source_space`.            |
    +----------------------------+--------------------------------------------+
    | `mne_show_environment`_    | Show information about the production      |
    |                            | environment of a file.                     |
    +----------------------------+--------------------------------------------+


.. _ch_misc:

Utilities
=========

.. tabularcolumns:: |p{0.3\linewidth}|p{0.65\linewidth}|
.. _BABDJHGH:
.. table::

    +----------------------------------+--------------------------------------------+
    | Name                             |   Purpose                                  |
    +==================================+============================================+
    | `mne_add_patch_info`_            | Add neighborhood information to a source   |
    |                                  | space file.                                |
    +----------------------------------+--------------------------------------------+
    | `mne_add_to_meas_info`_          | Utility to add new information to the      |
    |                                  | measurement info block of a fif file. The  |
    |                                  | source of information is another fif file. |
    +----------------------------------+--------------------------------------------+
    | `mne_add_triggers`_              | Modify the trigger channel STI 014 in a raw|
    |                                  | data file. The same effect can be reached  |
    |                                  | by using an event file for averaging in    |
    |                                  | :ref:`mne_process_raw` and                 |
    |                                  | :ref:`mne_browse_raw`.                     |
    +----------------------------------+--------------------------------------------+
    | `mne_annot2labels`_              | Convert parcellation data into label files.|
    +----------------------------------+--------------------------------------------+
    | `mne_anonymize`_                 | Remove subject-specific information from a |
    |                                  | fif data file.                             |
    +----------------------------------+--------------------------------------------+
    | `mne_average_forward_solutions`_ | Calculate an average of forward solutions, |
    |                                  | see :ref:`CHDBBFCA`.                       |
    +----------------------------------+--------------------------------------------+
    | `mne_brain_vision2fiff`_         | Convert EEG data from BrainVision format   |
    |                                  | to fif format.                             |
    +----------------------------------+--------------------------------------------+
    | `mne_change_baselines`_          | Change the dc offsets according to         |
    |                                  | specifications given in a text file.       |
    +----------------------------------+--------------------------------------------+
    | `mne_change_nave`_               | Change the number of averages in an        |
    |                                  | evoked-response data file. This is often   |
    |                                  | necessary if the file was derived from     |
    |                                  | several files.                             |
    +----------------------------------+--------------------------------------------+
    | `mne_check_eeg_locations`_       | Checks that the EEG electrode locations    |
    |                                  | have been correctly transferred from the   |
    |                                  | Polhemus data block to the channel         |
    |                                  | information tags                           |
    +----------------------------------+--------------------------------------------+
    | `mne_check_surface`_             | Check the validity of a FreeSurfer surface |
    |                                  | file or one of the surfaces within a BEM   |
    |                                  | file. This program simply checks for       |
    |                                  | topological errors in surface files.       |
    +----------------------------------+--------------------------------------------+
    | `mne_collect_transforms`_        | Collect coordinate transformations from    |
    |                                  | several sources into a single fif file.    |
    +----------------------------------+--------------------------------------------+
    | `mne_compensate_data`_           | Change the applied software gradient       |
    |                                  | compensation in an evoked-response data    |
    |                                  | file, see :ref:`BEHDDFBI`.                 |
    +----------------------------------+--------------------------------------------+
    | `mne_copy_processing_history`_   | Copy the processing history between files. |
    +----------------------------------+--------------------------------------------+
    | `mne_convert_dig_data`_          | Convert digitization data between          |
    |                                  | different formats.                         |
    +----------------------------------+--------------------------------------------+
    | `mne_convert_lspcov`_            | Convert the LISP format noise covariance   |
    |                                  | matrix output by graph into fif.           |
    +----------------------------------+--------------------------------------------+
    | `mne_convert_ncov`_              | Convert the ncov format noise covariance   |
    |                                  | file to fif.                               |
    +----------------------------------+--------------------------------------------+
    | `mne_convert_surface`_           | Convert FreeSurfer and text format surface |
    |                                  | files into Matlab mat files.               |
    +----------------------------------+--------------------------------------------+
    | `mne_cov2proj`_                  | Pick eigenvectors from a covariance matrix |
    |                                  | and create a signal-space projection (SSP) |
    |                                  | file out of them.                          |
    +----------------------------------+--------------------------------------------+
    | `mne_create_comp_data`_          | Create a fif file containing software      |
    |                                  | gradient compensation information from a   |
    |                                  | text file.                                 |
    +----------------------------------+--------------------------------------------+
    | `mne_ctf2fiff`_                  | Convert a CTF ds folder into a fif file.   |
    +----------------------------------+--------------------------------------------+
    | `mne_ctf_dig2fiff`_              | Convert text format digitization data to   |
    |                                  | fif format.                                |
    +----------------------------------+--------------------------------------------+
    | `mne_dicom_essentials`_          | List essential information from a          |
    |                                  | DICOM file.                                |
    |                                  | This utility is used by the script         |
    |                                  | mne_organize_dicom, see :ref:`BABEBJHI`.   |
    +----------------------------------+--------------------------------------------+
    | `mne_edf2fiff`_                  | Convert EEG data from the EDF/EDF+/BDF     |
    |                                  | formats to the fif format.                 |
    +----------------------------------+--------------------------------------------+
    | `mne_epochs2mat`_                | Apply bandpass filter to raw data and      |
    |                                  | extract epochs for subsequent processing   |
    |                                  | in Matlab.                                 |
    +----------------------------------+--------------------------------------------+
    | `mne_evoked_data_summary`_       | List summary of averaged data from a fif   |
    |                                  | file to the standard output.               |
    +----------------------------------+--------------------------------------------+
    | `mne_eximia2fiff`_               | Convert EEG data from the Nexstim eXimia   |
    |                                  | system to fif format.                      |
    +----------------------------------+--------------------------------------------+
    | `mne_fit_sphere_to_surf`_        | Fit a sphere to a surface given in fif     |
    |                                  | or FreeSurfer format.                      |
    +----------------------------------+--------------------------------------------+
    | `mne_fix_mag_coil_types`_        | Update the coil types for magnetometers    |
    |                                  | in a fif file.                             |
    +----------------------------------+--------------------------------------------+
    | `mne_fix_stim14`_                | Fix coding errors of trigger channel       |
    |                                  | STI 014, see :ref:`BABCDBDI`.              |
    +----------------------------------+--------------------------------------------+
    | `mne_flash_bem`_                 | Create BEM tessellation using multi-echo   |
    |                                  | FLASH MRI data, see :ref:`BABFCDJH`.       |
    +----------------------------------+--------------------------------------------+
    | `mne_insert_4D_comp`_            | Read Magnes compensation channel data from |
    |                                  | a text file and merge it with raw data     |
    |                                  | from other channels in a fif file.         |
    +----------------------------------+--------------------------------------------+
    | `mne_kit2fiff`_                  | Convert KIT data to FIF.                   |
    +----------------------------------+--------------------------------------------+
    | `mne_list_bem`_                  | List BEM information in text format.       |
    +----------------------------------+--------------------------------------------+
    | `mne_list_coil_def`_             | Create the coil description file. This     |
    |                                  | is run automatically at when the software  |
    |                                  | is set up, see :ref:`BJEHHJIJ`.            |
    +----------------------------------+--------------------------------------------+
    | `mne_list_proj`_                 | List signal-space projection data from a   |
    |                                  | fif file.                                  |
    +----------------------------------+--------------------------------------------+
    | `mne_list_source_space`_         | List source space information in text      |
    |                                  | format suitable for importing into         |
    |                                  | Neuromag MRIlab.                           |
    +----------------------------------+--------------------------------------------+
    | `mne_list_versions`_             | List versions and compilation dates of MNE |
    |                                  | software modules.                          |
    +----------------------------------+--------------------------------------------+
    | `mne_make_cor_set`_              | Used by mne_setup_mri to create fif format |
    |                                  | MRI description files from COR or mgh/mgz  |
    |                                  | format MRI data, see :ref:`BABCCEHF`.      |
    +----------------------------------+--------------------------------------------+
    | `mne_make_derivations`_          | Create a channel derivation data file.     |
    +----------------------------------+--------------------------------------------+
    | `mne_make_eeg_layout`_           | Make a topographical trace layout file     |
    |                                  | using the EEG electrode locations from     |
    |                                  | an actual measurement.                     |
    +----------------------------------+--------------------------------------------+
    | `mne_make_morph_maps`_           | Precompute the mapping data needed for     |
    |                                  | morphing between subjects, see             |
    |                                  | :ref:`CHDBBHDH`.                           |
    +----------------------------------+--------------------------------------------+
    | `mne_make_uniform_stc`_          | Create a spatially uniform stc file for    |
    |                                  | testing purposes.                          |
    +----------------------------------+--------------------------------------------+
    | `mne_mark_bad_channels`_         | Update the list of unusable channels in    |
    |                                  | a data file                                |
    +----------------------------------+--------------------------------------------+
    | `mne_morph_labels`_              | Morph label file definitions between       |
    |                                  | subjects.                                  |
    +----------------------------------+--------------------------------------------+
    | `mne_organize_dicom`_            | Organized DICOM MRI image files into       |
    |                                  | directories, see :ref:`BABEBJHI`.          |
    +----------------------------------+--------------------------------------------+
    | `mne_prepare_bem_model`_         | Perform the geometry calculations for      |
    |                                  | BEM forward solutions, see :ref:`CHDJFHEB`.|
    +----------------------------------+--------------------------------------------+
    | `mne_process_stc`_               | Manipulate stc files.                      |
    +----------------------------------+--------------------------------------------+
    | `mne_raw2mat`_                   | Convert raw data into a Matlab file.       |
    +----------------------------------+--------------------------------------------+
    | `mne_rename_channels`_           | Change the names and types of channels     |
    |                                  | in a fif file.                             |
    +----------------------------------+--------------------------------------------+
    | `mne_sensitivity_map`_           | Compute a sensitivity map and output       |
    |                                  | the result in a w-file.                    |
    +----------------------------------+--------------------------------------------+
    | `mne_sensor_locations`_          | Create a file containing the sensor        |
    |                                  | locations in text format.                  |
    +----------------------------------+--------------------------------------------+
    | `mne_show_fiff`_                 | List contents of a fif file.               |
    +----------------------------------+--------------------------------------------+
    | `mne_simu`_                      | Simulate MEG and EEG data.                 |
    +----------------------------------+--------------------------------------------+
    | `mne_smooth`_                    | Smooth a w or stc file.                    |
    +----------------------------------+--------------------------------------------+
    | `mne_surf2bem`_                  | Create a *fif* file describing the         |
    |                                  | triangulated compartment boundaries for    |
    |                                  | the boundary-element model (BEM),          |
    |                                  | see :ref:`BEHCACCJ`.                       |
    +----------------------------------+--------------------------------------------+
    | `mne_toggle_skips`_              | Change data skip tags in a raw file into   |
    |                                  | ignored skips or vice versa.               |
    +----------------------------------+--------------------------------------------+
    | `mne_transform_points`_          | Transform between MRI and MEG head         |
    |                                  | coordinate frames.                         |
    +----------------------------------+--------------------------------------------+
    | `mne_tufts2fiff`_                | Convert EEG data from the Tufts            |
    |                                  | University format to fif format.           |
    +----------------------------------+--------------------------------------------+
    | `mne_view_manual`_               | Starts a PDF reader to show this manual    |
    |                                  | from its standard location.                |
    +----------------------------------+--------------------------------------------+
    | `mne_volume_data2mri`_           | Convert volumetric data defined in a       |
    |                                  | source space created with                  |
    |                                  | mne_volume_source_space into an MRI        |
    |                                  | overlay.                                   |
    +----------------------------------+--------------------------------------------+
    | `mne_volume_source_space`_       | Make a volumetric source space,            |
    |                                  | see :ref:`BJEFEHJI`.                       |
    +----------------------------------+--------------------------------------------+
    | `mne_watershed_bem`_             | Do the segmentation for BEM using the      |
    |                                  | watershed algorithm, see :ref:`BABBDHAG`.  |
    +----------------------------------+--------------------------------------------+


Software component command-line arguments
#########################################

.. _mne_analyze:

mne_analyze
===========

Since mne_analyze is primarily an interactive analysis tool, there are only a
few command-line options:

``\---cd <*dir*>``

    Change to this directory before starting.

``\---subject <*name*>``

    Specify the default subject name for surface loading.

``\---digtrig <*name*>``

    Name of the digital trigger channel. The default value is 'STI
    014'. Underscores in the channel name will be replaced
    by spaces.

``\---digtrigmask <*number*>``

    Mask to be applied to the raw data trigger channel values before considering
    them. This option is useful if one wants to set some bits in a don't
    care state. For example, some finger response pads keep the trigger
    lines high if not in use, *i.e.*, a finger is
    not in place. Yet, it is convenient to keep these devices permanently
    connected to the acquisition system. The number can be given in
    decimal or hexadecimal format (beginning with 0x or 0X). For example,
    the value 255 (0xFF) means that only the lowest order byte (usually
    trigger lines 1 - 8 or bits 0 - 7) will be considered.

``\---visualizehpi``

    Start mne_analyze in the restricted *head
    position visualization* mode. For details, see :ref:`CHDEDFAE`.

``\---dig <*filename*>``

    Specify a file containing the head shape digitization data. This option
    is only usable if the *head position visualization* position
    visualization mode has been first invoked with the --visualizehpi
    option.

``\---hpi <*filename*>``

    Specify a file containing the transformation between the MEG device
    and head coordinate frames. This option is only usable if the *head
    position visualization* position visualization mode has
    been first invoked with the ``--visualizehpi`` option.

``\---scalehead``

    In *head position visualization* mode, scale
    the average scalp surface according to the head surface digitization
    data before aligning  them to the scalp surface. This option is
    recommended.

``\---rthelmet``

    Use the room-temperature helmet surface instead of the MEG sensor
    surface when showing the relative position of the MEG sensors and
    the head in the *head position visualization* mode.

.. note:: Before starting mne_analyze the ``SUBJECTS_DIR`` environment variable has to be set.

.. note:: Strictly speaking, trigger mask value zero would mean that all trigger inputs are ignored. However, for convenience,    setting the mask to zero or not setting it at all has the same effect    as 0xFFFFFFFF, *i.e.*, all bits set.

.. note:: The digital trigger channel can also be set with the MNE_TRIGGER_CH_NAME environment variable. Underscores in the variable    value will *not* be replaced with spaces by mne_analyze .    Using the ``--digtrig`` option supersedes the MNE_TRIGGER_CH_NAME    environment variable.

.. note:: The digital trigger channel mask can also be set with the MNE_TRIGGER_CH_MASK environment variable. Using the ``--digtrigmask`` option    supersedes the MNE_TRIGGER_CH_MASK environment variable.


.. _mne_average_estimates:

mne_average_estimates
=====================
This is a utility for averaging data in stc files. It requires that
all stc files represent data on one individual's cortical
surface and contain identical sets of vertices. mne_average_estimates uses
linear interpolation to resample data in time as necessary. The
command line arguments are:

``---desc <filenname>``

    Specifies the description file for averaging. The format of this
    file is described below.

The description file
--------------------

The description file for mne_average_estimates consists
of a sequence of tokens, separated by whitespace (space, tab, or
newline). If a token consists of several words it has to be enclosed
in quotes. One or more tokens constitute an phrase, which has a
meaning for the averaging definition. Any line starting with the
pound sign (#) is a considered to be a comment line. There are two
kinds of phrases in the description file: global and contextual.
The global phrases have the same meaning independent on their location
in the file while the contextual phrases have different effects depending
on their location in the file.

There are three types of contexts in the description file:
the global context, an input context,
and the output context. In the
beginning of the file the context is global for
defining global parameters. The input context
defines one of the input files (subjects) while the output context
specifies the destination for the average.

The global phrases are:

``tmin <*value/ms*>``

    The minimum time to be considered. The output stc file starts at
    this time point if the time ranges of the stc files include this
    time. Otherwise the output starts from the next later available
    time point.

``tstep <*step/ms*>``

    Time step between consecutive movie frames, specified in milliseconds.

``tmax <*value/ms*>``

    The maximum time point to be considered. A multiple of tstep will be
    added to the first time point selected until this value or the last time
    point in one of the input stc files is reached.

``integ  <:math:`\Delta t` /*ms*>``

    Integration time for each frame. Defaults to zero. The integration will
    be performed on sensor data. If the time specified for a frame is :math:`t_0`,
    the integration range will be :math:`t_0 - ^{\Delta t}/_2 \leq t \leq t_0 + ^{\Delta t}/_2`.

``stc <*filename*>``

    Specifies an input stc file. The filename can be specified with
    one of the ``-lh.stc`` and ``-rh.stc`` endings
    or without them. This phrase ends the present context and starts
    an input context.

``deststc <*filename*>``

    Specifies the output stc file. The filename can be specified with
    one of the ``-lh.stc`` and ``-rh.stc`` endings
    or without them. This phrase ends the present context and starts
    the output context.

``lh``

    Process the left hemisphere. By default, both hemispheres are processed.

``rh``

    Process the left hemisphere. By default, both hemispheres are processed.

The contextual phrases are:

``weight <*value*>``

    Specifies the weight of the current data set. This phrase is valid
    in the input and output contexts.

``abs``

    Specifies that the absolute value of the data should be taken. Valid
    in all contexts. If specified in the global context, applies to
    all subsequent input and output contexts. If specified in the input
    or output contexts, applies only to the data associated with that
    context.

``pow <*value*>``

    Specifies that the data should raised to the specified power. For
    negative values, the absolute value of the data will be taken and
    the negative sign will be transferred to the result, unless abs is
    specified. Valid in all contexts. Rules of application are identical
    to abs .

``sqrt``

    Means pow 0.5

The effects of the options can be summarized as follows.
Suppose that the description file includes :math:`P` contexts
and the temporally resampled data are organized in matrices :math:`S^{(p)}`,
where :math:`p = 1 \dotso P` is the subject index, and
the rows are the signals at different vertices of the cortical surface.
The average computed by mne_average_estimates is
then:

.. math::    A_{jk} = |w[\newcommand\sgn{\mathop{\mathrm{sgn}}\nolimits}\sgn(B_{jk})]^{\alpha}|B_{jk}|^{\beta}

with

.. math::    B_{jk} = \sum_{p = 1}^p {\bar{w_p}[\newcommand\sgn{\mathop{\mathrm{sgn}}\nolimits}\sgn(S_{jk}^{(p)})^{\alpha_p}|S_{jk}^{(p)}|^{\beta_p}}

and

.. math::    \bar{w_p} = w_p / \sum_{p = 1}^p {|w_p|}\ .

In the above, :math:`\beta_p` and :math:`w_p` are
the powers and weights assigned to each of the subjects whereas :math:`\beta` and :math:`w` are
the output weight and power value, respectively. The sign is either
included (:math:`\alpha_p = 1`, :math:`\alpha = 1`)
or omitted (:math:`\alpha_p = 2`, :math:`\alpha = 2`)
depending on the presence of abs phrases in the description file.

.. note:: mne_average_estimates requires    that the number of vertices in the stc files are the same and that    the vertex numbers are identical. This will be the case if the files    have been produced in mne_make_movie using    the ``--morph`` option.

.. note:: It is straightforward to read and write stc    files using the MNE Matlab toolbox described in :ref:`ch_matlab` and    thus write custom Matlab functions to realize more complicated custom    group analysis tools.


.. _mne_browse_raw:

mne_browse_raw
==============

``--cd <*dir*>``

    Change to this directory before starting.

``--raw <*name*>``

    Specifies the raw data file to be opened. If a raw data file is not
    specified, an empty interactive browser will open.

``--grad <*number*>``

    Apply software gradient compensation of the given order to the data loaded
    with the ``--raw`` option. This option is effective only
    for data acquired with the CTF and 4D Magnes MEG systems. If orders
    different from zero are requested for Neuromag data, an error message appears
    and data are not loaded. Any compensation already existing in the
    file can be undone or changed to another order by using an appropriate ``--grad`` options.
    Possible orders are 0 (No compensation), 1 - 3 (CTF data), and 101
    (Magnes data). This applies only to the data file loaded by specifying the ``--raw`` option.
    For interactive data loading, the software gradient compensation
    is specified in the corresponding file selection dialog, see :ref:`CACDCHAJ`.

``--filtersize <*size*>``

    Adjust the length of the FFT to be applied in filtering. The number will
    be rounded up to the next power of two. If the size is :math:`N`,
    the corresponding length of time is :math:`N/f_s`,
    where :math:`f_s` is the sampling frequency
    of your data. The filtering procedure includes overlapping tapers
    of length :math:`N/2` so that the total FFT
    length will actually be :math:`2N`. This
    value cannot be changed after the program has been started.

``--highpass <*value/Hz*>``

    Highpass filter frequency limit. If this is too low with respect
    to the selected FFT length and, the data will not be highpass filtered. It
    is best to experiment with the interactive version to find the lowest applicable
    filter for your data. This value can be adjusted in the interactive
    version of the program. The default is 0, *i.e.*,
    no highpass filter apart from that used during the acquisition will
    be in effect.

``--highpassw <*value/Hz*>``

    The width of the transition band of the highpass filter. The default
    is 6 frequency bins, where one bin is :math:`f_s / (2N)`. This
    value cannot be adjusted in the interactive version of the program.

``--lowpass <*value/Hz*>``

    Lowpass filter frequency limit. This value can be adjusted in the interactive
    version of the program. The default is 40 Hz.

``--lowpassw <*value/Hz*>``

    The width of the transition band of the lowpass filter. This value
    can be adjusted in the interactive version of the program. The default
    is 5 Hz.

``--eoghighpass <*value/Hz*>``

    Highpass filter frequency limit for EOG. If this is too low with respect
    to the selected FFT length and, the data will not be highpass filtered.
    It is best to experiment with the interactive version to find the
    lowest applicable filter for your data. This value can be adjusted in
    the interactive version of the program. The default is 0, *i.e.*,
    no highpass filter apart from that used during the acquisition will
    be in effect.

``--eoghighpassw <*value/Hz*>``

    The width of the transition band of the EOG highpass filter. The default
    is 6 frequency bins, where one bin is :math:`f_s / (2N)`.
    This value cannot be adjusted in the interactive version of the
    program.

``--eoglowpass <*value/Hz*>``

    Lowpass filter frequency limit for EOG. This value can be adjusted in
    the interactive version of the program. The default is 40 Hz.

``--eoglowpassw <*value/Hz*>``

    The width of the transition band of the EOG lowpass filter. This value
    can be adjusted in the interactive version of the program. The default
    is 5 Hz.

``--filteroff``

    Do not filter the data. This initial value can be changed in the
    interactive version of the program.

``--digtrig <*name*>``

    Name of the composite digital trigger channel. The default value
    is 'STI 014'. Underscores in the channel name
    will be replaced by spaces.

``--digtrigmask <*number*>``

    Mask to be applied to the trigger channel values before considering them.
    This option is useful if one wants to set some bits in a don't care
    state. For example, some finger response pads keep the trigger lines
    high if not in use, *i.e.*, a finger is not in
    place. Yet, it is convenient to keep these devices permanently connected
    to the acquisition system. The number can be given in decimal or
    hexadecimal format (beginning with 0x or 0X). For example, the value
    255 (0xFF) means that only the lowest order byte (usually trigger
    lines 1 - 8 or bits 0 - 7) will be considered.

``--allowmaxshield``

    Allow loading of unprocessed Elekta-Neuromag data with MaxShield
    on. These kind of data should never be used for source localization
    without further processing with Elekta-Neuromag software.

``--deriv <*name*>``

    Specifies the name of a derivation file. This overrides the use
    of a standard derivation file, see :ref:`CACFHAFH`.

``--sel <*name*>``

    Specifies the channel selection file to be used. This overrides
    the use of the standard channel selection files, see :ref:`CACCJEJD`.

.. note:: Strictly speaking, trigger mask value zero would mean that all trigger inputs are ignored. However, for convenience,    setting the mask to zero or not setting it at all has the same effect    as 0xFFFFFFFF, *i.e.*, all bits set.

.. note:: The digital trigger channel can also be set with the MNE_TRIGGER_CH_NAME environment variable. Underscores in the variable value will *not* be replaced with spaces. Using the ``--digtrig`` option supersedes the MNE_TRIGGER_CH_NAME    environment variable.

.. note:: The digital trigger channel mask can also be set with the MNE_TRIGGER_CH_MASK environment variable. Using the ``--digtrigmask`` option    supersedes the MNE_TRIGGER_CH_MASK environment variable.



.. _mne_compute_mne:

mne_compute_mne
===============

This program is gradually becoming obsolete. All of its functions will
be eventually included to :ref:`mne_make_movie`,
see :ref:`movies_and_snapshots`. At this time, :ref:`mne_compute_mne` is
still needed to produce time-collapsed w files unless you are willing
to write a Matlab script of your own for this purpose.


``--inv <*name*>``

    Load the inverse operator decomposition from here.

``--meas <*name*>``

    Load the MEG or EEG data from this file.

``--set <*number*>``

    The data set (condition) number to load. The list of data sets can
    be seen, *e.g.*, in mne_analyze , mne_browse_raw ,
    and xplotter .

``--bmin <*time/ms*>``

    Specifies the starting time of the baseline. In order to activate
    baseline correction, both ``--bmin`` and ``--bmax`` options
    must be present.

``--bmax <*time/ms*>``

    Specifies the finishing time of the baseline.

``--nave <*value*>``

    Specifies the number of averaged epochs in the input data. If the input
    data file is one produced by mne_process_raw or mne_browse_raw ,
    the number of averages is correct in the file. However, if subtractions
    or some more complicated combinations of simple averages are produced, *e.g.*,
    by using the xplotter software, the
    number of averages should be manually adjusted. This is accomplished
    either by employing this flag or by adjusting the number of averages
    in the data file with help of mne_change_nave .

``--snr <*value*>``

    An estimate for the amplitude SNR. The regularization parameter will
    be set as :math:`\lambda = ^1/_{\text{SNR}}`. If the SNR option is
    absent, the regularization parameter will be estimated from the
    data. The regularization parameter will be then time dependent.

``--snronly``

    Only estimate SNR and output the result into a file called SNR. Each
    line of the file contains three values: the time point in ms, the estimated
    SNR + 1, and the regularization parameter estimated from the data
    at this time point.

``--abs``

    Calculate the absolute value of the current and the dSPM for fixed-orientation
    data.

``--spm``

    Calculate the dSPM instead of the expected current value.

``--chi2``

    Calculate an approximate :math:`\chi_2^3` statistic
    instead of the *F* statistic. This is simply
    accomplished by multiplying the *F* statistic
    by three.

``--sqrtF``

    Take the square root of the :math:`\chi_2^3` or *F* statistic
    before outputting the stc file.

``--collapse``

    Make all frames in the stc file (or the wfile) identical. The value
    at each source location is the maximum value of the output quantity
    at this location over the analysis period. This option is convenient
    for determining the correct thresholds for the rendering of the
    final brain-activity movies.

``--collapse1``

    Make all frames in the stc file (or the wfile) identical. The value
    at each source location is the :math:`L_1` norm
    of the output quantity at this location over the analysis period.

``--collapse2``

    Make all frames in the stc file (or the wfile) identical. The value
    at each source location is the :math:`L_2` norm
    of the output quantity at this location over the analysis period.

``--SIcurrents``

    Output true current values in SI units (Am). By default, the currents are
    scaled so that the maximum current value is set to 50 (Am).

``--out <*name*>``

    Specifies the output file name. This is the 'stem' of
    the output file name. The actual name is derived by removing anything up
    to and including the last period from the end of <*name*> .
    According to the hemisphere, ``-lh`` or ``-rh`` is
    then appended. Finally, ``.stc`` or ``.w`` is added,
    depending on the output file type.

``--wfiles``

    Use binary w-files in the output whenever possible. The noise-normalization
    factors can be always output in this format.  The current estimates
    and dSPMs can be output as wfiles if one of the collapse options
    is selected.

``--pred <*name*>``

    Save the predicted data into this file. This is a fif file containing
    the predicted data waveforms, see :ref:`CHDCACDC`.

``--outputnorm <*name*>``

    Output noise-normalization factors to this file.

``--invnorm``

    Output inverse noise-normalization factors to the file defined by
    the ``--outputnorm`` option.

``--dip <*name*>``

    Specifies a dipole distribution snapshot file. This is a file containing the
    current distribution at a time specified with the ``--diptime`` option.
    The file format is the ASCII dip file format produced by the Neuromag
    source modelling software (xfit). Therefore, the file can be loaded
    to the Neuromag MRIlab MRI viewer to display the actual current
    distribution. This option is only effective if the ``--spm`` option
    is absent.

``--diptime <*time/ms*>``

    Time for the dipole snapshot, see ``--dip`` option above.

``--label <*name*>``

    Label to process. The label files are produced by tksurfer and specify
    regions of interests (ROIs). A label file name should end with ``-lh.label`` for
    left-hemisphere ROIs and with ``-rh.label`` for right-hemisphere
    ones. The corresponding output files are tagged with ``-lh-`` <*data type* ``.amp`` and ``-rh-`` <*data type* ``.amp`` , respectively. <*data type*> equals ``MNE`` for expected current
    data and ``spm`` for dSPM data. Each line of the output
    file contains the waveform of the output quantity at one of the
    source locations falling inside the ROI.

.. note:: The ``--tmin`` and ``--tmax`` options    which existed in previous versions of mne_compute_mne have    been removed. mne_compute_mne can now    process only the entire averaged epoch.


.. _mne_compute_raw_inverse:

mne_compute_raw_inverse
=======================

``--in <*filename*>``

    Specifies the input data file. This can be either an evoked data
    file or a raw data file.

``--bmin <*time/ms*>``

    Specifies the starting time of the baseline. In order to activate
    baseline correction, both ``--bmin`` and ``--bmax`` options
    must be present. This option applies to evoked data only.

``--bmax <*time/ms*>``

    Specifies the finishing time of the baseline. This option applies
    to evoked data only.

``--set <*number*>``

    The data set (condition) number to load. This is the sequential
    number of the condition. You can easily see the association by looking
    at the condition list in mne_analyze when
    you load the file.

``--inv <*name*>``

    Load the inverse operator decomposition from here.

``--nave <*value*>``

    Specifies the effective number of averaged epochs in the input data, :math:`L_{eff}`,
    as discussed in :ref:`CBBDGIAE`. If the input data file is
    one produced by mne_browse_raw or mne_process_raw ,
    the number of averages is correct in the file. However, if subtractions
    or some more complicated combinations of simple averages are produced,
    e.g., by  using the xplotter software,
    the number of averages should be manually adjusted along the guidelines
    given in :ref:`CBBDGIAE`. This is accomplished either by
    employing this flag or by adjusting the number of averages in the
    data file with help of the utility mne_change_nave .

``--snr <*value*>``

    An estimate for the amplitude SNR. The regularization parameter will
    be set as :math:`\lambda^2 = 1/SNR^2`. The default value is
    SNR = 1. Automatic selection of the regularization parameter is
    currently not supported.

``--spm``

    Calculate the dSPM instead of the expected current value.

``--picknormalcomp``

    The components of the estimates corresponding to directions tangential
    with the cortical mantle are zeroed out.

``--mricoord``

    Provide source locations and orientations in the MRI coordinate frame
    instead of the default head coordinate frame.

``--label <*name*>``

    Specifies a label file to process. For each label file, the values
    of the computed estimates stored in a fif file. For more details,
    see :ref:`implementation_details`. The label files are produced by tksurfer
    or mne_analyze and specify regions
    of interests (ROIs). A label file name should end with ``-lh.label`` for
    left-hemisphere ROIs and with ``-rh.label`` for right-hemisphere
    ones. The corresponding output files are tagged with ``-lh-`` <*data type*> ``.fif`` and ``-rh-`` <*data type*> ``.fif`` , respectively. <*data type*> equals ``'mne`` ' for expected
    current data and ``'spm`` ' for dSPM data.
    For raw data, ``_raw.fif`` is employed instead of ``.fif`` .
    The output files are stored in the same directory as the label files.

``--labelselout``

    Produces additional label files for each label processed, containing only
    those vertices within the input label which correspond to available
    source space vertices in the inverse operator. These files have the
    same name as the original label except that ``-lh`` and ``-rh`` are replaced
    by ``-sel-lh`` and ``-sel-rh`` , respectively.

``--align_z``

    Instructs the program to try to align the waveform signs within
    the label. For more information, see :ref:`implementation_details`. This
    flag will not have any effect if the inverse operator has been computed
    with the strict orientation constraint active.

``--labeldir <*directory*>``

    All previous ``--label`` options will be ignored when this
    option is encountered. For each label in the directory, the output
    file defined with the ``--out`` option will contain a summarizing
    waveform which is the average of the waveforms in the vertices of
    the label. The ``--labeldir`` option implies ``--align_z`` and ``--picknormalcomp`` options.

``--orignames``

    This option is used with the ``--labeldir`` option, above.
    With this option, the output file channel names will be the names
    of the label files, truncated to 15 characters, instead of names
    containing the vertex numbers.

``--out <*name*>``

    Required with ``--labeldir`` . This is the output file for
    the data.

``--extra <*name*>``

    By default, the output includes the current estimate signals and
    the digital trigger channel, see ``--digtrig`` option,
    below. With the ``--extra`` option, a custom set of additional
    channels can be included. The extra channel text file should contain
    the names of these channels, one channel name on each line. With
    this option present, the digital trigger channel is not included
    unless specified in the extra channel file.

``--noextra``

    No additional channels will be included with this option present.

``--digtrig <*name*>``

    Name of the composite digital trigger channel. The default value
    is 'STI 014'. Underscores in the channel name
    will be replaced by spaces.

``--split <*size/MB*>``

    Specifies the maximum size of the raw data files saved. By default, the
    output is split into files which are just below 2 GB so that the
    fif file maximum size is not exceed.

.. note:: The digital trigger channel can also be set with    the MNE_TRIGGER_CH_NAME environment variable. Underscores in the variable    value will *not* be replaced with spaces by mne_compute_raw_inverse .    Using the ``--digtrig`` option supersedes the MNE_TRIGGER_CH_NAME    environment variable.


.. _mne_convert_mne_data:

mne_convert_mne_data
====================

This utility allows the conversion of various fif files related to the MNE
computations to other formats. The two principal purposes of this utility are
to facilitate development of new analysis approaches with Matlab
and conversion of the forward model and noise covariance matrix
data into evoked-response type fif files, which can be accessed
and displayed with the Neuromag source modelling software.

.. note:: Most of the functions of mne_convert_mne_data are    now covered by the MNE Matlab toolbox covered in :ref:`ch_matlab`.    This toolbox is recommended to avoid creating additional files occupying    disk space.

The command-line options recognized by mne_convert_mne_data are:

``--fwd <*name*>``

    Specity the name of the forward solution file to be converted. Channels
    specified with the ``--bad`` option will be excluded from
    the file.

``--fixed``

    Convert the forward solution to the fixed-orientation mode before outputting
    the converted file. With this option only the field patterns corresponding
    to a dipole aligned with the estimated cortex surface normal are
    output.

``--surfsrc``

    When outputting a free-orientation forward model (three orthogonal dipole
    components present) rotate the dipole coordinate system at each
    source node so that the two tangential dipole components are output
    first, followed by the field corresponding to the dipole aligned
    with the estimated cortex surface normal. The orientation of the
    first two dipole components in the tangential plane is arbitrarily selected
    to create an orthogonal coordinate system.

``--noiseonly``

    When creating a 'measurement' fif file, do not
    output a forward model file, just the noise-covariance matrix.

``--senscov <*name*>``

    Specifies the fif file containing a sensor covariance matrix to
    be included with the output. If no other input files are specified
    only the covariance matrix is output

``--srccov <*name*>``

    Specifies the fif file containing the source covariance matrix to
    be included with the output. Only diagonal source covariance files
    can be handled at the moment.

``--bad <*name*>``

    Specifies the name of the file containing the names of the channels to
    be omitted, one channel name per line. This does not affect the output
    of the inverse operator since the channels have been already selected
    when the file was created.

``--fif``

    Output the forward model and the noise-covariance matrix into 'measurement' fif
    files. The forward model files are tagged with <*modalities*> ``-meas-fwd.fif`` and
    the noise-covariance matrix files with <*modalities*> ``-meas-cov.fif`` .
    Here, modalities is ``-meg`` if MEG is included, ``-eeg`` if
    EEG is included, and ``-meg-eeg`` if both types of signals
    are present. The inclusion of modalities is controlled by the ``--meg`` and ``--eeg`` options.

``--mat``

    Output the data into MATLAB mat files. This is the default. The
    forward model files are tagged with <*modalities*> ``-fwd.mat`` forward model
    and noise-covariance matrix output, with ``-inv.mat`` for inverse
    operator output, and with ``-inv-meas.mat`` for combined inverse
    operator and measurement data output, respectively. The meaning
    of <*modalities*> is the same
    as in the fif output, described above.

``--tag <*name*>``

    By default, all variables in the matlab output files start with
    ``mne\_``. This option allows to change this prefix to <*name*> _.

``--meg``

    Include MEG channels from the forward solution and noise-covariance
    matrix.

``--eeg``

    Include EEG channels from the forward solution and noise-covariance
    matrix.

``--inv <*name*>``

    Output the inverse operator data from the specified file into a
    mat file. The source and noise covariance matrices as well as active channels
    have been previously selected when the inverse operator was created
    with mne_inverse_operator . Thus
    the options ``--meg`` , ``--eeg`` , ``--senscov`` , ``--srccov`` , ``--noiseonly`` ,
    and ``--bad`` do not affect the output of the inverse operator.

``--meas <*name*>``

    Specifies the file containing measurement data to be output together with
    the inverse operator. The channels corresponding to the inverse operator
    are automatically selected from the file if ``--inv`` .
    option is present. Otherwise, the channel selection given with ``--sel`` option will
    be taken into account.

``--set <*number*>``

    Select the data set to be output from the measurement file.

``--bmin <*value/ms*>``

    Specifies the baseline minimum value setting for the measurement signal
    output.

``--bmax <*value/ms*>``

    Specifies the baseline maximum value setting for the measurement signal
    output.

.. note:: The ``--tmin`` and ``--tmax`` options    which existed in previous versions of mne_converted_mne_data have    been removed. If output of measurement data is requested, the entire    averaged epoch is now included.

Guide to combining options
--------------------------

The combination of options is quite complicated. The :ref:`BEHDCIII` should be
helpful to determine the combination of options appropriate for your needs.


.. tabularcolumns:: |p{0.38\linewidth}|p{0.1\linewidth}|p{0.2\linewidth}|p{0.3\linewidth}|
.. _BEHDCIII:
.. table:: Guide to combining mne_convert_mne_data options.

    +-------------------------------------+---------+--------------------------+-----------------------+
    | Desired output                      | Format  | Required options         | Optional options      |
    +-------------------------------------+---------+--------------------------+-----------------------+
    | forward model                       | fif     |   \---fwd <*name*>       | \---bad <*name*>      |
    |                                     |         |   \---out <*name*>       | \---surfsrc           |
    |                                     |         |   \---meg and/or \---eeg |                       |
    |                                     |         |   \---fif                |                       |
    +-------------------------------------+---------+--------------------------+-----------------------+
    | forward model                       | mat     |   \---fwd <*name*>       | \---bad <*name*>      |
    |                                     |         |   \---out <*name*>       | \---surfsrc           |
    |                                     |         |   \---meg and/or --eeg   |                       |
    +-------------------------------------+---------+--------------------------+-----------------------+
    | forward model and sensor covariance | mat     |   \---fwd <*name*>       | \---bad <*name*>      |
    |                                     |         |   \---out <*name*>       | \---surfsrc           |
    |                                     |         |   \---senscov <*name*>   |                       |
    |                                     |         |   \---meg and/or --eeg   |                       |
    +-------------------------------------+---------+--------------------------+-----------------------+
    | sensor covariance                   | fif     |   \---fwd <*name*>       | \---bad <*name*>      |
    |                                     |         |   \---out <*name*>       |                       |
    |                                     |         |   \---senscov <*name*>   |                       |
    |                                     |         |   \---noiseonly          |                       |
    |                                     |         |   \---fif                |                       |
    |                                     |         |   \---meg and/or --eeg   |                       |
    +-------------------------------------+---------+--------------------------+-----------------------+
    | sensor covariance                   | mat     |   \---senscov <*name*>   | \---bad <*name*>      |
    |                                     |         |   \---out <*name*>       |                       |
    +-------------------------------------+---------+--------------------------+-----------------------+
    | sensor covariance eigenvalues       | text    |   \---senscov <*name*>   | \---bad <*name*>      |
    |                                     |         |   \---out <*name*>       |                       |
    |                                     |         |   \---eig                |                       |
    +-------------------------------------+---------+--------------------------+-----------------------+
    | evoked MEG/EEG data                 | mat     |   \---meas <*name*>      | \---sel <*name*>      |
    |                                     |         |   \---out <*name*>       | \---set <*number*>    |
    +-------------------------------------+---------+--------------------------+-----------------------+
    | evoked MEG/EEG data forward model   | mat     |   \---meas <*name*>      | \---bad <*name*>      |
    |                                     |         |   \---fwd <*name*>       | \---set <*number*>    |
    |                                     |         |   \---out <*name*>       |                       |
    +-------------------------------------+---------+--------------------------+-----------------------+
    | inverse operator data               | mat     |   \---inv <*name*>       |                       |
    |                                     |         |   \---out <*name*>       |                       |
    +-------------------------------------+---------+--------------------------+-----------------------+
    | inverse operator data evoked        | mat     |   \–--inv <*name*>       |                       |
    | MEG/EEG data                        |         |   \–--meas <*name*>      |                       |
    |                                     |         |   \–--out <*name*>       |                       |
    +-------------------------------------+---------+--------------------------+-----------------------+

Matlab data structures
----------------------

The Matlab output provided by mne_convert_mne_data is
organized in structures, listed in :ref:`BEHCICCA`. The fields
occurring in these structures are listed in :ref:`BABCBIGF`.


The symbols employed in variable size descriptions are:

``nloc``

    Number
    of source locations

``nsource``

    Number
    of sources. For fixed orientation sources nsource = nloc whereas nsource = 3*nloc for
    free orientation sources

``nchan``

    Number
    of measurement channels.

``ntime``

    Number
    of time points in the measurement data.

.. _BEHCICCA:
.. table:: Matlab structures produced by mne_convert_mne_data.

    ===============  =======================================
    Structure        Contents
    ===============  =======================================
    <*tag*> _meas      Measured data
    <*tag*> _inv       The inverse operator decomposition
    <*tag*> _fwd       The forward solution
    <*tag*> _noise     A standalone noise-covariance matrix
    ===============  =======================================

The prefix given with the ``--tag`` option is indicated <*tag*> , see :ref:`mne_convert_mne_data`. Its default value is MNE.


.. tabularcolumns:: |p{0.14\linewidth}|p{0.13\linewidth}|p{0.73\linewidth}|
.. _BABCBIGF:
.. table:: The fields of Matlab structures.


    +-----------------------+-----------------+------------------------------------------------------------+
    | Variable              | Size            | Description                                                |
    +-----------------------+-----------------+------------------------------------------------------------+
    | fwd                   | nsource x nchan | The forward solution, one source on each row. For free     |
    |                       |                 | orientation sources, the fields of the three orthogonal    |
    |                       |                 | dipoles for each location are listed consecutively.        |
    +-----------------------+-----------------+------------------------------------------------------------+
    | names ch_names        | nchan (string)  | String array containing the names of the channels included |
    +-----------------------+-----------------+------------------------------------------------------------+
    | ch_types              | nchan x 2       | The column lists the types of the channels (1 = MEG,       |
    |                       |                 | 2 = EEG). The second column lists the coil types, see      |
    |                       |                 | :ref:`BGBBHGEC` and :ref:`CHDBDFJE`. For EEG electrodes,   |
    |                       |                 | this value equals one.                                     |
    +-----------------------+-----------------+------------------------------------------------------------+
    | ch_pos                | nchan x 3       | The location information for each channel. The first three |
    |                       |                 | values specify the origin of the sensor coordinate system  |
    |                       |                 | or the location of the electrode. For MEG channels, the    |
    |                       |                 | following nine number specify the *x*, *y*, and            |
    |                       |                 | *z*-direction unit vectors of the sensor coordinate system.|
    |                       |                 | For EEG electrodes the first unit vector specifies the     |
    |                       |                 | location of the reference electrode. If the reference is   |
    |                       |                 | not specified this value is all zeroes. The remaining unit |
    |                       |                 | vectors are irrelevant for EEG electrodes.                 |
    +-----------------------+-----------------+------------------------------------------------------------+
    | ch_lognos             | nchan x 1       | Logical channel numbers as listed in the fiff file         |
    +-----------------------+-----------------+------------------------------------------------------------+
    | ch_units              | nchan x 2       | Units and unit multipliers as listed in the fif file. The  |
    |                       |                 | unit of the data is listed in the first column (T = 112,   |
    |                       |                 | T/m = 201, V = 107). At present, the second column will be |
    |                       |                 | always zero, *i.e.*, no unit multiplier.                   |
    +-----------------------+-----------------+------------------------------------------------------------+
    | ch_cals               | nchan x 2       | Even if the data comes from the conversion already         |
    |                       |                 | calibrated, the original calibration factors are included. |
    |                       |                 | The first column is the range member of the fif data       |
    |                       |                 | structures and while the second is the cal member. To get  |
    |                       |                 | calibrated values in the units given in ch_units from the  |
    |                       |                 | raw data, the data must be multiplied with the product of  |
    |                       |                 | range and cal.                                             |
    +-----------------------+-----------------+------------------------------------------------------------+
    | sfreq                 | 1               | The sampling frequency in Hz.                              |
    +-----------------------+-----------------+------------------------------------------------------------+
    | lowpass               | 1               | Lowpass filter frequency (Hz)                              |
    +-----------------------+-----------------+------------------------------------------------------------+
    | highpass              | 1               | Highpass filter frequency (Hz)                             |
    +-----------------------+-----------------+------------------------------------------------------------+
    | source_loc            | nloc x 3        | The source locations given in the coordinate frame         |
    |                       |                 | indicated by the coord_frame member.                       |
    +-----------------------+-----------------+------------------------------------------------------------+
    | source_ori            | nsource x 3     | The source orientations                                    |
    +-----------------------+-----------------+------------------------------------------------------------+
    | source_selection      | nsource x 2     | Indication of the sources selected from the complete source|
    |                       |                 | spaces. Each row contains the number of the source in the  |
    |                       |                 | complete source space (starting with 0) and the source     |
    |                       |                 | space number (1 or 2). These numbers refer to the order the|
    |                       |                 | two hemispheres where listed when mne_make_source_space was|
    |                       |                 | invoked. mne_setup_source_space lists the left hemisphere  |
    |                       |                 | first.                                                     |
    +-----------------------+-----------------+------------------------------------------------------------+
    | coord_frame           | string          | Name of the coordinate frame employed in the forward       |
    |                       |                 | calculations. Possible values are 'head' and 'mri'.        |
    +-----------------------+-----------------+------------------------------------------------------------+
    | mri_head_trans        | 4 x 4           | The coordinate frame transformation from mri the MEG 'head'|
    |                       |                 | coordinates.                                               |
    +-----------------------+-----------------+------------------------------------------------------------+
    | meg_head_trans        | 4 x 4           | The coordinate frame transformation from the MEG device    |
    |                       |                 | coordinates to the MEG head coordinates                    |
    +-----------------------+-----------------+------------------------------------------------------------+
    | noise_cov             | nchan x nchan   | The noise covariance matrix                                |
    +-----------------------+-----------------+------------------------------------------------------------+
    | source_cov            | nsource         | The elements of the diagonal source covariance matrix.     |
    +-----------------------+-----------------+------------------------------------------------------------+
    | sing                  | nchan           | The singular values of                                     |
    |                       |                 | :math:`A = C_0^{-^1/_2} G R^C = U \Lambda V^T`             |
    |                       |                 | with :math:`R` selected so that                            |
    |                       |                 | :math:`\text{trace}(AA^T) / \text{trace}(I) = 1`           |
    |                       |                 | as discussed in :ref:`CHDDHAGE`                            |
    +-----------------------+-----------------+------------------------------------------------------------+
    | eigen_fields          | nchan x nchan   | The rows of this matrix are the left singular vectors of   |
    |                       |                 | :math:`A`, i.e., the columns of :math:`U`, see above.      |
    +-----------------------+-----------------+------------------------------------------------------------+
    | eigen_leads           | nchan x nsource | The rows of this matrix are the right singular vectors of  |
    |                       |                 | :math:`A`, i.e., the columns of :math:`V`, see above.      |
    +-----------------------+-----------------+------------------------------------------------------------+
    | noise_eigenval        | nchan           | In terms of :ref:`CHDDHAGE`, eigenvalues of :math:`C_0`,   |
    |                       |                 | i.e., not scaled with number of averages.                  |
    +-----------------------+-----------------+------------------------------------------------------------+
    | noise_eigenvec        | nchan           | Eigenvectors of the noise covariance matrix. In terms of   |
    |                       |                 | :ref:`CHDDHAGE`, :math:`U_C^T`.                            |
    +-----------------------+-----------------+------------------------------------------------------------+
    | data                  | nchan x ntime   | The measured data. One row contains the data at one time   |
    |                       |                 | point.                                                     |
    +-----------------------+-----------------+------------------------------------------------------------+
    | times                 | ntime           | The time points in the above matrix in seconds             |
    +-----------------------+-----------------+------------------------------------------------------------+
    | nave                  | 1               | Number of averages as listed in the data file.             |
    +-----------------------+-----------------+------------------------------------------------------------+
    | meas_times            | ntime           | The time points in seconds.                                |
    +-----------------------+-----------------+------------------------------------------------------------+

.. note:: The Matlab files can also be read in Python using :py:func:`scipy.io.loadmat`


.. _mne_do_forward_solution:

mne_do_forward_solution
=======================

This utility accepts the following options:

``--subject <*subject*>``

    Defines the name of the subject. This can be also accomplished
    by setting the SUBJECT environment variable.

``--src <*name*>``

    Source space name to use. This option overrides the ``--spacing`` option. The
    source space is searched first from the current working directory
    and then from ``$SUBJECTS_DIR/`` <*subject*> /bem.
    The source space file must be specified exactly, including the ``fif`` extension.

``--spacing <*spacing/mm*>  or ``ico-`` <*number  or ``oct-`` <*number*>``

    This is an alternate way to specify the name of the source space
    file. For example, if ``--spacing 6`` is given on the command
    line, the source space files searched for are./<*subject*> -6-src.fif
    and ``$SUBJECTS_DIR/$SUBJECT/`` bem/<*subject*> -6-src.fif.
    The first file found is used. Spacing defaults to 7 mm.

``--bem <*name*>``

    Specifies the BEM to be used. The name of the file can be any of <*name*> , <*name*> -bem.fif, <*name*> -bem-sol.fif.
    The file is searched for from the current working directory and
    from ``bem`` . If this option is omitted, the most recent
    BEM file in the ``bem`` directory is used.

``--mri <*name*>``

    The name of the MRI description file containing the MEG/MRI coordinate
    transformation. This file was saved as part of the alignment procedure
    outlined in :ref:`CHDBEHDC`. The file is searched for from
    the current working directory and from ``mri/T1-neuromag/sets`` .
    The search order for MEG/MRI coordinate transformations is discussed
    below.

``--trans	 <*name*>``

    The name of a text file containing the 4 x 4 matrix for the coordinate transformation
    from head to mri coordinates, see below. If the option ``--trans`` is
    present, the ``--mri`` option is not required. The search
    order for MEG/MRI coordinate transformations is discussed below.

``--meas <*name*>``

    This file is the measurement fif file or an off-line average file
    produced thereof. It is recommended that the average file is employed for
    evoked-response data and the original raw data file otherwise. This
    file provides the MEG sensor locations and orientations as well as
    EEG electrode locations as well as the coordinate transformation between
    the MEG device coordinates and MEG head-based coordinates.

``--fwd <*name*>``

    This file will contain the forward solution as well as the coordinate transformations,
    sensor and electrode location information, and the source space
    data. A name of the form <*name*> ``-fwd.fif`` is
    recommended. If this option is omitted the forward solution file
    name is automatically created from the measurement file name and
    the source space name.

``--destdir <*directory*>``

    Optionally specifies a directory where the forward solution will
    be stored.

``--mindist <*dist/mm*>``

    Omit source space points closer than this value to the inner skull surface.
    Any source space points outside the inner skull surface are automatically
    omitted. The use of this option ensures that numerical inaccuracies
    for very superficial sources do not cause unexpected effects in
    the final current estimates. Suitable value for this parameter is
    of the order of the size of the triangles on the inner skull surface.
    If you employ the seglab software
    to create the triangulations, this value should be about equal to
    the wish for the side length of the triangles.

``--megonly``

    Omit EEG forward calculations.

``--eegonly``

    Omit MEG forward calculations.

``--all``

    Compute the forward solution for all vertices on the source space.

``--overwrite``

    Overwrite the possibly existing forward model file.

``--help``

    Show usage information for the script.

The MEG/MRI transformation is determined by the following
search sequence:

- If the ``--mri`` option was
  present, the file is looked for literally as specified, in the directory
  of the measurement file specified with the ``--meas`` option,
  and in the directory $SUBJECTS_DIR/$SUBJECT/mri/T1-neuromag/sets.
  If the file is not found, the script exits with an error message.

- If the ``--trans`` option was present, the file is
  looked up literally as specified. If the file is not found, the
  script exists with an error message.

- If neither ``--mri`` nor ``--trans`` option
  was not present, the following default search sequence is engaged:

  - The ``.fif`` ending in the
    measurement file name is replaced by ``-trans.fif`` . If
    this file is present, it will be used.

  - The newest file whose name ends with ``-trans.fif`` in
    the directory of the measurement file is looked up. If such a file
    is present, it will be used.

  - The newest file whose name starts with ``COR-`` in
    directory $SUBJECTS_DIR/$SUBJECT/mri/T1-neuromag/sets is looked
    up. If such a file is present, it will be used.

  - If all the above searches fail, the script exits with an error
    message.

This search sequence is designed to work well with the MEG/MRI
transformation files output by mne_analyze ,
see :ref:`CACEHGCD`. It is recommended that -trans.fif file
saved with the Save default and Save... options in
the mne_analyze alignment dialog
are used because then the $SUBJECTS_DIR/$SUBJECT directory will
be composed of files which are dependent on the subjects's
anatomy only, not on the MEG/EEG data to be analyzed.

.. note:: If the standard MRI description file and BEM    file selections are appropriate and the 7-mm source space grid spacing    is appropriate, only the ``--meas`` option is necessary.    If EEG data is not used ``--megonly`` option should be    included.

.. note:: If it is conceivable that the current-density    transformation will be incorporated into the inverse operator, specify    a source space with patch information for the forward computation.    This is not mandatory but saves a lot of time when the inverse operator    is created, since the patch information does not need to be created    at that stage.

.. note:: The MEG head to MRI transformation matrix specified    with the ``--trans`` option should be a text file containing    a 4-by-4 matrix:

.. math::    T = \begin{bmatrix}
		R_{11} & R_{12} & R_{13} & x_0 \\
		R_{13} & R_{13} & R_{13} & y_0 \\
		R_{13} & R_{13} & R_{13} & z_0 \\
		0 & 0 & 0 & 1
		\end{bmatrix}
	     
defined so that if the augmented location vectors in MRI
head and MRI coordinate systems are denoted by :math:`r_{head}[x_{head}\ y_{head}\ z_{head}\ 1]` and :math:`r_{MRI}[x_{MRI}\ y_{MRI}\ z_{MRI}\ 1]`,
respectively,

.. math::    r_{MRI} = T r_{head}

.. note:: It is not possible to calculate an EEG forward    solution with a single-layer BEM.


.. _mne_do_inverse_operator:

mne_do_inverse_operator
=======================

``--fwd <*name of the forward solution file*>``

    This is the forward solution file produced in the computations step described
    in :ref:`BABCHEJD`.

``--meg``

    Employ MEG data in the inverse calculation. If neither ``--meg`` nor ``--eeg`` is
    set only MEG channels are included.

``--eeg``

    Employ EEG data in the inverse calculation. If neither ``--meg`` nor ``--eeg`` is
    set only MEG channels are included.

``--fixed``

    Use fixed source orientations normal to the cortical mantle. By default,
    the source orientations are not constrained. If ``--fixed`` is specified,
    the ``--loose`` flag is ignored.

``--loose <*amount*>``

    Use a 'loose' orientation constraint. This means
    that the source covariance matrix entries corresponding to the current
    component normal to the cortex are set equal to one and the transverse
    components are set to <*amount*> .
    Recommended value of amount is 0.1...0.6.

``--depth``

    Employ depth weighting with the standard settings. For details,
    see :ref:`depth_weighting` and :ref:`inverse_operator`.

``--bad <*name*>``

    Specifies a text file to designate bad channels, listed one channel name
    (like MEG 1933) on each line of the file. Be sure to include both
    noisy and flat (non-functioning) channels in the list. If bad channels
    were designated using mne_mark_bad_channels in
    the measurement file which was specified with the ``--meas`` option when
    the forward solution was computed, the bad channel information will
    be automatically included. Also, any bad channel information in
    the noise-covariance matrix file will be included.

``--noisecov <*name*>``

    Name of the noise-covariance matrix file computed with one of the methods
    described in :ref:`BABDEEEB`. By default, the script looks
    for a file whose name is derived from the forward solution file
    by replacing its ending ``-`` <*anything*> ``-fwd.fif`` by ``-cov.fif`` .
    If this file contains a projection operator, which will automatically
    attached to the noise-covariance matrix by mne_browse_raw and mne_process_raw ,
    no ``--proj`` option is necessary because mne_inverse_operator will
    automatically include the projectors from the noise-covariance matrix
    file. For backward compatibility, --senscov can be used as a synonym
    for --noisecov.

``--noiserank <*value*>``

    Specifies the rank of the noise covariance matrix explicitly rather than
    trying to reduce it automatically. This option is sheldom needed,

``--megreg <*value*>``

    Regularize the MEG part of the noise-covariance matrix by this amount.
    Suitable values are in the range 0.05...0.2. For details, see :ref:`cov_regularization`.

``--eegreg <*value*>``

    Like ``--megreg`` but applies to the EEG channels.

``--diagnoise``

    Omit the off-diagonal terms of the noise covariance matrix. This option
    is irrelevant to most users.

``--fmri <*name*>``

    With help of this w file, an *a priori* weighting
    can be applied to the source covariance matrix. The source of the weighting
    is usually fMRI but may be also some other data, provided that the weighting can
    be expressed as a scalar value on the cortical surface, stored in
    a w file. It is recommended that this w file is appropriately smoothed (see :ref:`CHDEBAHH`)
    in mne_analyze , tksurfer or
    with mne_smooth_w to contain
    nonzero values at all vertices of the triangular tessellation of
    the cortical surface. The name of the file given is used as a stem of
    the w files. The actual files should be called <*name*> ``-lh.pri`` and <*name*> ``-rh.pri`` for
    the left and right hemisphere weight files, respectively. The application
    of the weighting is discussed in :ref:`mne_fmri_estimates`.

``--fmrithresh <*value*>``

    This option is mandatory and has an effect only if a weighting function
    has been specified with the ``--fmri`` option. If the value
    is in the *a priori* files falls below this value
    at a particular source space point, the source covariance matrix
    values are multiplied by the value specified with the ``--fmrioff`` option
    (default 0.1). Otherwise it is left unchanged.

``--fmrioff <*value*>``

    The value by which the source covariance elements are multiplied
    if the *a priori* weight falls below the threshold
    set with ``--fmrithresh`` , see above.

``--srccov <*name*>``

    Use this diagonal source covariance matrix. By default the source covariance
    matrix is a multiple of the identity matrix. This option is irrelevant
    to most users.

``--proj <*name*>``

    Include signal-space projection information from this file.

``--inv <*name*>``

    Save the inverse operator decomposition here. By default, the script looks
    for a file whose name is derived from the forward solution file by
    replacing its ending ``-fwd.fif`` by <*options*> ``-inv.fif`` , where
    <*options*> includes options ``--meg``, ``--eeg``, and ``--fixed`` with the double
    dashes replaced by single ones.

``--destdir <*directory*>``

    Optionally specifies a directory where the inverse operator will
    be stored.

.. note:: If bad channels are included in the calculation,    strange results may ensue. Therefore, it is recommended that the    data to be analyzed is carefully inspected with to assign the bad    channels correctly.

.. note:: For convenience, the MNE software includes bad-channel    designation files which can be used to ignore all magnetometer or    all gradiometer channels in Vectorview measurements. These files are    called ``vv_grad_only.bad`` and ``vv_mag_only.bad`` , respectively.    Both files are located in ``$MNE_ROOT/share/mne/templates`` .


.. _mne_forward_solution:

mne_forward_solution
====================

``--src <*name*>``

    Source space name to use. The name of the file must be specified exactly,
    including the directory. Typically, the source space files reside
    in $SUBJECTS_DIR/$SUBJECT/bem.

``--bem <*name*>``

    Specifies the BEM to be used. These files end with bem.fif or bem-sol.fif and
    reside in $SUBJECTS_DIR/$SUBJECT/bem. The former file contains only
    the BEM surface information while the latter files contain the geometry
    information precomputed with :ref:`mne_prepare_bem_model`,
    see :ref:`CHDJFHEB`. If precomputed geometry is not available,
    the linear collocation solution will be computed by mne_forward_solution .

``--origin <*x/mm*> : <*x/mm*> : <*z/mm*>``

    Indicates that the sphere model should be used in the forward calculations.
    The origin is specified in MEG head coordinates unless the ``--mricoord`` option
    is present. The MEG sphere model solution computed using the analytical
    Sarvas formula. For EEG, an approximative solution described in

``--eegmodels <*name*>``

    This option is significant only if the sphere model is used and
    EEG channels are present. The specified file contains specifications
    of the EEG sphere model layer structures as detailed in :ref:`CHDIAFIG`. If this option is absent the file ``$HOME/.mne/EEG_models`` will
    be consulted if it exists.

``--eegmodel <*model name*>``

    Specifies the name of the sphere model to be used for EEG. If this option
    is missing, the model Default will
    be employed, see :ref:`CHDIAFIG`.

``--eegrad <*radius/mm*>``

    Specifies the radius of the outermost surface (scalp) of the EEG sphere
    model, see :ref:`CHDIAFIG`. The default value is 90 mm.

``--eegscalp``

    Scale the EEG electrode locations to the surface of the outermost sphere
    when using the sphere model.

``--accurate``

    Use accurate MEG sensor coil descriptions. This is the recommended
    choice. More information

``--fixed``

    Compute the solution for sources normal to the cortical mantle only. This
    option should be used only for surface-based and discrete source
    spaces.

``--all``

    Compute the forward solution for all vertices on the source space.

``--label <*name*>``

    Compute the solution only for points within the specified label. Multiple
    labels can be present. The label files should end with ``-lh.label`` or ``-rh.label`` for
    left and right hemisphere label files, respectively. If ``--all`` flag
    is present, all surface points falling within the labels are included.
    Otherwise, only decimated points with in the label are selected.

``--mindist <*dist/mm*>``

    Omit source space points closer than this value to the inner skull surface.
    Any source space points outside the inner skull surface are automatically
    omitted. The use of this option ensures that numerical inaccuracies
    for very superficial sources do not cause unexpected effects in
    the final current estimates. Suitable value for this parameter is
    of the order of the size of the triangles on the inner skull surface.
    If you employ the seglab software to create the triangulations, this
    value should be about equal to the wish for the side length of the
    triangles.

``--mindistout <*name*>``

    Specifies a file name to contain the coordinates of source space points
    omitted due to the ``--mindist`` option.

``--mri <*name*>``

    The name of the MRI description file containing the MEG/MRI coordinate
    transformation. This file was saved as part of the alignment procedure
    outlined in :ref:`CHDBEHDC`. These files typically reside in ``$SUBJECTS_DIR/$SUBJECT/mri/T1-neuromag/sets`` .

``--trans	 <*name*>``

    The name of a text file containing the 4 x 4 matrix for the coordinate transformation
    from head to mri coordinates. With ``--trans``, ``--mri`` option is not
    required.

``--notrans``

    The MEG/MRI coordinate transformation is taken as the identity transformation, *i.e.*,
    the two coordinate systems are the same. This option is useful only
    in special circumstances. If more than one of the ``--mri`` , ``--trans`` ,
    and ``--notrans`` options are specified, the last one remains
    in effect.

``--mricoord``

    Do all computations in the MRI coordinate system. The forward solution
    matrix is not affected by this option if the source orientations
    are fixed to be normal to the cortical mantle. If all three source components
    are included, the forward three source orientations parallel to
    the coordinate axes is computed. If ``--mricoord`` is present, these
    axes correspond to MRI coordinate system rather than the default
    MEG head coordinate system. This option is useful only in special
    circumstances.

``--meas <*name*>``

    This file is the measurement fif file or an off-line average file
    produced thereof. It is recommended that the average file is employed for
    evoked-response data and the original raw data file otherwise. This
    file provides the MEG sensor locations and orientations as well as
    EEG electrode locations as well as the coordinate transformation between
    the MEG device coordinates and MEG head-based coordinates.

``--fwd <*name*>``

    This file will contain the forward solution as well as the coordinate transformations,
    sensor and electrode location information, and the source space
    data. A name of the form <*name*>-fwd.fif is
    recommended.

``--meg``

    Compute the MEG forward solution.

``--eeg``

    Compute the EEG forward solution.

``--grad``

    Include the derivatives of the fields with respect to the dipole
    position coordinates to the output, see :ref:`BJEFEJJG`.


.. _mne_inverse_operator:

mne_inverse_operator
====================

``--meg``

    Employ MEG data in the calculation of the estimates.

``--eeg``

    Employ EEG data in the calculation of the estimates. Note: The EEG
    computations have not been thoroughly tested at this time.

``--fixed``

    Use fixed source orientations normal to the cortical mantle. By default,
    the source orientations are not constrained.

``--loose <amount>``

    Employ a loose orientation constraint (LOC). This means that the source
    covariance matrix entries corresponding to the current component
    normal to the cortex are set equal to one and the transverse components
    are set to <*amount*> . Recommended
    value of amount is 0.2...0.6.

``--loosevar <amount>``

    Use an adaptive loose orientation constraint. This option can be
    only employed if the source spaces included in the forward solution
    have the patch information computed, see :ref:`setting_up_source_space`.

``--fwd <name>``

    Specifies the name of the forward solution to use.

``--noisecov <name>``

    Specifies the name of the noise-covariance matrix to use. If this
    file contains a projection operator, attached by :ref:`mne_browse_raw` and :ref:`mne_process_raw`,
    no additional projection vectors can be added with the ``--proj`` option. For
    backward compatibility, ``--senscov`` can be used as a synonym for ``--noisecov``.

``--noiserank <value>``

    Specifies the rank of the noise covariance matrix explicitly rather than
    trying to reduce it automatically. This option is seldom needed,

``--gradreg <value>``

    Regularize the planar gradiometer section (channels for which the unit
    of measurement is T/m) of the noise-covariance matrix by the given
    amount. The value is restricted to the range 0...1. For details, see :ref:`cov_regularization`.

``--magreg <value>``

    Regularize the magnetometer and axial gradiometer section (channels
    for which the unit of measurement is T) of the noise-covariance matrix
    by the given amount. The value is restricted to the range 0...1.
    For details, see :ref:`cov_regularization`.

``--eegreg <value>``

    Regularize the EEG section of the noise-covariance matrix by the given
    amount. The value is restricted to the range 0...1. For details, see :ref:`cov_regularization`.

``--diagnoise``

    Omit the off-diagonal terms from the noise-covariance matrix in
    the computations. This may be useful if the amount of signal-free
    data has been insufficient to calculate a reliable estimate of the
    full noise-covariance matrix.

``--srccov <name>``

    Specifies the name of the diagonal source-covariance matrix to use.
    By default the source covariance matrix is a multiple of the identity matrix.
    This option can be employed to incorporate the fMRI constraint.
    The software to create a source-covariance matrix file from fMRI
    data will be provided in a future release of this software package.

``--depth``

    Employ depth weighting. For details, see :ref:`depth_weighting`.

``--weightexp <value>``

    This parameter determines the steepness of the depth weighting function
    (default = 0.8). For details, see :ref:`depth_weighting`.

``--weightlimit <value>``

    Maximum relative strength of the depth weighting (default = 10). For
    details, see :ref:`depth_weighting`.

``--fmri <name>``

    With help of this w file, an *a priori* weighting
    can be applied to the source covariance matrix. The source of the
    weighting is usually fMRI but may be also some other data, provided
    that the weighting  can be expressed as a scalar value on the cortical
    surface, stored in a w file. It is recommended that this w file
    is appropriately smoothed (see :ref:`CHDEBAHH`) in mne_analyze , tksurfer or
    with mne_smooth_w to contain
    nonzero values at all vertices of the triangular tessellation of
    the cortical surface. The name of the file given is used as a stem of
    the w files. The actual files should be called <*name*> ``-lh.pri`` and <*name*> ``-rh.pri`` for
    the left and right hemsphere weight files, respectively. The application
    of the weighting is discussed in :ref:`mne_fmri_estimates`.

``--fmrithresh <value>``

    This option is mandatory and has an effect only if a weighting function
    has been specified with the ``--fmri`` option. If the value
    is in the *a priori* files falls below this value
    at a particular source space point, the source covariance matrix
    values are multiplied by the value specified with the ``--fmrioff`` option
    (default 0.1). Otherwise it is left unchanged.

``--fmrioff <value>``

    The value by which the source covariance elements are multiplied
    if the *a priori* weight falls below the threshold
    set with ``--fmrithresh`` , see above.

``--bad <name>``

    A text file to designate bad channels, listed one channel name on each
    line of the file. If the noise-covariance matrix specified with the ``--noisecov`` option
    contains projections, bad channel lists can be included only if
    they specify all channels containing non-zero entries in a projection
    vector. For example, bad channels can usually specify all magnetometers
    or all gradiometers since the projection vectors for these channel
    types are completely separate. Similarly, it is possible to include
    MEG data only or EEG data only by using only one of ``--meg`` or ``--eeg`` options
    since the projection vectors for MEG and EEG are always separate.

``--surfsrc``

    Use a source coordinate system based on the local surface orientation
    at the source location. By default, the three dipole components are
    pointing to the directions of the x, y, and z axis of the coordinate system
    employed in the forward calculation (usually the MEG head coordinate
    frame). This option changes the orientation so that the first two
    source components lie in the plane normal to the surface normal
    at the source location and the third component is aligned with it.
    If patch information is available in the source space, the normal
    is the average patch normal, otherwise the vertex normal at the source
    location is used. If the ``--loose`` or ``--loosevar`` option
    is employed, ``--surfsrc`` is implied.

``--exclude <name>``

    Exclude the source space points defined by the given FreeSurfer 'label' file
    from the source reconstruction. This is accomplished by setting
    the corresponding entries in the source-covariance matrix equal
    to zero. The name of the file should end with ``-lh.label``
    if it refers to the left hemisphere and with ``-rh.label`` if
    it lists points in the right hemisphere, respectively.

``--proj <name>``

    Include signal-space projection (SSP) information from this file. For information
    on SSP, see :ref:`CACCHABI`. If the projections are present in
    the noise-covariance matrix, the ``--proj`` option is
    not allowed.

``--csd``

    Compute the inverse operator for surface current densities instead
    of the dipole source amplitudes. This requires the computation of patch
    statistics for the source space. Since this computation is time consuming,
    it is recommended that the patch statistics are precomputed and
    the source space file containing the patch information is employed
    already when the forward solution is computed, see :ref:`setting_up_source_space` and :ref:`BABCHEJD`.
    For technical details of the patch information, please consult :ref:`patch_stats`. This option is considered experimental at
    the moment.

``--inv <name>``

    Save the inverse operator decomposition here.


.. _mne_make_movie:

mne_make_movie
==============

Input files
-----------

``--inv <*name*>``

    Load the inverse operator decomposition from here.

``--meas <*name*>``

    Load the MEG or EEG data from this file.

``--set <*number*>``

    The data set (condition) number to load. This is the sequential
    number of the condition. You can easily see the association by looking
    at the condition list in mne_analyze when
    you load the file.

``--stcin <*name*>``

    Specifies an stc file to read as input.

Times and baseline
------------------

``--tmin <*time/ms*>``

    Specifies the starting time employed in the analysis. If ``--tmin`` option
    is missing the analysis starts from the beginning of the epoch.

``--tmax <*time/ms*>``

    Specifies the finishing time employed in the analysis. If ``--tmax`` option
    is missing the analysis extends to the end of the epoch.

``--tstep <*step/ms*>``

    Time step between consequtive movie frames, specified in milliseconds.

``--integ  <*:math:`\Delta`t/ms*>``

    Integration time for each frame. Defaults to zero. The integration will
    be performed on sensor data. If the time specified for a frame is :math:`t_0`,
    the integration range will be :math:`t_0 - \Delta t/2 \leq t \leq t_0 + \Delta t/2`.

``--pick <*time/ms*>``

    Pick a time for the production of rgb, tif, jpg, png, or w files.
    Several pick options may be present. The time must be with in the
    analysis interval, indicated by the ``--tmin`` and ``--tmax`` options.
    The ``--rgb`` , ``--tif`` , ``--jpg`` , ``--png`` , and ``--w`` options
    control which file types are actually produced. When a ``--pick`` option
    is encountered, the effect of any preceding ``--pickrange`` option
    is ignored.

``--pickrange``

    All previous ``-pick`` options will be ignored. Instead,
    snapshots are produced as indicated by the ``--tmin`` , ``--tmax`` ,
    and ``--tstep`` options. This is useful, *e.g.*,
    for producing input for scripts merging the individual graphics
    snapshots into a composite "filmstrip" reprensentation.
    However, such scripts are not yet part of the MNE software.

``--bmin <*time/ms*>``

    Specifies the starting time of the baseline. In order to activate
    baseline correction, both ``--bmin`` and ``--bmax`` options
    must be present.

``--bmax <*time/ms*>``

    Specifies the finishing time of the baseline.

``--baselines <*file_name*>``

    Specifies a file which contains the baseline settings. Each line
    of the file should contain a name of a channel, followed by the
    baseline value, separated from the channel name by a colon. The
    baseline values must be specified in basic units, i.e., Teslas/meter
    for gradiometers, Teslas for magnetometers, and Volts for EEG channels.
    If some channels are missing from the baseline file, warning messages are
    issued: for these channels, the ``--bmin`` and ``--bmax`` settings will
    be used.

Options controlling the estimates
---------------------------------

``--nave <*value*>``

    Specifies the effective number of averaged epochs in the input data, :math:`L_{eff}`,
    as discussed in :ref:`CBBDGIAE`. If the input data file is
    one produced by :ref:`mne_browse_raw` or :ref:`mne_process_raw`, the
    number of averages is correct in the file. However, if subtractions
    or some more complicated combinations of simple averages are produced,
    e.g., by  using the xplotter software,
    the number of averages should be manually adjusted along the guidelines
    given in :ref:`CBBDGIAE`. This is accomplished either by
    employing this flag or by adjusting the number of averages in the
    data file with help of the utility mne_change_nave .

``--snr <*value*>``

    An estimate for the amplitude SNR. The regularization parameter will
    be set as :math:`\lambda^2 = 1/SNR^2`. The default value is
    SNR = 3. Automatic selection of the regularization parameter is
    currently not supported.

``--spm``

    Calculate the dSPM instead of the expected current value.

``--sLORETA``

    Calculate the noise-normalized estimate using the sLORETA approach.
    sLORETA solutions have in general a smaller location bias than either
    the expected current (MNE) or the dSPM.

``--signed``

    Indicate the current direction with respect to the cortex outer
    normal by sign. Currents flowing out of the cortex are thus considered
    positive (warm colors) and currents flowing into the cortex negative (cold
    colors).

``--picknormalcomp``

    The components of the estimates corresponding to directions tangential
    with the cortical mantle are zeroed out.

.. _CBBBBHIF:

Visualization options
---------------------

``--subject <*subject*>``

    Specifies the subject whose MRI data is employed in the visualization.
    This must be the same subject that was used for computing the current
    estimates. The environment variable SUBJECTS_DIR must be set to
    point to a locations where the subjects are to be found.

``--morph <*subject*>``

    Morph the data to to the cortical surface of another subject. The Quicktime
    movie, stc-file, graphics snapshot, and w-file outputs are affected
    by this option, *i.e.*, they will take the morphing
    into account and will represent the data on the cortical surface
    of the subject defined with this option. The stc files morphed to
    a single subject's cortical surface are used by mne_average_estimates to
    combine data from different subjects.
    If morphing is selected appropriate smoothing must be specified
    with the ``--smooth`` option. The morphing process can
    be made faster by precomputing the necessary morphing maps with mne_make_morph_maps ,
    see :ref:`CHDBBHDH`. More information about morphing and averaging
    can be found in :ref:`ch_morph`.

``--morphgrade <*number*>``

    Adjusts the number of vertices in the stc files produced when morphing
    is in effect. By default the number of vertices is 10242 corresponding
    to --morphgrade value 5. Allowed values are 3, 4, 5, and 6 corresponding
    to 642, 2562, 10242, and 40962 vertices, respectively.

``--surface <*surface name*>``

    Name of the surface employed in the visualization. The default is inflated .

``--curv <*name*>``

    Specify a nonstandard curvature file name. The default curvature files
    are ``lh.curv`` and ``rh.curv`` . With this option,
    the names become ``lh.`` <*name*> and ``rh.`` <*name*> .

``--patch <*name*> [: <*angle/deg*> ]``

    Specify the name of a surface patch to be used for visualization instead
    of the complete cortical surface. A complete name of a patch file
    in the FreeSurface surf directory must be given. The name should
    begin with lh or rh to allow association of the patch with a hemisphere.
    Maximum of two ``--patch`` options can be in effect, one patch for each
    hemisphere. If the name refers to a flat patch, the name can be
    optionally followed by a colon and a rotation angle in degrees.
    The flat patch will be then rotated counterclockwise by this amount
    before display. You can check a suitable value for the rotation
    angle by loading the patch interactively in mne_analyze .

``--width <*value*>``

    Width of the graphics output frames in pixels. The default width
    is 600 pixels.

``--height <*value*>``

    Height of the graphics output frames in pixels. The default height
    is 400 pixels.

``--mag <*factor*>``

    Magnify the the visualized scene by this factor.

``--lh``

    Select the left hemisphere for graphics output. By default, both hemisphere
    are processed.

``--rh``

    Select the right hemisphere for graphics output. By default, both hemisphere
    are processed.

``--view <*name*>``

    Select the name of the view for mov, rgb, and tif graphics output files.
    The default viewnames, defined in ``$MNE_ROOT/share/mne/mne_analyze/eyes`` ,
    are *lat* (lateral), *med* (medial), *ven* (ventral),
    and *occ* (occipital). You can override these
    defaults by creating the directory .mne under your home directory
    and copying the eyes file there. Each line of the eyes file contais
    the name of the view, the viewpoint for the left hemisphere, the
    viewpoint for the right hemisphere, left hemisphere up vector, and
    right hemisphere up vector. The entities are separated by semicolons.
    Lines beginning with the pound sign (#) are considered to be comments.

``--smooth <*nstep*>``

    Number of smoothsteps to take when producing the output frames. Depending
    on the source space decimation, an appropriate number is 4 - 7.
    Smoothing does not have any effect for the original brain if stc
    files are produced. However, if morphing is selected smoothing is
    mandatory even with stc output. For details of the smoothing procedure,
    see :ref:`CHDEBAHH`.

``--nocomments``

    Do not include the comments in the image output files or movies.

``--noscalebar``

    Do not include the scalebar in the image output files or movies.

``--alpha <*value*>``

    Adjust the opacity of maps shown on the cortical surface (0 = transparent,
    1 = totally opaque). The default value is 1.

Thresholding
------------

``--fthresh <*value*>``

    Specifies the threshold for the displayed colormaps. At the threshold,
    the overlaid color will be equal to the background surface color.
    For currents, the value will be multiplied by :math:`1^{-10}`.
    The default value is 8.

``--fmid <*value*>``

    Specifies the midpoint for the displayed colormaps. At this value, the
    overlaid color will be read (positive values) or blue (negative values).
    For currents, the value will be multiplied by :math:`1^{-10}`.
    The default value is 15.

``--fmax <*value*>``

    Specifies the maximum point for the displayed colormaps. At this value,
    the overlaid color will bright yellow (positive values) or light
    blue (negative values). For currents, the value will be multiplied
    by :math:`1^{-10}`. The default value is 20.

``--fslope <*value*>``

    Included for backwards compatibility. If this option is specified
    and ``--fmax`` option is *not* specified, :math:`F_{max} = F_{mid} + 1/F_{slope}`.

Output files
------------

``--mov <*name*>``

    Produce QuickTime movie files. This is the 'stem' of
    the ouput file name. The actual name is derived by stripping anything
    up to and including the last period from the end of <*name*> .
    According to the hemisphere, ``-lh`` or ``-rh`` is
    then appended. The name of the view is indicated with ``-`` <*viename*> .
    Finally, ``.mov`` is added to indicate a QuickTime output
    file. The movie is produced for all times as dictated by the ``--tmin`` , ``--tmax`` , ``--tstep`` ,
    and ``--integ`` options.

``--qual <*value*>``

    Quality of the QuickTime movie output. The default quality is 80 and
    allowed range is 25 - 100. The size of the movie files is a monotonously
    increasing function of the movie quality.

``--rate <*rate*>``

    Specifies the frame rate of the QuickTime movies. The default value is :math:`1/(10t_{step})`,
    where :math:`t_{step}` is the time between subsequent
    movie frames produced in seconds.

``--rgb <*name*>``

    Produce rgb snapshots. This is the 'stem' of the
    ouput file name. The actual name is derived by stripping anything
    up to and including the last period from the end of <*name*> .
    According to the hemisphere, ``-lh`` or ``-rh`` is
    then appended. The name of the view is indicated with ``-`` <*viename*> .
    Finally, ``.rgb`` is added to indicate an rgb output file.
    Files are produced for all picked times as dictated by the ``--pick`` and ``--integ`` options.

``--tif <*name*>``

    Produce tif snapshots. This is the 'stem' of the
    ouput file name. The actual name is derived by stripping anything
    up to and including the last period from the end of <*name*> .
    According to the hemisphere, ``-lh`` or ``-rh`` is
    then appended. The name of the view is indicated with ``-`` <*viename*> .
    Finally, ``.tif`` is added to indicate an rgb output file.
    Files are produced for all picked times as dictated by the ``--pick`` and ``--integ`` options.
    The tif output files are *not* compressed. Pass
    the files through an image processing program to compress them.

``--jpg <*name*>``

    Produce jpg snapshots. This is the 'stem' of the
    ouput file name. The actual name is derived by stripping anything
    up to and including the last period from the end of <*name*> .
    According to the hemisphere, ``-lh`` or ``-rh`` is
    then appended. The name of the view is indicated with ``-`` <*viename*> .
    Finally, ``.jpg`` is added to indicate an rgb output file.
    Files are produced for all picked times as dictated by the ``--pick`` and ``--integ`` options.

``--png <*name*>``

    Produce png snapshots. This is the 'stem' of the
    ouput file name. The actual name is derived by stripping anything
    up to and including the last period from the end of <*name*> .
    According to the hemisphere, ``-lh`` or ``-rh`` is
    then appended. The name of the view is indicated with ``-`` <*viename*> .
    Finally, ``.png`` is added to indicate an rgb output file.
    Files are produced for all picked times as dictated by the ``--pick`` and ``--integ`` options.

``--w <*name*>``

    Produce w file snapshots. This is the 'stem' of
    the ouput file name. The actual name is derived by stripping anything
    up to and including the last period from the end of <*name*> .
    According to the hemisphere, ``-lh`` .w or ``-rh`` .w
    is then appended. Files are produced for all picked times as dictated
    by the ``--pick`` and ``--integ`` options.

``--stc <*name*>``

    Produce stc files for either the original subject or the one selected with
    the ``--morph`` option. These files will contain data only
    for the decimated locations. If morphing is selected, appropriate
    smoothing is mandatory. The morphed maps will be decimated with
    help of a subdivided icosahedron so that the morphed stc files will
    always contain 10242 vertices. These morphed stc files can be easily
    averaged together, e.g., in Matlab since they always contain an
    identical set of vertices.

``--norm <*name*>``

    Indicates that a separate w file
    containing the noise-normalization values will be produced. The
    option ``--spm`` must also be present. Nevertheless, the
    movies and stc files output will
    contain MNE values. The noise normalization data files will be called <*name*>- <*SNR*> ``-lh.w`` and <*name*>- <*SNR*> ``-rh.w`` .

.. _CBBHHCEF:

Label processing
----------------

``--label <*name*>``

    Specifies a label file to process. For each label file, the values
    of the computed estimates are listed in text files. The label files
    are produced by tksurfer or mne_analyze and
    specify regions of interests (ROIs). A label file name should end
    with ``-lh.label`` for left-hemisphere ROIs and with ``-rh.label`` for
    right-hemisphere ones. The corresponding output files are tagged
    with ``-lh-`` <*data type*> ``.amp`` and ``-rh-`` <*data type*> ``.amp``, respectively. <*data type*> equals ``'mne`` ' for
    expected current data and ``'spm`` ' for
    dSPM data. Each line of the output file contains the waveform of
    the output quantity at one of the source locations falling inside
    the ROI. For more information about the label output formats, see :ref:`CACJJGFA`.

``--labelcoords``

    Include coordinates of the vertices in the output. The coordinates will
    be listed in millimeters in the coordinate system which was specified
    for the forward model computations. This option cannot be used with
    stc input files (``--stcin`` ) because the stc files do
    not contain the coordinates of the vertices.

``--labelverts``

    Include vertex numbers in the output. The numbers refer to the complete
    triangulation of the corresponding surface and are zero based. The
    vertex numbers are by default on the first row or first column of the
    output file depending on whether or not the ``--labeltimebytime`` option
    is present.

``--labeltimebytime``

    Output the label data time by time instead of the default vertex-by-vertex
    output.

``--labeltag <*tag*>``

    End the output files with the specified tag. By default, the output files
    will end with ``-mne.amp`` or ``-spm.amp`` depending
    on whether MNE or one of the noise-normalized estimates (dSPM or sLORETA)
    was selected.

``--labeloutdir <*directory*>``

    Specifies the directory where the output files will be located.
    By default, they will be in the current working directory.

``--labelcomments``

    Include comments in the output files. The comment lines begin with the
    percent sign to make the files compatible with Matlab.

``--scaleby <*factor*>``

    By default, the current values output to the files will be in the
    actual physical units (Am). This option allows scaling of the current
    values to other units. mne_analyze typically
    uses 1e10 to bring the numbers to a human-friendly scale.

Using stc file input
--------------------

The ``--stcin`` option allows input of stc files.
This feature has several uses:

- QuickTime movies can be produced from
  existing stc files without having to resort to EasyMeg.

- Graphics snapshot can be produced from existing stc files.

- Existing stc files can be temporally resampled with help of
  the ``--tmin`` , ``--tmax`` , ``--tstep`` ,
  and ``--integ`` options.

- Existing stc files can be morphed to another cortical surface
  by specifying the ``--morph`` option.

- Timecourses can be inquired and stored into text files with
  help of the ``--label`` options, see above.


.. _mne_make_source_space:

mne_make_source_space
=====================

``--subject <name>``

    Name of the subject.

``--morph <name>``

    Name of the subject to morph the source space to.

``--spacing <dist>``

    Approximate source space spacing in mm.

``--ico <grade>``

    Use the subdivided icosahedron or octahedron in downsampling instead of the --spacing option.

``--oct <grade>``

    Same as --ico -grade.

``--surf <names>``

    Surface file names (separated by colons)

``--src <name>``

    Name of the output file.


.. _mne_process_raw:

mne_process_raw
===============

``--cd <*dir*>``

    Change to this directory before starting.

``--raw <*name*>``

    Specifies the raw data file to be opened. This option is required.

``--grad <*number*>``

    Apply software gradient compensation of the given order to the data loaded
    with the ``--raw`` option. This option is effective only
    for data acquired with the CTF and 4D Magnes MEG systems. If orders
    different from zero are requested for Neuromag data, an error message appears
    and data are not loaded. Any compensation already existing in the
    file can be undone or changed to another order by using an appropriate ``--grad`` options.
    Possible orders are 0 (No compensation), 1 - 3 (CTF data), and 101
    (Magnes data). The same compensation will be applied to all loaded data
    files.

``--filtersize <*size*>``

    Adjust the length of the FFT to be applied in filtering. The number will
    be rounded up to the next power of two. If the size is :math:`N`,
    the corresponding length of time is :math:`N/f_s`,
    where :math:`f_s` is the sampling frequency
    of your data. The filtering procedure includes overlapping tapers
    of length :math:`N/2` so that the total FFT
    length will actually be :math:`2N`. This
    value cannot be changed after the program has been started.

``--highpass <*value/Hz*>``

    Highpass filter frequency limit. If this is too low with respect
    to the selected FFT length and, the data will not be highpass filtered. It
    is best to experiment with the interactive version to find the lowest applicable
    filter for your data. This value can be adjusted in the interactive
    version of the program. The default is 0, *i.e.*,
    no highpass filter apart from that used during the acquisition will
    be in effect.

``--highpassw <*value/Hz*>``

    The width of the transition band of the highpass filter. The default
    is 6 frequency bins, where one bin is :math:`f_s / (2N)`. This
    value cannot be adjusted in the interactive version of the program.

``--lowpass <*value/Hz*>``

    Lowpass filter frequency limit. This value can be adjusted in the interactive
    version of the program. The default is 40 Hz.

``--lowpassw <*value/Hz*>``

    The width of the transition band of the lowpass filter. This value
    can be adjusted in the interactive version of the program. The default
    is 5 Hz.

``--eoghighpass <*value/Hz*>``

    Highpass filter frequency limit for EOG. If this is too low with respect
    to the selected FFT length and, the data will not be highpass filtered.
    It is best to experiment with the interactive version to find the
    lowest applicable filter for your data. This value can be adjusted in
    the interactive version of the program. The default is 0, *i.e.*,
    no highpass filter apart from that used during the acquisition will
    be in effect.

``--eoghighpassw <*value/Hz*>``

    The width of the transition band of the EOG highpass filter. The default
    is 6 frequency bins, where one bin is :math:`f_s / (2N)`.
    This value cannot be adjusted in the interactive version of the
    program.

``--eoglowpass <*value/Hz*>``

    Lowpass filter frequency limit for EOG. This value can be adjusted in
    the interactive version of the program. The default is 40 Hz.

``--eoglowpassw <*value/Hz*>``

    The width of the transition band of the EOG lowpass filter. This value
    can be adjusted in the interactive version of the program. The default
    is 5 Hz.

``--filteroff``

    Do not filter the data. This initial value can be changed in the
    interactive version of the program.

``--digtrig <*name*>``

    Name of the composite digital trigger channel. The default value
    is 'STI 014'. Underscores in the channel name
    will be replaced by spaces.

``--digtrigmask <*number*>``

    Mask to be applied to the trigger channel values before considering them.
    This option is useful if one wants to set some bits in a don't care
    state. For example, some finger response pads keep the trigger lines
    high if not in use, *i.e.*, a finger is not in
    place. Yet, it is convenient to keep these devices permanently connected
    to the acquisition system. The number can be given in decimal or
    hexadecimal format (beginning with 0x or 0X). For example, the value
    255 (0xFF) means that only the lowest order byte (usually trigger
    lines 1 - 8 or bits 0 - 7) will be considered.

``--proj <*name*>``

    Specify the name of the file of the file containing a signal-space
    projection (SSP) operator. If ``--proj`` options are present
    the data file is not consulted for an SSP operator. The operator
    corresponding to average EEG reference is always added if EEG data
    are present.

``--projon``

    Activate the projections loaded. One of the options ``--projon`` or ``--projoff`` must
    be present on the mne_processs_raw command line.

``--projoff``

    Deactivate the projections loaded. One of the options ``--projon`` or ``--projoff`` must
    be present on the mne_processs_raw command line.

``--makeproj``

    Estimate the noise subspace from the data and create a new signal-space
    projection operator instead of using one attached to the data file
    or those specified with the ``--proj`` option. The following
    eight options define the parameters of the noise subspace estimation. More
    information on the signal-space projection can be found in :ref:`CACCHABI`.

``--projevent <*no*>``

    Specifies the events which identify the time points of interest
    for projector calculation. When this option is present, ``--projtmin`` and ``--projtmax`` are
    relative to the time point of the event rather than the whole raw
    data file.

``--projtmin <*time/s*>``

    Specify the beginning time for the calculation of the covariance matrix
    which serves as the basis for the new SSP operator. This option
    is required with ``--projevent`` and defaults to the beginning
    of the raw data file otherwise. This option is effective only if ``--makeproj`` or ``--saveprojtag`` options
    are present.

``--projtmax <*time/s*>``

    Specify the ending time for the calculation of the covariance matrix which
    serves as the basis for the new SSP operator. This option is required
    with ``--projevent`` and defaults to the end of the raw data
    file otherwise. This option is effective only if ``--makeproj`` or ``--saveprojtag`` options
    are present.

``--projngrad <*number*>``

    Number of SSP components to include for planar gradiometers (default
    = 5). This value is system dependent. For example, in a well-shielded
    quiet environment, no planar gradiometer projections are usually
    needed.

``--projnmag <*number*>``

    Number of SSP components to include for magnetometers / axial gradiometers
    (default = 8). This value is system dependent. For example, in a
    well-shielded quiet environment, 3 - 4 components are need
    while in a noisy environment with light shielding even more than
    8 components may be necessary.

``--projgradrej <*value/ fT/cm*>``

    Rejection limit for planar gradiometers in the estimation of the covariance
    matrix frfixom which the new SSP operator is derived. The default
    value is 2000 fT/cm. Again, this value is system dependent.

``--projmagrej <*value/ fT*>``

    Rejection limit for planar gradiometers in the estimation of the covariance
    matrix from which the new SSP operator is derived. The default value
    is 3000 fT. Again, this value is system dependent.

``--saveprojtag <*tag*>``

    This option defines the names of files to hold the SSP operator.
    If this option is present the ``--makeproj`` option is
    implied. The SSP operator file name is formed by removing the trailing ``.fif`` or ``_raw.fif`` from
    the raw data file name by appending  <*tag*> .fif
    to this stem. Recommended value for <*tag*> is ``-proj`` .

``--saveprojaug``

    Specify this option if you want to use the projection operator file output
    in the Elekta-Neuromag Signal processor (graph) software.

``--eventsout <*name*>``

    List the digital trigger channel events to the specified file. By default,
    only transitions from zero to a non-zero value are listed. If multiple
    raw data files are specified, an equal number of ``--eventsout`` options
    should be present. If the file name ends with .fif, the output will
    be in fif format, otherwise a text event file will be output.

``--allevents``

    List all transitions to file specified with the ``--eventsout`` option.

``--events <*name*>``

    Specifies the name of a fif or text format event file (see :ref:`CACBCEGC`) to be associated with a raw data file to be
    processed. If multiple raw data files are specified, the number
    of ``--events`` options can be smaller or equal to the
    number of raw data files. If it is equal, the event filenames will
    be associated with the raw data files in the order given. If it
    is smaller, the remaining raw data files for which an event file
    is not specified will *not* have an event file associated
    with them. The event file format is recognized from the file name:
    if it ends with ``.fif`` , the file is assumed to be in
    fif format, otherwise a text file is expected.

``--ave <*name*>``

    Specifies the name of an off-line averaging description file. For details
    of the format of this file, please consult :ref:`CACBBDGC`.
    If multiple raw data files are specified, the number of ``--ave`` options
    can be smaller or equal to the number of raw data files. If it is
    equal, the averaging description file names will be associated with
    the raw data files in the order given. If it is smaller, the last
    description file will be used for the remaining raw data files.

``--saveavetag <*tag*>``

    If this option is present and averaging is evoked with the ``--ave`` option,
    the outfile and logfile options in the averaging description file
    are ignored. Instead, trailing ``.fif`` or ``_raw.fif`` is
    removed from the raw data file name and <*tag*> ``.fif`` or <*tag*> ``.log`` is appended
    to create the output and log file names, respectively.

``--gave <*name*>``

    If multiple raw data files are specified as input and averaging
    is requested, the grand average over all data files will be saved
    to <*name*> .

``--cov <*name*>``

    Specify the name of a description file for covariance matrix estimation. For
    details of the format of this file, please see :ref:`CACEBACG`.
    If multiple raw data files are specified, the number of ``--cov`` options can
    be smaller or equal to the number of raw data files. If it is equal, the
    averaging description file names will be associated with the raw data
    files in the order given. If it is smaller, the last description
    file will be used for the remaining raw data files.

``--savecovtag <*tag*>``

    If this option is present and covariance matrix estimation is evoked with
    the ``--cov`` option, the outfile and logfile options in
    the covariance estimation description file are ignored. Instead,
    trailing ``.fif`` or ``_raw.fif`` is removed from
    the raw data file name and <*tag*> .fif or <*tag*> .log
    is appended to create the output and log file names, respectively.
    For compatibility with other MNE software scripts, ``--savecovtag -cov`` is recommended.

``--savehere``

    If the ``--saveavetag`` and ``--savecovtag`` options
    are used to generate the file output file names, the resulting files
    will go to the same directory as raw data by default. With this
    option the output files will be generated in the current working
    directory instead.

``--gcov <*name*>``

    If multiple raw data files are specified as input and covariance matrix estimation
    is requested, the grand average over all data files will be saved
    to <*name*> . The details of
    the covariance matrix estimation are given in :ref:`CACHAAEG`.

``--save <*name*>``

    Save a filtered and optionally down-sampled version of the data
    file to <*name*> . If multiple
    raw data files are specified, an equal number of ``--save`` options
    should be present. If <*filename*> ends
    with ``.fif`` or ``_raw.fif`` , these endings are
    deleted. After these modifications, ``_raw.fif`` is inserted
    after the remaining part of the file name. If the file is split
    into multiple parts (see ``--split`` option below), the
    additional parts will be called <*name*> ``-`` <*number*> ``_raw.fif``

``--split <*size/MB*>``

    Specifies the maximum size of the raw data files saved with the ``--save`` option.
    By default, the output is split into files which are just below
    2 GB so that the fif file maximum size is not exceed.

``--anon``

    Do not include any subject information in the output files created with
    the ``--save`` option.

``--decim <*number*>``

    The data are decimated by this factor before saving to the file
    specified with the ``--save`` option. For decimation to
    succeed, the data must be lowpass filtered to less than third of
    the sampling frequency effective after decimation.


.. _mne_redo_file:

mne_redo_file
=============

Usage: ``mne_redo_file file-to-redo``


.. _mne_redo_file_nocwd:

mne_redo_file_nocwd
===================

Usage: ``mne_redo_file_nocwd file-to-redo``


.. _mne_setup_forward_model:

mne_setup_forward_model
=======================

``--subject <*subject*>``

    Defines the name of the subject. This can be also accomplished
    by setting the SUBJECT environment variable.

``--surf``

    Use the FreeSurfer surface files instead of the default ASCII triangulation
    files. Please consult :ref:`BABDBBFC` for the standard file
    naming scheme.

``--noswap``

    Traditionally, the vertices of the triangles in 'tri' files
    have been ordered so that, seen from the outside of the triangulation,
    the vertices are ordered in clockwise fashion. The fif files, however,
    employ the more standard convention with the vertices ordered counterclockwise.
    Therefore, mne_setup_forward_model by
    default reverses the vertex ordering before writing the fif file.
    If, for some reason, you have counterclockwise-ordered tri files
    available this behavior can be turned off by defining ``--noswap`` .
    When the fif file is created, the vertex ordering is checked and
    the process is aborted if it is incorrect after taking into account
    the state of the swapping. Should this happen, try to run mne_setup_forward_model again including
    the ``--noswap`` flag. In particular, if you employ the seglab software
    to create the triangulations (see :ref:`create_bem_model`), the ``--noswap`` flag
    is required. This option is ignored if ``--surf`` is specified

``--ico <*number*>``

    This option is relevant (and required) only with the ``--surf`` option and
    if the surface files have been produced by the watershed algorithm.
    The watershed triangulations are isomorphic with an icosahedron,
    which has been recursively subdivided six times to yield 20480 triangles.
    However, this number of triangles results in a long computation
    time even in a workstation with generous amounts of memory. Therefore,
    the triangulations have to be decimated. Specifying ``--ico 4`` yields 5120 triangles per surface while ``--ico 3`` results
    in 1280 triangles. The recommended choice is ``--ico 4`` .

``--homog``

    Use a single compartment model (brain only) instead a three layer one
    (scalp, skull, and brain). Only the ``inner_skull.tri`` triangulation
    is required. This model is usually sufficient for MEG but invalid
    for EEG. If you are employing MEG data only, this option is recommended
    because of faster computation times. If this flag is specified,
    the options ``--brainc`` , ``--skullc`` , and ``--scalpc`` are irrelevant.

``--brainc <*conductivity/ S/m*>``

    Defines the brain compartment conductivity. The default value is 0.3 S/m.

``--skullc <*conductivity/ S/m*>``

    Defines the skull compartment conductivity. The default value is 0.006 S/m
    corresponding to a conductivity ratio 1/50 between the brain and
    skull compartments.

``--scalpc <*conductivity/ S/m*>``

    Defines the brain compartment conductivity. The default value is 0.3 S/m.

``--innershift <*value/mm*>``

    Shift the inner skull surface outwards along the vertex normal directions
    by this amount.

``--outershift <*value/mm*>``

    Shift the outer skull surface outwards along the vertex normal directions
    by this amount.

``--scalpshift <*value/mm*>``

    Shift the scalp surface outwards along the vertex normal directions by
    this amount.

``--nosol``

    Omit the BEM model geometry dependent data preparation step. This
    can be done later by running mne_setup_forward_model without the ``--nosol`` option.

``--model <*name*>``

    Name for the BEM model geometry file. The model will be created into
    the directory bem as <*name*>- ``bem.fif`` .	If
    this option is missing, standard model names will be used (see below).


.. _mne_setup_mri:

mne_setup_mri
=============

This command sets up the directories ``subjects/$SUBJECT/mri/T1-neuromag`` and
``subjects/$SUBJECT/mri/brain-neuromag`` .


.. _mne_setup_source_space:

mne_setup_source_space
======================

``--subject <*name*>``

    Name of the subject in SUBJECTS_DIR. In the absence of this option,
    the SUBJECT environment variable will be consulted. If it is not
    defined, mne_setup_source_space exits
    with an error.

``--morph <*name*>``

    Name of a subject in SUBJECTS_DIR. If this option is present, the source
    space will be first constructed for the subject defined by the --subject
    option or the SUBJECT environment variable and then morphed to this
    subject. This option is useful if you want to create a source spaces
    for several subjects and want to directly compare the data across
    subjects at the source space vertices without any morphing procedure
    afterwards. The drawback of this approach is that the spacing between
    source locations in the "morph" subject is not going
    to be as uniform as it would be without morphing.

``--surf <*name1*>: <*name2*>:...``

    FreeSurfer surface file names specifying the source surfaces, separated
    by colons.

``--spacing <*spacing/mm*>``

    Specifies the approximate grid spacing of the source space in mm.

``--ico <*number*>``

    Instead of using the traditional method for cortical surface decimation
    it is possible to create the source space using the topology of
    a recursively subdivided icosahedron ( <*number*> > 0)
    or an octahedron ( <*number*>  < 0).
    This method uses the cortical surface inflated to a sphere as a
    tool to find the appropriate vertices for the source space. The
    benefit of the ``--ico`` option is that the source space will have triangulation
    information between the decimated vertices included, which some
    future versions of MNE software may be able to utilize. The number
    of triangles increases by a factor of four in each subdivision,
    starting from 20 triangles in an icosahedron and 8 triangles in
    an octahedron. Since the number of vertices on a closed surface
    is :math:`n_{vert} = (n_{tri} + 4) / 2`, the number of vertices in
    the *k* th subdivision of an icosahedron and an
    octahedron are :math:`10 \cdot 4^k +2` and :math:`4_{k + 1} + 2`,
    respectively. The recommended values for <*number*> and
    the corresponding number of source space locations are listed in Table 3.1.

``--all``

    Include all nodes to the output. The active dipole nodes are identified
    in the fif file by a separate tag. If tri files were used as input
    the output file will also contain information about the surface
    triangulation. This option is always recommended to include complete
    information.

``--src <*name*>``

    Output file name. Use a name <*dir*>/<*name*>-src.fif

.. note:: If both ``--ico`` and ``--spacing`` options    are present the later one on the command line takes precedence.

.. note:: Due to the differences between the FreeSurfer    and MNE libraries, the number of source space points generated with    the ``--spacing`` option may be different between the current    version of MNE and versions 2.5 or earlier (using ``--spacing`` option    to mne_setup_source_space ) if    the FreeSurfer surfaces employ the (old) quadrangle format or if    there are topological defects on the surfaces. All new FreeSurfer    surfaces are specified as triangular tessellations and are e of    defects.


.. _mne_show_environment:

mne_show_environment
====================

Usage: ``mne_show_environment files``


Utility command-line arguments
##############################

.. _mne_add_patch_info:

mne_add_patch_info
==================

Purpose
-------

The utility mne_add_patch_info uses
the detailed cortical surface geometry information to add data about
cortical patches corresponding to each source space point. A new
copy of the source space(s) included in the input file is created
with the patch information included. In addition to the patch information, mne_add_patch_info can
optionally calculate distances, along the cortical surface, between
the vertices selected to the source space.

.. note:: Depending on the speed of your computer and the options selected, mne_add_patch_info takes 5 - 30 minutes to run.

.. _CJAGCDCC:

Command line options
--------------------

mne_add_patch_info accepts
the following command-line options:

``--verbose``

    Provide verbose output during the calculations.

``--dist  <*dist/mm*>``

    Invokes the calculation of distances between vertices included in
    the source space along the cortical surface. Only pairs whose distance in
    the three-dimensional volume is less than the specified distance are
    considered. For details, see :ref:`CJAIFJDD`, below.

``--src  <*name*>``

    The input source space file. The source space files usually end
    with ``-src.fif`` .

``--srcp  <*name*>``

    The output source space file which will contain the patch information.
    If the file exists it will overwritten without asking for permission.
    A recommended naming convention is to add the letter ``p`` after the
    source spacing included in the file name. For example, if the input
    file is ``mh-7-src.fif`` , a recommended output file name
    is ``mh-7p-src.fif`` .

``--w  <*name*>``

    Name of a w file, which will contain the patch area information. Two
    files will be created:  <*name*> ``-lh.w`` and  <*name*> ``-rh.w`` .
    The numbers in the files are patch areas in :math:`\text{mm}^2`.
    The source space vertices are marked with value 150.

``--labeldir  <*directory*>``

    Create a label file corresponding to each of the patches in the
    given directory. The directory must be created before running mne_add_patch_info .

.. _CJAIFJDD:

Computational details
---------------------

By default, mne_add_patch_info creates
a copy of the source space(s) with the following additional information
for each vertex in the original dense triangulation of the cortex:

- The number of the closest active source
  space vertex and

- The distance to this vertex.

This information can be used to determine, *e.g.*,
the sizes of the patches, their average normals, and the standard
deviation of the normal directions. This information is also returned
by the mne_read_source_space Matlab function as described in Table 10.28.

The ``--dist`` option to mne_add_patch_info invokes
the calculation of inter-vertex distances. These distances are computed
along the the cortical surface (usually the white matter) on which
the source space vertices are located.

Since the calculation of all possible distances would take
a very long time, the distance given with the ``--dist`` option allows
restriction to the neighborhood of each source space vertex. This
neighborhood is defined as the sphere around each source space vertex,
with radius given by the ``--dist`` option. Because the distance calculation
is done along the folded cortical surface whose details are given
by the dense triangulation of the cortical surface produced by FreeSurfer,
some of the distances computed will be larger than the value give
with --dist.


.. _mne_add_to_meas_info:

mne_add_to_meas_info
====================

Add new data to meas info.

``--add <name>``

    The file to add.

``--dest <name>``

    the destination file.


.. _mne_add_triggers:

mne_add_triggers
================

Purpose
-------

The utility mne_add_triggers modifies
the digital trigger channel (STI 014) in raw data files
to include additional transitions. Since the raw data file is modified,
it is possible to make irreversible changes. Use this utility with
caution. It is recommended that you never run mne_add_triggers on
an original raw data file.

Command line options
--------------------

mne_add_triggers accepts
the following command-line options:

``--raw  <*name*>``

    Specifies the raw data file to be modified.

``--trg  <*name*>``

    Specifies the trigger line modification list. This text file should
    contain two entries per line: the sample number and the trigger
    number to be added into the file. The number of the first sample
    in the file is zero. It is recommended that trigger numbers whose
    binary equivalent has lower eight bits equal to zero are used to
    avoid conflicts with the ordinary triggers occurring in the file.

``--delete``

    Delete the triggers defined by the trigger file instead of adding
    them. This enables changing the file to its original state, provided
    that the trigger file is preserved.

.. note:: Since :ref:`mne_browse_raw` and :ref:`mne_process_raw` can employ an event file which effectively adds new trigger instants, mne_add_triggers is    for the most part obsolete but it has been retained in the MNE software    suite for backward compatibility.



.. _mne_annot2labels:

mne_annot2labels
================

The utility mne_annot2labels converts
cortical parcellation data into a set of labels. The parcellation
data are read from the directory ``$SUBJECTS_DIR/$SUBJECT/label`` and
the resulting labels are written to the current directory. mne_annot2labels requires
that the environment variable ``$SUBJECTS_DIR`` is set.
The command line options for mne_annot2labels are:

``--subject  <*name*>``

    Specifies the name of the subject. If this option is not present
    the ``$SUBJECT`` environment variable is consulted. If
    the subject name cannot be determined, the program quits.

``--parc  <*name*>``

    Specifies the parcellation name to convert. The corresponding parcellation
    file names will be ``$SUBJECTS_DIR/$SUBJECT/label/``  <*hemi*> ``h.``  <*name*> ``.annot`` where  <*hemi*> is ``l`` or ``r`` for the
    left and right hemisphere, respectively.


.. _mne_anonymize:

mne_anonymize
=============

Depending no the settings during acquisition in the Elekta-Neuromag EEG/MEG
systems the data files may contain subject identifying information
in unencrypted form. The utility mne_anonymize was
written to clear tags containing such information from a fif file.
Specifically, this utility removes the following tags from the fif
file:

.. _CHDEHBCG:

.. table:: Tags cleared by mne_anonymize .

    ========================  ==============================================
    Tag                       Description
    ========================  ==============================================
    FIFF_SUBJ_FIRST_NAME      First name of the subject
    FIFF_SUBJ_MIDDLE_NAME     Middle name of the subject
    FIFF_SUBJ_LAST_NAME       Last name of the subject
    FIFF_SUBJ_BIRTH_DAY       Birthday of the subject (Julian day number)
    FIFF_SUBJ_SEX             The sex of the subject
    FIFF_SUBJ_HAND            Handedness of the subject
    FIFF_SUBJ_WEIGHT          Weight of the subject in kg
    FIFF_SUBJ_HEIGHT          Height of the subject in m
    FIFF_SUBJ_COMMENT         Comment about the subject
    ========================  ==============================================

.. note:: mne_anonymize normally    keeps the FIFF_SUBJ_HIS_ID tag which can be used to identify the    subjects uniquely after the information listed in :ref:`CHDEHBCG` have    been removed. If the ``--his`` option is specified on the command line,    the FIFF_SUBJ_HIS_ID tag will be removed as well. The data of the    tags listed in :ref:`CHDEHBCG` and the optional FIFF_SUBJ_HIS_ID    tag are overwritten with zeros and the space claimed by omitting    these tags is added to the free space list of the file. Therefore, after mne_anonymize has    processed a data file there is no way to recover the removed information.    Use this utility with caution.

mne_anonymize recognizes
the following command-line options:

``--his``

    Remove the FIFF_SUBJ_HIS_ID tag as well, see above.

``--file  <*name*>``

    Specifies the name of the file to be modified.

.. note:: You need write permission to the file to be    processed.


.. _mne_average_forward_solutions:

mne_average_forward_solutions
=============================

``--fwd <*name*> :[ <*weight*> ]``

    Specifies a forward solution to include. If no weight is specified,
    1.0 is assumed. In the averaging process the weights are divided
    by their sum. For example, if two forward solutions are averaged
    and their specified weights are 2 and 3, the average is formed with
    a weight of 2/5 for the first solution and 3/5 for the second one.

``--out <*name*>``

    Specifies the output file which will contain the averaged forward solution.


.. _mne_brain_vision2fiff:

mne_brain_vision2fiff
=====================

The utility mne_brain_vision2fiff was
created to import BrainVision EEG data. This utility also helps
to import the eXimia (Nexstim) TMS-compatible EEG system data to
the MNE software. The utility uses an optional fif file containing
the head digitization data to allow source modeling. The MNE Matlab
toolbox contains the function fiff_write_dig_file to
write a digitization file based on digitization data available in
another format, see :ref:`ch_matlab`.

.. note::

    mne_brain_vision2fiff reads events from the ``vmrk`` file referenced in the
    ``vhdr`` file, but it only includes events whose "Type" is ``Stimulus`` and
    whose "description" is given by ``S<number>``. All other events are ignored.


The command-line options of mne_brain_vision2fiff are:

``--header <*name*>``

    The name of the BrainVision header file. The extension of this file
    is ``vhdr`` . The header file typically refers to a marker
    file (``vmrk`` ) which is automatically processed and a
    digital trigger channel (STI 014) is formed from the marker information.
    The ``vmrk`` file is ignored if the ``--eximia`` option
    is present.

``--dig <*name*>``

    The name of the fif file containing the digitization data.

``--orignames``

    Use the original EEG channel labels. If this option is absent the EEG
    channels will be automatically renamed to EEG 001, EEG 002, *etc.*

``--eximia``

    Interpret this as an eXimia data file. The first three channels
    will be thresholded and interpreted as trigger channels. The composite
    digital trigger channel will be composed in the same way as in the
    :ref:`mne_kit2fiff` utility. In addition, the fourth channel
    will be assigned as an EOG channel. This option is normally used
    by the :ref:`mne_eximia2fiff` script.

``--split <*size/MB*>``

    Split the output data into several files which are no more than <*size*> MB.
    By default, the output is split into files which are just below
    2 GB so that the fif file maximum size is not exceeded.

``--out <*filename*>``

    Specifies the name of the output fif format data file. If <*filename*> ends
    with ``.fif`` or ``_raw.fif`` , these endings are
    deleted. After these modifications, ``_raw.fif`` is inserted
    after the remaining part of the file name. If the file is split
    into multiple parts, the additional parts will be called
    <*name*> ``-`` <*number*> ``_raw.fif`` .


.. _mne_change_baselines:

mne_change_baselines
====================

The utility mne_change_baselines computes
baseline values and applies them to an evoked-response data file.
The command-line options are:

``--in  <*name*>``

    Specifies the input data file.

``--set  <*number*>``

    The data set number to compute baselines from or to apply baselines
    to. If this option is omitted, all average data sets in the input file
    are processed.

``--out  <*name*>``

    The output file.

``--baselines  <*name*>``

    Specifies a text file which contains the baseline values to be applied. Each
    line should contain a channel name, colon, and the baseline value
    given in 'native' units (T/m, T, or V). If this
    option is encountered, the limits specified by previous ``--bmin`` and ``--bmax`` options will not
    have an effect.

``--list  <*name*>``

    Specifies a text file to contain the baseline values. Listing is
    provided only if a specific data set is selected with the ``--set`` option.

``--bmin  <*value/ms*>``

    Lower limit of the baseline. Effective only if ``--baselines`` option is
    not present. Both ``--bmin`` and ``--bmax`` must
    be present to compute the baseline values. If either ``--bmin`` or ``--bmax`` is
    encountered, previous ``--baselines`` option will be ignored.

``--bmax  <*value/ms*>``

    Upper limit of the baseline.


.. _mne_change_nave:

mne_change_nave
===============

Usage: ``mne_change_nave --nave <number> <meas file> ...``


.. _mne_check_eeg_locations:

mne_check_eeg_locations
=======================

Some versions of the Neuromag acquisition software did not
copy the EEG channel location information properly from the Polhemus
digitizer information data block to the EEG channel information
records if the number of EEG channels exceeds 60. The purpose of mne_check_eeg_locations is
to detect this problem and fix it, if requested. The command-line
options are:

``--file  <*name*>``

    Specify the measurement data file to be checked or modified.

``--dig  <*name*>``

    Name of the file containing the Polhemus digitizer information. Default
    is the data file name.

``--fix``

    By default mne_check_eeg_locations only
    checks for missing EEG locations (locations close to the origin).
    With --fix mne_check_eeg_locations reads
    the Polhemus data from the specified file and copies the EEG electrode
    location information to the channel information records in the measurement
    file. There is no harm running mne_check_eeg_locations on
    a data file even if the EEG channel locations were correct in the
    first place.


.. _mne_check_surface:

mne_check_surface
=================
This program just reads a surface file to check whether it is valid.

``--surf <name>``

    The input file (FreeSurfer surface format).

``--bem <name>``

    The input file (a BEM fif file)

``--id <id>``

    Surface id to list (default : 4)

        * 4 for outer skin (scalp) surface
	  * 3 for outer skull surface
	  * 1 for inner skull surface


``--checkmore``

    Do more thorough testing


.. _mne_collect_transforms:

mne_collect_transforms
======================

The utility mne_collect_transforms collects
coordinate transform information from various sources and saves
them into a single fif file. The coordinate transformations used
by MNE software are summarized in Figure 5.1. The output
of mne_collect_transforms may
include all transforms referred to therein except for the sensor
coordinate system transformations :math:`T_{s_1} \dotso T_{s_n}`.
The command-line options are:

``--meas <*name*>``

    Specifies a measurement data file which provides :math:`T_1`.
    A forward solution or an inverse operator file can also be specified
    as implied by Table 5.1.

``--mri <*name*>``

    Specifies an MRI description or a standalone coordinate transformation
    file produced by mne_analyze which
    provides :math:`T_2`. If the ``--mgh`` option
    is not present mne_collect_transforms also
    tries to find :math:`T_3`, :math:`T_4`, :math:`T_-`,
    and :math:`T_+` from this file.

``--mgh <*name*>``

    An MRI volume volume file in mgh or mgz format.
    This file provides :math:`T_3`. The transformation :math:`T_4` will
    be read from the talairach.xfm file referred to in the MRI volume.
    The fixed transforms :math:`T_-` and :math:`T_+` will
    also be created.

``--out <*name*>``

    Specifies the output file. If this option is not present, the collected transformations
    will be output on screen but not saved.


.. _mne_compensate_data:

mne_compensate_data
===================

``--in <*name*>``

    Specifies the input data file.

``--out <*name*>``

    Specifies the output data file.

``--grad <*number*>``

    Specifies the desired compensation grade in the output file. The value
    can be 1, 2, 3, or 101. The values starting from 101 will be used
    for 4D Magnes compensation matrices.

.. note:: Only average data is included in the output. Evoked-response data files produced with mne_browse_raw or mne_process_raw may    include standard errors of mean, which can not be re-compensated    using the above method and are thus omitted.

.. note:: Raw data cannot be compensated using mne_compensate_data . For this purpose, load the data to mne_browse_raw or mne_process_raw , specify    the desired compensation grade, and save a new raw data file.


.. _mne_copy_processing_history:

mne_copy_processing_history
===========================

In order for the inverse operator calculation to work correctly
with data processed with the Elekta-Neuromag Maxfilter (TM) software,
the so-called *processing history* block must
be included in data files. Previous versions of the MNE Matlab functions
did not copy processing history to files saved. As of March 30,
2009, the Matlab toolbox routines fiff_start_writing_raw and fiff_write_evoked have
been enchanced to include these data to the output file as appropriate.
If you have older raw data files created in Matlab from input which
has been processed Maxfilter, it is necessary to copy the *processing
history* block from the original to modified raw data
file using the mne_copy_processing_history utility described
below. The raw data processing programs mne_browse_raw and mne_process_raw have
handled copying of the processing history since revision 2.5 of
the MNE software.

mne_copy_processing_history is
simple to use:

``mne_copy_processing_history --from``  <*from*> ``--to``  <*to*> ,

where  <*from*> is an
original raw data file containing the processing history and  <*to*> is
a file output with older MNE Matlab routines. Be careful: this operation
cannot be undone. If the  <*from*> file
does not have the processing history block or the  <*to*> file
already has it, the destination file remains unchanged.


.. _mne_convert_dig_data:

mne_convert_dig_data
====================

Converts Polhemus digitization data between different file formats.
The input formats are:

``fif``

    The
    standard format used in MNE. The digitization data are typically
    present in the measurement files.

``hpts``

    A text format which is a translation
    of the fif format data, see :ref:`CJADJEBH` below.

``elp``

    A text format produced by the *Source
    Signal Imaging, Inc.* software. For description of this "probe" format,
    see http://www.sourcesignal.com/formats_probe.html.

The data can be output in fif and hpts formats.
Only the last command-line option specifying an input file will
be honored. Zero or more output file options can be present on the
command line.

.. note:: The elp and hpts input    files may contain textual EEG electrode labels. They will not be    copied to the fif format output.

The command-line options of mne_convert_dig_data are:

``--fif <*name*>``

    Specifies the name of an input fif file.

``--hpts <*name*>``

    Specifies the name of an input hpts file.

``--elp <*name*>``

    Specifies the name of an input elp file.

``--fifout <*name*>``

    Specifies the name of an output fif file.

``--hptsout <*name*>``

    Specifies the name of an output hpts file.

``--headcoord``

    The fif and hpts input
    files are assumed to contain data in the  MNE head coordinate system,
    see :ref:`BJEBIBAI`. With this option present, the data are
    transformed to the MNE head coordinate system with help of the fiducial
    locations in the data. Use this option if this is not the case or
    if you are unsure about the definition of the coordinate system
    of the fif and hpts input
    data. This option is implied with elp input
    files. If this option is present, the fif format output file will contain
    the transformation between the original digitizer data coordinates
    the MNE head coordinate system.

.. _CJADJEBH:

The hpts format
---------------

The hpts format digitzer
data file may contain comment lines starting with the pound sign
(#) and data lines of the form:

 <*category*> <*identifier*> <*x/mm*> <*y/mm*> <*z/mm*>

where

`` <*category*>``

    defines the type of points. Allowed categories are: hpi , cardinal (fiducial ),eeg ,
    and extra corresponding to head-position
    indicator coil locations, cardinal landmarks, EEG electrode locations,
    and additional head surface points, respectively. Note that tkmedit does not
    recognize the fiducial as an
    alias for cardinal .

`` <*identifier*>``

    identifies the point. The identifiers are usually sequential numbers. For
    cardinal landmarks, 1 = left auricular point, 2 = nasion, and 3
    = right auricular point. For EEG electrodes, identifier = 0 signifies
    the reference electrode. Some programs (not tkmedit )
    accept electrode labels as identifiers in the eeg category.

`` <*x/mm*> , <*y/mm*> , <*z/mm*>``

    Location of the point, usually in the MEG head coordinate system, see :ref:`BJEBIBAI`.
    Some programs have options to accept coordinates in meters instead
    of millimeters. With ``--meters`` option, mne_transform_points lists
    the coordinates in meters.


.. _mne_convert_lspcov:

mne_convert_lspcov
==================

The utility mne_convert_lspcov converts a LISP-format noise-covariance file,
produced by the Neuromag signal processor, graph into fif format.

The command-line options are:

``--lspcov <*name*>``

    The LISP noise-covariance matrix file to be converted.

``--meas <*name*>``

    A fif format measurement file used to assign channel names to the noise-covariance
    matrix elements. This file should have precisely the same channel
    order within MEG and EEG as the LISP-format covariance matrix file.

``--out <*name*>``

    The name of a fif format output file. The file name should end with
    -cov.fif.text format output file. No information about the channel names
    is included. The covariance matrix file is listed row by row. This
    file can be loaded to MATLAB, for example

``--outasc <*name*>``

    The name of a text format output file. No information about the channel
    names is included. The covariance matrix file is listed row by row.
    This file can be loaded to MATLAB, for example


.. _mne_convert_ncov:

mne_convert_ncov
================

The ncov file format was used to store the noise-covariance
matrix file. The MNE software requires that the covariance matrix
files are in fif format. The utility mne_convert_ncov converts
ncov files to fif format.

The command-line options are:

``--ncov <*name*>``

    The ncov file to be converted.

``--meas <*name*>``

    A fif format measurement file used to assign channel names to the noise-covariance
    matrix elements. This file should have precisely the same channel
    order within MEG and EEG as the ncov file. Typically, both the ncov
    file and the measurement file are created by the now mature off-line
    averager, meg_average.


.. _mne_convert_surface:

mne_convert_surface
===================

The utility mne_convert_surface converts
surface data files between different formats.

.. note:: The MNE Matlab toolbox functions enable    reading of FreeSurfer surface files directly. Therefore, the ``--mat``   option has been removed. The dfs file format conversion functionality    has been moved here from mne_convert_dfs .    Consequently, mne_convert_dfs has    been removed from MNE software.

.. _BABEABAA:

command-line options
--------------------

mne_convert_surface accepts
the following command-line options:

``--fif <*name*>``

    Specifies a fif format input file. The first surface (source space)
    from this file will be read.

``--tri <*name*>``

    Specifies a text format input file. The format of this file is described in :ref:`BEHDEFCD`.

``--meters``

    The unit of measure for the vertex locations in a text input files
    is meters instead of the default millimeters. This option does not
    have any effect on the interpretation of the FreeSurfer surface
    files specified with the ``--surf`` option.

``--swap``

    Swap the ordering or the triangle vertices. The standard convention in
    the MNE software is to have the vertices in text format files ordered
    so that the vector cross product of the vectors from vertex 1 to
    2 and 1 to 3 gives the direction of the outward surface normal. This
    is also called the counterclockwise ordering. If your text input file
    does not comply with this right-hand rule, use the ``--swap`` option.
    This option does not have any effect on the interpretation of the FreeSurfer surface
    files specified with the ``--surf`` option.

``--surf <*name*>``

    Specifies a FreeSurfer format
    input file.

``--dfs <*name*>``

    Specifies the name of a dfs file to be converted. The surfaces produced
    by BrainSuite are in the dfs format.

``--mghmri <*name*>``

    Specifies a mgh/mgz format MRI data file which will be used to define
    the coordinate transformation to be applied to the data read from
    a dfs file to bring it to the FreeSurfer MRI
    coordinates, *i.e.*, the coordinate system of
    the MRI stack in the file. In addition, this option can be used
    to insert "volume geometry" information to the FreeSurfer
    surface file output (``--surfout`` option). If the input file already
    contains the volume geometry information, --replacegeom is needed
    to override the input volume geometry and to proceed to writing
    the data.

``--replacegeom``

    Replaces existing volume geometry information. Used in conjunction
    with the ``--mghmri`` option described above.

``--fifmri <*name*>``

    Specifies a fif format MRI destription file which will be used to define
    the coordinate transformation to be applied to the data read from
    a dfs file to bring it to the same coordinate system as the MRI stack
    in the file.

``--trans <*name*>``

    Specifies the name of a text file which contains the coordinate
    transformation to be applied to the data read from the dfs file
    to bring it to the MRI coordinates, see below. This option is rarely
    needed.

``--flip``

    By default, the dfs surface nodes are assumed to be in a right-anterior-superior
    (RAS) coordinate system with its origin at the left-posterior-inferior
    (LPI) corner of the MRI stack. Sometimes the dfs file has left and
    right flipped. This option reverses this flip, *i.e.*,
    assumes the surface coordinate system is left-anterior-superior
    (LAS) with its origin in the right-posterior-inferior (RPI) corner
    of the MRI stack.

``--shift <*value/mm*>``

    Shift the surface vertices to the direction of the surface normals
    by this amount before saving the surface.

``--surfout <*name*>``

    Specifies a FreeSurfer format output file.

``--fifout <*name*>``

    Specifies a fif format output file.

``--triout <*name*>``

    Specifies an ASCII output file that will contain the surface data
    in the triangle file format desribed in :ref:`BEHDEFCD`.

``--pntout <*name*>``

    Specifies a ASCII output file which will contain the vertex numbers only.

``--metersout``

    With this option the ASCII output will list the vertex coordinates
    in meters instead of millimeters.

``--swapout``

    Defines the vertex ordering of ASCII triangle files to be output.
    For details, see ``--swap`` option, above.

``--smfout <*name*>``

    Specifies a smf (Simple Model Format) output file. For details of this
    format, see http://people.sc.fsu.edu/~jburkardt/data/smf/smf.txt.

.. note:: Multiple output options can be specified to    produce outputs in several different formats with a single invocation    of mne_convert_surface .

The coordinate transformation file specified with the ``--trans`` should contain
a 4 x 4 coordinate transformation matrix:

.. math::    T = \begin{bmatrix}
		R_{11} & R_{12} & R_{13} & x_0 \\
		R_{13} & R_{13} & R_{13} & y_0 \\
		R_{13} & R_{13} & R_{13} & z_0 \\
		0 & 0 & 0 & 1
		\end{bmatrix}

defined so that if the augmented location vectors in the
dfs file and MRI coordinate systems are denoted by :math:`r_{dfs} = [x_{dfs} y_{dfs} z_{dfs} 1]^T` and :math:`r_{MRI} = [x_{MRI} y_{MRI} z_{MRI} 1]^T`,
respectively,

.. math::    r_{MRI} = Tr_{dfs}


.. _mne_cov2proj:

mne_cov2proj
============

Purpose
-------

The utility mne_cov2proj picks
eigenvectors from a covariance matrix and outputs them as a signal-space
projection (SSP) file.

Command line options
--------------------

mne_cov2proj accepts the
following command-line options:

``--cov  <*name*>``

    The covariance matrix file to be used a source. The covariance matrix
    files usually end with ``-cov.fif`` .

``--proj  <*name*>``

    The output file to contain the projection. It is recommended that
    the file name ends with ``-proj.fif`` .

``--bad  <*name*>``

    Specify channels not to be included when an eigenvalue decomposition
    of the covariance matrix is computed.

``--include  <*val1*> [: <*val2*> ]``

    Select an eigenvector or a range of eigenvectors to include. It
    is recommended that magnetometers, gradiometers, and EEG data are handled
    separately with help of the ``--bad`` , ``--meg`` , ``--megmag`` , ``--meggrad`` ,
    and ``--eeg`` options.

``--meg``

    After loading the covariance matrix, modify it so that only elements corresponding
    to MEG channels are included.

``--eeg``

    After loading the covariance matrix, modify it so that only elements corresponding
    to EEG channels are included.

``--megmag``

    After loading the covariance matrix, modify it so that only elements corresponding
    to MEG magnetometer channels are included.

``--meggrad``

    After loading the covariance matrix, modify it so that only elements corresponding
    to MEG planar gradiometer channels are included.

.. note:: The ``--megmag`` and ``--meggrad`` employ    the Vectorview channel numbering scheme to recognize MEG magnetometers    (channel names ending with '1') and planar gradiometers    (other channels). Therefore, these options are only meaningful in    conjunction with data acquired with a Neuromag Vectorview system.


.. _mne_create_comp_data:

mne_create_comp_data
====================

``--in <*name*>``

    Specifies the input text file containing the compensation data.

``--kind <*value*>``

    The compensation type to be stored in the output file with the data. This
    value defaults to 101 for the Magnes compensation and does not need
    to be changed.

``--out <*name*>``

    Specifies the output fif file containing the compensation channel weight
    matrix :math:`C_{(k)}`, see :ref:`BEHDDFBI`.

The format of the text-format compensation data file is:

 <*number of MEG helmet channels*> <*number of compensation channels included*>
 <*cname_1*> <*cname_2*> ...
 <*name_1*> <*weights*>
 <*name_2*> <*weights*> ...

In the above <*name_k*> denote
names of MEG helmet channels and <*cname_k*>
those of the compensation channels, respectively. If the channel
names contain spaces, they must be surrounded by quotes, for example, ``"MEG 0111"`` .



.. _mne_ctf2fiff:

mne_ctf2fiff
============

``--verbose``

    Produce a verbose listing of the conversion process to stdout.

``--ds <*directory*>``

    Read the data from this directory

``--omit <*filename*>``

    Read the names of channels to be omitted from this text file. Enter one
    channel name per line. The names should match exactly with those
    listed in the CTF data structures. By default, all channels are included.

``--fif <*filename*>``

    The name of the output file. If the length of the raw data exceeds
    the 2-GByte fif file limit, several output files will be produced.
    These additional 'extension' files will be tagged
    with ``_001.fif`` , ``_002.fif`` , etc.

``--evoked``

    Produce and evoked-response fif file instead of a raw data file.
    Each trial in the CTF data file is included as a separate category
    (condition). The maximum number of samples in each trial is limited
    to 25000.

``--infoonly``

    Write only the measurement info to the output file, do not include data.

During conversion, the following files are consulted from
the ds directory:

`` <*name*> .res4``

    This file contains most of the header information pertaining the acquisition.

`` <*name*> .hc``

    This file contains the HPI coil locations in sensor and head coordinates.

`` <*name*> .meg4``

    This file contains the actual MEG data. If the data are split across several
    files due to the 2-GByte file size restriction, the 'extension' files
    are called <*name*> ``.`` <*number*> ``_meg4`` .

`` <*name*> .eeg``

    This is an optional input file containing the EEG electrode locations. More
    details are given below.

If the <*name*> ``.eeg`` file,
produced from the Polhemus data file with CTF software, is present,
it is assumed to contain lines with the format:

 <*number*> <*name*> <*x/cm*> <*y/cm*> <*z/cm*>

The field <*number*> is
a sequential number to be assigned to the converted data point in
the fif file. <*name*> is either
a name of an EEG channel, one of ``left`` , ``right`` ,
or ``nasion`` to indicate a fiducial landmark, or any word
which is not a name of any channel in the data. If <*name*> is
a name of an EEG channel available in the data, the location is
included in the Polhemus data as an EEG electrode locations and
inserted as the location of the EEG electrode. If the name is one
of the fiducial landmark names, the point is included in the Polhemus
data as a fiducial landmark. Otherwise, the point is included as
an additional head surface points.

The standard ``eeg`` file produced by CTF software
does not contain the fiducial locations. If desired, they can be
manually copied from the ``pos`` file which was the source
of the ``eeg`` file.

.. note:: In newer CTF data the EEG position information maybe present in the ``res4`` file. If the ``eeg`` file    is present, the positions given there take precedence over the information in the ``res4`` file.

.. note:: mne_ctf2fiff converts both epoch mode and continuous raw data file into raw data fif files. It is not advisable to use epoch mode files with time gaps between the epochs because the data will be discontinuous in the resulting fif file with jumps at the junctions between epochs. These discontinuities    produce artefacts if the raw data is filtered in mne_browse_raw , mne_process_raw ,    or graph .

.. note:: The conversion process includes a transformation from the CTF head coordinate system convention to that used in the Neuromag systems.


.. _mne_ctf_dig2fiff:

mne_ctf_dig2fiff
================

The input data to mne_ctf_dig2fiff is
a text file, which contains the coordinates of the digitization
points in centimeters. The first line should contain a single number
which is the number of points listed in the file. Each of the following
lines contains a sequential number of the point, followed by the
three coordinates. mne_ctf_dig2fiff ignores
any text following the :math:`z` coordinate
on each line. If the ``--numfids`` option is specified,
the first three points indicate the three fiducial locations (1
= nasion, 2 = left auricular point, 3 = right auricular point).
Otherwise, the input file must end with three lines beginning with ``left`` , ``right`` ,
or ``nasion`` to indicate the locations of the fiducial
landmarks, respectively.

.. note:: The sequential numbers should be unique within a file. I particular, the numbers 1, 2, and 3 must not be appear more than once if the ``--numfids`` options is used.

The command-line options for mne_ctf_dig2fiff are:

``--dig <*name*>``

    Specifies the input data file in CTF output format.

``--numfids``

    Fiducial locations are numbered instead of labeled, see above.

``--hpts <*name*>``

    Specifies the output hpts file. The format of this text file is
    described in :ref:`CJADJEBH`.

``--fif <*name*>``

    Specifies the output fif file.


.. _mne_dicom_essentials:

mne_dicom_essentials
====================

Print essential information about a dicom file.

``--in <name>``

    The input file.


.. _mne_edf2fiff:

mne_edf2fiff
============

The mne_edf2fiff allows
conversion of EEG data from EDF, EDF+, and BDF formats to the fif
format. Documentation for these three input formats can be found
at:

``EDF:``

    http://www.edfplus.info/specs/edf.html

``EDF+:``

    http://www.edfplus.info/specs/edfplus.html

``BDF:``

    http://www.biosemi.com/faq/file_format.htm

EDF (European Data Format) and EDF+ are 16-bit formats while
BDF is a 24-bit variant of this format used by the EEG systems manufactured
by a company called BioSemi.

None of these formats support electrode location information
and  head shape digitization information. Therefore, this information
has to be provided separately. Presently hpts and elp file formats
are supported to include digitization data. For information on these
formats, see :ref:`CJADJEBH` and http://www.sourcesignal.com/formats_probe.html.
Note that it is mandatory to have the three fiducial locations (nasion
and the two auricular points) included in the digitization data.
Using the locations of the fiducial points the digitization data
are converted to the MEG head coordinate system employed in the
MNE software, see :ref:`BJEBIBAI`. In the comparison of the
channel names only the initial segment up to the first '-' (dash)
in the EDF/EDF+/BDF channel name is significant.

The EDF+ files may contain an annotation channel which can
be used to store trigger information. The Time-stamped Annotation
Lists (TALs) on the annotation  data can be converted to a trigger
channel (STI 014) using an annotation map file which associates
an annotation label with a number on the trigger channel. The TALs
can be listed with the ``--listtal`` option,
see below.

.. warning:: The data samples in a BDF file    are represented in a 3-byte (24-bit) format. Since 3-byte raw data    buffers are not presently supported in the fif format    these data will be changed to 4-byte integers in the conversion.    Since the maximum size of a fif file is 2 GBytes, the maximum size of    a BDF file to be converted is approximately 1.5 GBytes

.. warning:: The EDF/EDF+/BDF formats support channel    dependent sampling rates. This feature is not supported by mne_edf2fiff .    However, the annotation channel in the EDF+ format can have a different    sampling rate. The annotation channel data is not included in the    fif files output.

Using mne_edf2fiff
------------------

The command-line options of mne_edf2fiff are:

``--edf <*filename*>``

    Specifies the name of the raw data file to process.

``--tal <*filename*>``

    List the time-stamped annotation list (TAL) data from an EDF+ file here.
    This output is useful to assist in creating the annotation map file,
    see the ``--annotmap`` option, below.
    This output file is an event file compatible with mne_browse_raw and mne_process_raw ,
    see :ref:`ch_browse`. In addition, in the mapping between TAL
    labels and trigger numbers provided by the ``--annotmap`` option is
    employed to assign trigger numbers in the event file produced. In
    the absence of the ``--annotmap`` option default trigger number 1024
    is used.

``--annotmap <*filename*>``

    Specify a file which maps the labels of the TALs to numbers on a trigger
    channel (STI 014) which will be added to the output file if this
    option is present. This annotation map file
    may contain comment lines starting with the '%' or '#' characters.
    The data lines contain a label-number pair, separated by a colon.
    For example, a line 'Trigger-1:9' means that each
    annotation labeled with the text 'Trigger-1' will
    be translated to the number 9 on the trigger channel.

``--elp <*filename*>``

    Specifies the name of the an electrode location file. This file
    is in the "probe" file format used by the *Source
    Signal Imaging, Inc.* software. For description of the
    format, see http://www.sourcesignal.com/formats_probe.html. Note
    that some other software packages may produce electrode-position
    files with the elp ending not
    conforming to the above specification. As discussed above, the fiducial
    marker locations, optional in the "probe" file
    format specification are mandatory for mne_edf2fiff .
    When this option is encountered on the command line any previously
    specified hpts file will be ignored.

``--hpts <*filename*>``

    Specifies the name of an electrode position file in  the hpts format discussed
    in :ref:`CJADJEBH`. The mandatory entries are the fiducial marker
    locations and the EEG electrode locations. It is recommended that
    electrode (channel) names instead of numbers are used to label the
    EEG electrode locations. When this option is encountered on the
    command line any previously specified elp file
    will be ignored.

``--meters``

    Assumes that the digitization data in an hpts file
    is given in meters instead of millimeters.

``--fif <*filename*>``

    Specifies the name of the fif file to be output.

Post-conversion tasks
---------------------

This section outlines additional steps to be taken to use
the EDF/EDF+/BDF file is converted to the fif format in MNE:

- Some of the channels may not have a
  digitized electrode location associated with them. If these channels
  are used for EOG or EMG measurements, their channel types should
  be changed to the correct ones using the :ref:`mne_rename_channels` utility,
  EEG channels which do not have a location
  associated with them should be assigned to be MISC channels.

- After the channel types are correctly defined, a topographical
  layout file can be created for mne_browse_raw and mne_analyze using
  the :ref:`mne_make_eeg_layout` utility.

- The trigger channel name in BDF files is "Status".
  This must be specified with the ``--digtrig`` option or with help of
  the MNE_TRIGGER_CH_NAME environment variable when :ref:`mne_browse_raw` or
  :ref:`mne_process_raw` is invoked.

- Only the two least significant bytes on the "Status" channel
  of BDF files are significant as trigger information the ``--digtrigmask``
  0xff option MNE_TRIGGER_CH_MASK environment variable should be used
  to specify this to :ref:`mne_browse_raw` and :ref:`mne_process_raw`,


.. _mne_epochs2mat:

mne_epochs2mat
==============

The utility mne_epochs2mat converts
epoch data including all or selected channels from a raw data file
to a simple binary file with an associated description file in Matlab
mat file format. With help of the description file, a matlab program
can easily read the epoch data from the simple binary file. Signal
space projection and bandpass filtering can be optionally applied
to the raw data prior to saving the epochs.

.. note:: The MNE Matlab toolbox described in :ref:`ch_matlab` provides direct    access to raw fif files without conversion with mne_epochs2mat first.    Therefore, it is recommended that you use the Matlab toolbox rather than mne_epochs2mat which    creates large files occupying disk space unnecessarily. An exception    to this is the case where you apply a filter to the data and save    the band-pass filtered epochs.

Command-line options
--------------------

mne_epochs2mat accepts
the following command-line options are:

``--raw <*name*>``

    Specifies the name of the raw data fif file to use as input.

``--mat <*name*>``

    Specifies the name of the destination file. Anything following the last
    period in the file name will be removed before composing the output
    file name. The binary epoch file will be called <*trimmed name*> ``.epochs`` and
    the corresponding Matlab description file will be <*trimmed name*> ``_desc.mat`` .

``--tag <*tag*>``

    By default, all Matlab variables included in the description file
    start with ``mne\_``. This option changes the prefix to <*tag*> _.

``--events <*name*>``

    The file containing the event definitions. This can be a text or
    fif format file produced by :ref:`mne_process_raw` or
    :ref:`mne_browse_raw`. With help of this file it is possible
    to select virtually any data segment from the raw data file. If
    this option is missing, the digital trigger channel in the raw data
    file or a fif format event file produced automatically by mne_process_raw or mne_browse_raw is
    consulted for event information.

``--event <*name*>``

    Event number identifying the epochs of interest.

``--tmin <*time/ms*>``

    The starting point of the epoch with respect to the event of interest.

``--tmax <*time/ms*>``

    The endpoint of the epoch with respect to the event of interest.

``--sel <*name*>``

    Specifies a text file which contains the names of the channels to include
    in the output file, one channel name per line. If the ``--inv`` option
    is specified, ``--sel`` is ignored. If neither ``--inv`` nor ``--sel`` is
    present, all MEG and EEG channels are included. The digital trigger
    channel can be included with the ``--includetrig`` option, described
    below.

``--inv <*name*>``

    Specifies an inverse operator, which will be employed in two ways. First,
    the channels included to output will be those included in the inverse
    operator. Second, any signal-space projection operator present in
    the inverse operator file will be applied to the data. This option
    cancels the effect of ``--sel`` and ``--proj`` options.

``--digtrig <*name*>``

    Name of the composite digital trigger channel. The default value
    is 'STI 014'. Underscores in the channel name
    will be replaced by spaces.

``--digtrigmask <*number*>``

    Mask to be applied to the trigger channel values before considering them.
    This option is useful if one wants to set some bits in a don't care
    state. For example, some finger response pads keep the trigger lines
    high if not in use, *i.e.*, a finger is not in
    place. Yet, it is convenient to keep these devices permanently connected
    to the acquisition system. The number can be given in decimal or
    hexadecimal format (beginning with 0x or 0X). For example, the value
    255 (0xFF) means that only the lowest order byte (usually trigger
    lines 1 - 8 or bits 0 - 7) will be considered.

``--includetrig``

    Add the digital trigger channel to the list of channels to output.
    This option should not be used if the trigger channel is already
    included in the selection specified with the ``--sel`` option.

``--filtersize <*size*>``

    Adjust the length of the FFT to be applied in filtering. The number will
    be rounded up to the next power of two. If the size is :math:`N`,
    the corresponding length of time is :math:`^N/_{f_s}`,
    where :math:`f_s` is the sampling frequency
    of your data. The filtering procedure includes overlapping tapers
    of length :math:`^N/_2` so that the total FFT
    length will actually be :math:`2N`. The default
    value is 4096.

``--highpass <*value/Hz*>``

    Highpass filter frequency limit. If this is too low with respect
    to the selected FFT length and data file sampling frequency, the
    data will not be highpass filtered. You can experiment with the
    interactive version to find the lowest applicable filter for your
    data. This value can be adjusted in the interactive version of the
    program. The default is 0, i.e., no highpass filter in effect.

``--highpassw <*value/Hz*>``

    The width of the transition band of the highpass filter. The default
    is 6 frequency bins, where one bin is :math:`^{f_s}/_{(2N)}`.

``--lowpass <*value/Hz*>``

    Lowpass filter frequency limit. This value can be adjusted in the interactive
    version of the program. The default is 40 Hz.

``--lowpassw <*value/Hz*>``

    The width of the transition band of the lowpass filter. This value
    can be adjusted in the interactive version of the program. The default
    is 5 Hz.

``--filteroff``

    Do not filter the data.

``--proj <*name*>``

    Include signal-space projection (SSP) information from this file.
    If the ``--inv`` option is present, ``--proj`` has
    no effect.

.. note:: Baseline has not been subtracted from the epochs. This has to be done in subsequent processing with Matlab if so desired.

.. note:: Strictly speaking, trigger mask value zero would mean that all trigger inputs are ignored. However, for convenience,    setting the mask to zero or not setting it at all has the same effect    as 0xFFFFFFFF, *i.e.*, all bits set.

.. note:: The digital trigger channel can also be set with the MNE_TRIGGER_CH_NAME environment variable. Underscores in the variable    value will *not* be replaced with spaces by mne_browse_raw or mne_process_raw .    Using the ``--digtrig`` option supersedes the MNE_TRIGGER_CH_NAME    environment variable.

.. note:: The digital trigger channel mask can also be    set with the MNE_TRIGGER_CH_MASK environment variable. Using the ``--digtrigmask`` option    supersedes the MNE_TRIGGER_CH_MASK environment variable.

The binary epoch data file
--------------------------

mne_epochs2mat saves the
epoch data extracted from the raw data file is a simple binary file.
The data are stored as big-endian single-precision floating point
numbers. Assuming that each of the total of :math:`p` epochs
contains :math:`n` channels and :math:`m` time
points, the data :math:`s_{jkl}` are ordered
as

.. math::    s_{111} \dotso s_{1n1} s_{211} \dotso s_{mn1} \dotso s_{mnp}\ ,

where the first index stands for the time point, the second
for the channel, and the third for the epoch number, respectively.
The data are not calibrated, i.e., the calibration factors present
in the Matlab description file have to be applied to get to physical
units as described below.

.. note:: The maximum size of an epoch data file is 2 Gbytes, *i.e.*, 0.5 Gsamples.

Matlab data structures
----------------------

The Matlab description files output by mne_epochs2mat contain
a data structure <*tag*>_epoch_info .
The fields of the this structure are listed in :ref:`BEHFDCIH`.
Further explanation of the epochs member
is provided in :ref:`BEHHAGHE`.


.. tabularcolumns:: |p{0.15\linewidth}|p{0.15\linewidth}|p{0.6\linewidth}|
.. _BEHIFJIJ:
.. table:: The fields of the raw data info structure.

    +-----------------------+-----------------+------------------------------------------------------------+
    | Variable              | Size            | Description                                                |
    +-----------------------+-----------------+------------------------------------------------------------+
    | orig_file             | string          | The name of the original fif file specified with the       |
    |                       |                 | ``--raw`` option.                                          |
    +-----------------------+-----------------+------------------------------------------------------------+
    | epoch_file            | string          | The name of the epoch data file produced by mne_epocs2mat. |
    +-----------------------+-----------------+------------------------------------------------------------+
    | nchan                 | 1               | Number of channels.                                        |
    +-----------------------+-----------------+------------------------------------------------------------+
    | nepoch                | 1               | Total number of epochs.                                    |
    +-----------------------+-----------------+------------------------------------------------------------+
    | epochs                | nepoch x 5      | Description of the content of the epoch data file,         |
    |                       |                 | see :ref:`BEHHAGHE`.                                       |
    +-----------------------+-----------------+------------------------------------------------------------+
    | sfreq                 | 1               | The sampling frequency in Hz.                              |
    +-----------------------+-----------------+------------------------------------------------------------+
    | lowpass               | 1               | Lowpass filter frequency (Hz)                              |
    +-----------------------+-----------------+------------------------------------------------------------+
    | highpass              | 1               | Highpass filter frequency (Hz)                             |
    +-----------------------+-----------------+------------------------------------------------------------+
    | ch_names              | nchan (string)  | String array containing the names of the channels included |
    +-----------------------+-----------------+------------------------------------------------------------+
    | ch_types              | nchan x 2       | The column lists the types of the channels (1 = MEG, 2 =   |
    |                       |                 | EEG). The second column lists the coil types, see          |
    |                       |                 | :ref:`BGBBHGEC` and :ref:`CHDBDFJE`. For EEG electrodes,   |
    |                       |                 | this value equals one.                                     |
    +-----------------------+-----------------+------------------------------------------------------------+
    | ch_lognos             | nchan x 1       | Logical channel numbers as listed in the fiff file         |
    +-----------------------+-----------------+------------------------------------------------------------+
    | ch_units              | nchan x 2       | Units and unit multipliers as listed in the fif file.      |
    |                       |                 | The unit of the data is listed in the first column         |
    |                       |                 | (T = 112, T/m = 201, V = 107). At present, the second      |
    |                       |                 | column will be always zero, *i.e.*, no unit multiplier.    |
    +-----------------------+-----------------+------------------------------------------------------------+
    | ch_pos                | nchan x 12      | The location information for each channel. The first three |
    |                       |                 | values specify the origin of the sensor coordinate system  |
    |                       |                 | or the location of the electrode. For MEG channels, the    |
    |                       |                 | following nine number specify the *x*, *y*, and            |
    |                       |                 | *z*-direction unit vectors of the sensor coordinate        |
    |                       |                 | system. For EEG electrodes the first vector after the      |
    |                       |                 | electrode location specifies the location of the reference |
    |                       |                 | electrode. If the reference is not specified this value is |
    |                       |                 | all zeroes. The remaining unit vectors are irrelevant for  |
    |                       |                 | EEG electrodes.                                            |
    +-----------------------+-----------------+------------------------------------------------------------+
    | ch_cals               | nchan x 2       | The raw data output by mne_raw2mat are not calibrated.     |
    |                       |                 | The first column is the range member of the fiff data      |
    |                       |                 | structures and while the second is the cal member. To      |
    |                       |                 | get calibrated data values in the units given in           |
    |                       |                 | ch_units from the raw data, the data must be multiplied    |
    |                       |                 | with the product of range and cal .                        |
    +-----------------------+-----------------+------------------------------------------------------------+
    | meg_head_trans        | 4 x 4           | The coordinate frame transformation from the MEG device    |
    |                       |                 | coordinates to the MEG head coordinates.                   |
    +-----------------------+-----------------+------------------------------------------------------------+


.. tabularcolumns:: |p{0.2\linewidth}|p{0.6\linewidth}|
.. _BEHHAGHE:
.. table:: The epochs member of the raw data info structure.

    +---------------+------------------------------------------------------------------+
    | Column        | Contents                                                         |
    +---------------+------------------------------------------------------------------+
    | 1             | The raw data type (2 or 16 = 2-byte signed integer, 3 = 4-byte   |
    |               | signed integer, 4 = single-precision float). The epoch data are  |
    |               | written using the big-endian byte order. The data are stored     |
    |               | sample by sample.                                                |
    +---------------+------------------------------------------------------------------+
    | 2             | Byte location of this epoch in the binary epoch file.            |
    +---------------+------------------------------------------------------------------+
    | 3             | First sample of this epoch in the original raw data file.        |
    +---------------+------------------------------------------------------------------+
    | 4             | First sample of the epoch with respect to the event.             |
    +---------------+------------------------------------------------------------------+
    | 5             | Number of samples in the epoch.                                  |
    +---------------+------------------------------------------------------------------+

.. note:: For source modelling purposes, it is recommended    that the MNE Matlab toolbox, see :ref:`ch_matlab` is employed    to read the measurement info instead of using the channel information    in the raw data info structure described in :ref:`BEHIFJIJ`.


.. _mne_evoked_data_summary:

mne_evoked_data_summary
=======================

Print a summary of evoked-response data sets in a file (averages only).

``--in <name>``

    The input file.


.. _mne_eximia2fiff:

mne_eximia2fiff
===============

Usage:

``mne_eximia2fiff`` [``--dig`` dfile ] [``--orignames`` ] file1 file2 ...

where file1 file2 ...
are eXimia ``nxe`` files and the ``--orignames`` option
is passed on to :ref:`mne_brain_vision2fiff`.
If you want to convert all data files in a directory, say

``mne_eximia2fiff *.nxe``

The optional file specified with the ``--dig`` option is assumed
to contain digitizer data from the recording in the Nexstim format.
The resulting fif data file will contain these data converted to
the fif format as well as the coordinate transformation between
the eXimia digitizer and MNE head coordinate systems.

.. note:: This script converts raw data files only.


.. _mne_fit_sphere_to_surf:

mne_fit_sphere_to_surf
======================

Purpose
-------

The utility mne_fit_sphere_to_surf finds
the sphere which best fits a given surface.

Command line options
--------------------

mne_fit_sphere_to_surf accepts
the following command-line options:

``--bem  <*name*>``

    A BEM file to use. The names of these files usually end with ``bem.fif`` or ``bem-sol.fif`` .

``--surf  <*name*>``

    A FreeSurfer surface file to read. This is an alternative to using
    a surface from the BEM file.

``--scalp``

    Use the scalp surface instead of the inner skull surface in sphere
    fitting. If the surface is specified with the ``--surf`` option,
    this one is irrelevant.

``--mritrans  <*name*>``

    A file containing a transformation matrix between the MEG head coordinates
    and MRI coordinates. With this option, the sphere origin will be
    output in MEG head coordinates. Otherwise the output will be in MRI
    coordinates.


.. _mne_fix_mag_coil_types:

mne_fix_mag_coil_types
======================

The purpose of mne_fix_mag_coil_types is
to change coil type 3022 to 3024 in the MEG channel definition records
in the data files specified on the command line.

As shown in Tables 5.2 and 5.3, the Neuromag Vectorview systems
can contain magnetometers with two different coil sizes (coil types
3022 and 3023 vs. 3024). The systems incorporating coils of type
3024 were introduced last. At some sites the data files have still
defined the magnetometers to be of type 3022 to ensure compatibility
with older versions of Neuromag software. In the MNE software as
well as in the present version of Neuromag software coil type 3024
is fully supported. Therefore, it is now safe to upgrade the data
files to use the true coil type.

If the ``--magnes`` option is specified, the 4D
Magnes magnetometer coil type (4001) is changed to 4D Magnes gradiometer
coil type (4002). Use this option always and *only
if* your Magnes data comes from a system with axial gradiometers
instead of magnetometers. The fif converter included with the Magnes
system does not assign the gradiometer coil type correctly.

.. note:: The effect of the difference between the coil sizes of magnetometer types 3022 and 3024 on the current estimates computed by the MNE software is very small. Therefore the use of mne_fix_mag_coil_types is not mandatory.


.. _mne_fix_stim14:

mne_fix_stim14
==============

Some earlier versions of the Neuromag acquisition software
had a problem with the encoding of the eighth bit on the digital
stimulus channel STI 014. This problem has been now fixed. Old data
files can be fixed with mne_fix_stim14 ,
which takes raw data file names as arguments. mne_fix_stim14 also
changes the calibration of STI 014 to unity. If the encoding of
STI 014 is already correct, running mne_fix_stim14 will
not have any effect on the raw data.

In newer Neuromag Vectorview systems with 16-bit digital
inputs the upper two bytes of the samples may be incorrectly set
when stimulus input 16 is used and the data are acquired in the
32-bit  mode. This problem can be fixed by running mne_fix_stim14 on
a raw data file with the ``--32`` option:

``mne_fix_stim14 --32``  <*raw data file*>

In this case, the correction will be applied to the stimulus
channels 'STI101' and 'STI201'.


.. _mne_flash_bem:

mne_flash_bem
=============

``--help``

    Prints the usage information.

``--usage``

    Prints the usage information.

``--noconvert``

    Skip conversion of the original MRI data. The original data are
    not needed and the preparatory steps 1.-3. listed below
    are thus not needed.

``--noflash30``

    The 30-degree flip angle data are not used.

``--unwarp  <*type*>``

    Run grad_unwarp with ``--unwarp``  <*type*> option on each of the converted
    data sets.


.. _mne_insert_4D_comp:

mne_insert_4D_comp
==================

Import Magnes WH3600 reference channel data from a text file.

``--in <name>``

    The name of the fif file containing the helmet data.

``--ref <name>``

    The name of the text file containing the reference channel data.

``--out <name>``

    The output fif file.


.. _mne_kit2fiff:

mne_kit2fiff
============

The utility mne_kit2fiff was
created in collaboration with Alec Maranz and Asaf Bachrach to import
their MEG data acquired with the 160-channel KIT MEG system to MNE
software.

To import the data, the following input files are mandatory:

- The Polhemus data file (elp file)
  containing the locations of the fiducials and the head-position
  indicator (HPI) coils. These data are usually given in the CTF/4D
  head coordinate system. However, mne_kit2fiff does
  not rely on this assumption. This file can be exported directly from
  the KIT system.

- A file containing the locations of the HPI coils in the MEG
  device coordinate system. These data are used together with the elp file
  to establish the coordinate transformation between the head and
  device coordinate systems. This file can be produced easily by manually
  editing one of the files exported by the KIT system.

- A sensor data file (sns file)
  containing the locations and orientations of the sensors. This file
  can be exported directly from the KIT system.

.. note:: The output fif file will use the Neuromag head    coordinate system convention, see :ref:`BJEBIBAI`. A coordinate    transformation between the CTF/4D head coordinates and the Neuromag    head coordinates is included. This transformation can be read with    MNE Matlab Toolbox routines, see :ref:`ch_matlab`.

The following input files are optional:

- A head shape data file (hsp file)
  containing locations of additional points from the head surface.
  These points must be given in the same coordinate system as that
  used for the elp file and the
  fiducial locations must be within 1 mm from those in the elp file.

- A raw data file containing the raw data values, sample by
  sample, as text. If this file is not specified, the output fif file
  will only contain the measurement info block.

By default mne_kit2fiff includes
the first 157 channels, assumed to be the MEG channels, in the output
file. The compensation channel data are not converted by default
but can be added, together with other channels, with the ``--type`` .
The channels from 160 onwards are designated as miscellaneous input
channels (MISC 001, MISC 002, etc.). The channel names and types
of these channels can be afterwards changed with the :ref:`mne_rename_channels`
utility. In addition, it is possible to synthesize
the digital trigger channel (STI 014) from available analog
trigger channel data, see the ``--stim`` option, below.
The synthesized trigger channel data value at sample :math:`k` will
be:

.. math::    s(k) = \sum_{p = 1}^n {t_p(k) 2^{p - 1}}\ ,

where :math:`t_p(k)` are the thresholded
from the input channel data d_p(k):

.. math::    t_p(k) = \Bigg\{ \begin{array}{l}
		 0 \text{  if  } d_p(k) \leq t\\
		 1 \text{  if  } d_p(k) > t
	     \end{array}\ .

The threshold value :math:`t` can
be adjusted with the ``--stimthresh`` option, see below.

mne_kit2fiff accepts the following command-line options:

``--elp <*filename*>``

    The name of the file containing the locations of the fiducials and
    the HPI coils. This option is mandatory.

``--hsp <*filename*>``

    The name of the file containing the locations of the fiducials and additional
    points on the head surface. This file is optional.

``--sns <*filename*>``

    The name of file containing the sensor locations and orientations. This
    option is mandatory.

``--hpi <*filename*>``

    The name of a text file containing the locations of the HPI coils
    in the MEG device coordinate frame, given in millimeters. The order of
    the coils in this file does not have to be the same as that in the elp file.
    This option is mandatory.

``--raw <*filename*>``

    Specifies the name of the raw data file. If this file is not specified, the
    output fif file will only contain the measurement info block.

``--sfreq <*value/Hz*>``

    The sampling frequency of the data. If this option is not specified, the
    sampling frequency defaults to 1000 Hz.

``--lowpass <*value/Hz*>``

    The lowpass filter corner frequency used in the data acquisition.
    If not specified, this value defaults to 200 Hz.

``--highpass <*value/Hz*>``

    The highpass filter corner frequency used in the data acquisition.
    If not specified, this value defaults to 0 Hz (DC recording).

``--out <*filename*>``

    Specifies the name of the output fif format data file. If this file
    is not specified, no output is produced but the elp , hpi ,
    and hsp files are processed normally.

``--stim <*chs*>``

    Specifies a colon-separated list of numbers of channels to be used
    to synthesize a digital trigger channel. These numbers refer to
    the scanning order channels as listed in the sns file,
    starting from one. The digital trigger channel will be the last
    channel in the file. If this option is absent, the output file will
    not contain a trigger channel.

``--stimthresh <*value*>``

    The threshold value used when synthesizing the digital trigger channel,
    see above. Defaults to 1.0.

``--add <*chs*>``

    Specifies a colon-separated list of numbers of channels to include between
    the 157 default MEG channels and the digital trigger channel. These
    numbers refer to the scanning order channels as listed in the sns file,
    starting from one.

.. note:: The mne_kit2fiff utility    has not been extensively tested yet.


.. _mne_list_bem:

mne_list_bem
============

The utility mne_list_bem outputs
the BEM meshes in text format. The default output data contains
the *x*, *y*, and *z* coordinates
of the vertices, listed in millimeters, one vertex per line.

The command-line options are:

``--bem <*name*>``

    The BEM file to be listed. The file name normally ends with -bem.fif or -bem-sol.fif .

``--out <*name*>``

    The output file name.

``--id <*number*>``

    Identify the surface to be listed. The surfaces are numbered starting with
    the innermost surface. Thus, for a three-layer model the surface numbers
    are: 4 = scalp, 3 = outer skull, 1 = inner skull
    Default value is 4.

``--gdipoli``

    List the surfaces in the format required by Thom Oostendorp's
    gdipoli program. This is also the default input format for mne_surf2bem .

``--meters``

    List the surface coordinates in meters instead of millimeters.

``--surf``

    Write the output in the binary FreeSurfer format.

``--xfit``

    Write a file compatible with xfit. This is the same effect as using
    the options ``--gdipoli`` and ``--meters`` together.


.. _mne_list_coil_def:

mne_list_coil_def
=================

List available coil definitions.

``--in <def>``

    Validate a coil definition file.

``--out <name>``

    List the coil definitions to this file (default: stdout).

``--type <type>``

    Coil type to list.


.. _mne_list_proj:

mne_list_proj
=============

``--in <name>``

    Input file.

``--ascin <name>``

    Input file.

``--exclude <name>``

    Exclude these channels from the projection (set entries to zero).

``--chs <name>``

    Specify a file which contains a channel selection for the output (useful for graph)

``--asc <name>``

    Text output.

``--fif <name>``

    Fif output.

``--lisp <name>``

    Lisp output.


.. _mne_list_source_space:

mne_list_source_space
=====================

The utility mne_list_source_space outputs
the source space information into text files suitable for loading
into the Neuromag MRIlab software.

``--src <*name*>``

    The source space to be listed. This can be either the output from mne_make_source_space
    (`*src.fif`), output from the forward calculation (`*fwd.fif`), or
    the output from the inverse operator decomposition (`*inv.fif`).

``--mri <*name*>``

    A file containing the transformation between the head and MRI coordinates
    is specified with this option. This file can be either a Neuromag
    MRI description file, the output from the forward calculation (`*fwd.fif`),
    or the output from the inverse operator decomposition (`*inv.fif`).
    If this file is included, the output will be in head coordinates.
    Otherwise the source space will be listed in MRI coordinates.

``--dip <*name*>``

    Specifies the 'stem' for the Neuromag text format
    dipole files to be output. Two files will be produced: <*stem*> -lh.dip
    and <*stem*> -rh.dip. These correspond
    to the left and right hemisphere part of the source space, respectively.
    This source space data can be imported to MRIlab through the File/Import/Dipoles menu
    item.

``--pnt <*name*>``

    Specifies the 'stem' for Neuromag text format
    point files to be output. Two files will be produced: <*stem*> -lh.pnt
    and <*stem*> -rh.pnt. These correspond
    to the left and right hemisphere part of the source space, respectively.
    This source space data can be imported to MRIlab through the File/Import/Strings menu
    item.

``--exclude <*name*>``

    Exclude the source space points defined by the given FreeSurfer 'label' file
    from the output. The name of the file should end with ``-lh.label``
    if it refers to the left hemisphere and with ``-rh.label`` if
    it lists points in the right hemisphere, respectively.

``--include <*name*>``

    Include only the source space points defined by the given FreeSurfer 'label' file
    to the output. The file naming convention is the same as described
    above under the ``--exclude`` option. Are 'include' labels are
    processed before the 'exclude' labels.

``--all``

    Include all nodes in the output files instead of only those active
    in the source space. Note that the output files will be huge if
    this option is active.


.. _mne_list_versions:

mne_list_versions
=================

The utility mne_list_versions lists
version numbers and compilation dates of all software modules that
provide this information. This administration utility is located
in ``$MNE_ROOT/bin/admin`` , The output from mne_list_versions or
output of individual modules with ``--version`` option
is useful when bugs are reported to the developers of MNE software.


.. _mne_make_cor_set:

mne_make_cor_set
================

The utility mne_make_cor_set creates
a fif format MRI description
file optionally including the MRI data using FreeSurfer MRI volume
data as input. The command-line options are:

``--dir <*directory*>``

    Specifies a directory containing the MRI volume in COR format. Any
    previous ``--mgh`` options are cancelled when this option
    is encountered.

``--withdata``

    Include the pixel data to the output file. This option is implied
    with the ``--mgh`` option.

``--mgh <*name*>``

    An MRI volume volume file in mgh or mgz format.
    The ``--withdata`` option is implied with this type of
    input. Furthermore, the :math:`T_3` transformation,
    the Talairach transformation :math:`T_4` from
    the talairach.xfm file referred to in the MRI volume, and the the
    fixed transforms :math:`T_-` and :math:`T_+` will
    added to the output file. For definition of the coordinate transformations,
    see :ref:`CHDEDFIB`.

``--talairach <*name*>``

    Take the Talairach transform from this file instead of the one specified
    in mgh/mgz files.

``--out <*name*>``

    Specifies the output file, which is a fif-format MRI description
    file.


.. _mne_make_derivations:

mne_make_derivations
====================

Purpose
-------

In mne_browse_raw , channel
derivations are defined as linear combinations of real channels
existing in the data files. The utility mne_make_derivations reads
derivation data from a suitably formatted text file and produces
a fif file containing the weights of derived channels as a sparse
matrix. Two input file formats are accepted:

- A file containing arithmetic expressions
  defining the derivations and

- A file containing a matrix which specifies the weights of
  the channels in each derivation.

Both of these formats are described in

Command-line options
--------------------

mne_make_derivations recognizes
the following command-line options:

``--in  <*name*>``

    Specifies a measurement file which contains the EEG electrode locations.
    This file is not modified.

``--inmat  <*name*>``

    Specifies the output file where the layout is stored. Suffix ``.lout`` is recommended
    for layout files. mne_analyze and mne_browse_raw look
    for the custom layout files from the directory ``$HOME/.mne/lout`` .

``--trans``

    Indicates that the file specified with the ``--inmat`` option
    contains a transpose of the derivation matrix.

``--thresh  <*value*>``

    Specifies the threshold between values to be considered zero and non-zero
    in the input file specified with the ``--inmat`` option.
    The default threshold is :math:`10^{-6}`.

``--out  <*name*>``

    Specifies output fif file to contain the derivation data. The recommended
    name of the derivation file has the format  <:math:`name`> ``-deriv.fif`` .

``--list  <*name*>``

    List the contents of a derivation file to standard output. If this
    option is missing and ``--out`` is specified, the content
    of the output file will be listed once it is complete. If neither ``--list`` nor ``--out`` is present,
    and ``--in`` or ``--inmat`` is specified, the
    interpreted contents of the input file is listed.

Derivation file formats
-----------------------

All lines in the input files starting with the pound sign
(#) are considered to be comments. The format of a derivation in
a arithmetic input file is:

.. math::    \langle name \rangle = [\langle w_1 \rangle *] \langle name_1 \rangle + [\langle w_2 \rangle *] \langle name_2 \rangle \dotso

where <:math:`name`> is the
name of the derived channel, :math:`name_k` are
the names of the channels comprising the derivation, and :math:`w_k` are
their weights. Note that spaces are necessary between the items.
Channel names containing spaces must be put in quotes. For example,

``EEG-diff = "EEG 003" - "EEG 002"``

defines a channel ``EEG-diff`` which is a difference
between ``EEG 003`` and ``EEG 002`` . Similarly,

``EEG-der = 3 * "EEG 010" - 2 * "EEG 002"``

defines a channel which is three times ``EEG 010`` minus
two times ``EEG 002`` .

The format of a matrix derivation file is:

.. math::    \langle nrow \rangle \langle ncol \rangle \langle names\ of\ the\ input\ channels \rangle \langle name_1 \rangle \langle weights \rangle \dotso

The combination of the two arithmetic examples, above can
be thus represented as:

``2 3 "EEG 002" "EEG 003" "EEG 010" EEG-diff -1 1  0 EEG-der -2 0  3``

Before a derivation is accepted to use by mne_browse_raw ,
the following criteria have to be met:

- All channels to be combined into a single
  derivation must have identical units of measure.

- All channels in a single derivation have to be of the same
  kind, *e.g.*, MEG channels or EEG channels.

- All channels specified in a derivation have to be present
  in the currently loaded data set.

The validity check is done when a derivation file is loaded
into mne_browse_raw , see :ref:`CACFHAFH`.

.. note:: You might consider renaming the EEG channels    with descriptive labels related to the standard 10-20 system using    the :ref:`mne_rename_channels` utility. This allows you to use standard EEG    channel names in the derivations you define as well as in the channel    selection files used in mne_browse_raw ,    see :ref:`CACCJEJD`.


.. _mne_make_eeg_layout:

mne_make_eeg_layout
===================

Purpose
-------

Both MNE software (mne_analyze and mne_browse_raw)
and Neuromag software (xplotter and xfit)
employ text layout files to create topographical displays of MEG
and EEG data. While the MEG channel layout is fixed, the EEG layout
varies from experiment to experiment, depending on the number of
electrodes used and the electrode cap configuration. The utility mne_make_eeg_layout was
created to produce custom EEG layout files based on the EEG electrode
location information included in the channel description records.

mne_make_eeg_layout uses
azimuthal equidistant projection to map the EEG channel locations
onto a plane. The mapping consists of the following steps:

- A sphere is fitted to the electrode
  locations and the locations are translated by the location of the
  origin of the best-fitting sphere.

- The spherical coordinates (:math:`r_k`, :math:`\theta_k`, and :math:`\phi_k`)
  corresponding to each translated electrode location are computed.

- The projected locations :math:`u_k = R \theta_k \cos{\phi_k}` and :math:`v_k = R \theta_k \sin{\phi_k}` are
  computed. By default, :math:`R = 20/{^{\pi}/_2}`, *i.e.* at
  the equator (:math:`\theta = ^{\pi}/_2`) the multiplier is
  20. This projection radius can be adjusted with the ``--prad`` option.
  Increasing or decreasing :math:`R` makes
  the spacing between the channel viewports larger or smaller, respectively.

- A viewport with width 5 and height 4 is placed centered at
  the projected location. The width and height of the viewport can
  be adjusted with the ``--width`` and ``--height`` options

The command-line options are:

``--lout  <*name*>``

    Specifies the name of the layout file to be output.

``--nofit``

    Do not fit a sphere to the electrode locations but use a standard sphere
    center (:math:`x = y = 0`, and :math:`z = 40` mm) instead.

``--prad  <*value*>``

    Specifies a non-standard projection radius :math:`R`,
    see above.

``--width  <*value*>``

    Specifies the width of the viewports. Default value = 5.

``--height  <*value*>``

    Specifies the height of the viewports. Default value = 4.


.. _mne_make_morph_maps:

mne_make_morph_maps
===================
Prepare the mapping data for subject-to-subject morphing.

``--redo``

    Recompute the morphing maps even if they already exist.

``--from <subject>``

    Compute morphing maps from this subject.

``--to <subject>``

    Compute morphing maps to this subject.

``--all``

    Do all combinations. If this is used without either ``--from`` or ``--to`` options,
    morphing maps for all possible combinations are computed. If ``--from`` or ``--to`` is
    present, only maps between the specified subject and all others
    are computed.

.. note:: Because all morphing map files contain maps in both directions, the choice of ``--from`` and ``--to`` options    only affect the naming of the morphing map files to be produced. mne_make_morph_maps creates directory ``$SUBJECTS_DIR/morph-maps`` if necessary.


.. _mne_make_uniform_stc:

mne_make_uniform_stc
====================

The output will have a time range from -100 to 300 ms.
There will be one cycle of 5-Hz sine wave, with the peaks at 50 and 150 ms

``--src <name>``

    Source space to use.

``--stc <name>``

    Stc file to produce.

``--maxval <value>``

    Maximum value (at 50 ms, default 10).

``--all``

    Include all points to the output files.


.. _mne_mark_bad_channels:

mne_mark_bad_channels
=====================

This utility adds or replaces information about unusable
(bad) channels. The command line options are:

``--bad  <*filename*>``

    Specify a text file containing the names of the bad channels, one channel
    name per line. The names of the channels in this file must match
    those in the data file exactly. If this option is missing, the bad channel
    information is cleared.

``<*data file name*>``

    The remaining arguments are taken as data file names to be modified.


.. _mne_morph_labels:

mne_morph_labels
================
Morph label files from one brain to another.

``--from <*subject*>``

    Name of the subject for which the labels were originally defined.

``--to <*subject*>``

    Name of the subject for which the morphed labels should be created.

``--labeldir <*directory*>``

    A directory containing the labels to morph.

``--prefix <prefix>``

    Adds <*prefix*> in the beginning
    of the output label names. A dash will be inserted between <*prefix*> and
    the rest of the name.

``--smooth <number>``

    Apply smoothing with the indicated number of iteration steps (see :ref:`CHDEBAHH`) to the labels before morphing them. This is
    advisable because otherwise the resulting labels may have little
    holes in them since the morphing map is not a bijection. By default,
    two smoothsteps are taken.

As the labels are morphed, a directory with the name of the
subject specified with the ``--to`` option is created under
the directory specified with ``--labeldir`` to hold the
morphed labels.


.. _mne_organize_dicom:

mne_organize_dicom
==================


.. _mne_prepare_bem_model:

mne_prepare_bem_model
=====================

``--bem <*name*>``

    Specify the name of the file containing the triangulations of the BEM
    surfaces and the conductivities of the compartments. The standard
    ending for this file is ``-bem.fif`` and it is produced
    either with the utility :ref:`mne_surf2bem` (:ref:`BEHCACCJ`) or the
    convenience script :ref:`mne_setup_forward_model`,
    see :ref:`CIHDBFEG`.

``--sol <*name*>``

    Specify the name of the file containing the triangulation and conductivity
    information together with the BEM geometry matrix computed by mne_prepare_bem_model .
    The standard ending for this file is ``-bem-sol.fif`` .

``--method <*approximation method*>``

    Select the BEM approach. If <*approximation method*> is ``constant`` ,
    the BEM basis functions are constant functions on each triangle
    and the collocation points are the midpoints of the triangles. With ``linear`` ,
    the BEM basis functions are linear functions on each triangle and
    the collocation points are the vertices of the triangulation. This
    is the preferred method to use. The accuracy will be the same or
    better than in the constant collocation approach with about half
    the number of unknowns in the BEM equations.


.. _mne_process_stc:

mne_process_stc
===============

``--stc <name>``

    Specify the stc file to process.

``--out <name>``

    Specify a stc  output file name.

``--outasc <name>``

    Specify a text output file name.

``--scaleto <scale>``

    Scale the data so that the maximum is this value.

``--scaleby <scale>``

    Multiply the values by this.


.. _mne_raw2mat:

mne_raw2mat
===========

The utility mne_raw2mat converts
all or selected channels from a raw data file to a Matlab mat file.
In addition, this utility can provide information about the raw
data file so that the raw data can be read directly from the original
fif file using Matlab file I/O routines.

.. note:: The MNE Matlab toolbox described in :ref:`ch_matlab` provides    direct access to raw fif files without a need for conversion to    mat file format first. Therefore, it is recommended that you use    the Matlab toolbox rather than  mne_raw2mat which    creates large files occupying disk space unnecessarily.

Command-line options
--------------------

mne_raw2mat accepts the
following command-line options:

``--raw <*name*>``

    Specifies the name of the raw data fif file to convert.

``--mat <*name*>``

    Specifies the name of the destination Matlab file.

``--info``

    With this option present, only information about the raw data file
    is included. The raw data itself is omitted.

``--sel <*name*>``

    Specifies a text file which contains the names of the channels to include
    in the output file, one channel name per line. If the ``--info`` option
    is specified, ``--sel`` does not have any effect.

``--tag <*tag*>``

    By default, all Matlab variables included in the output file start
    with ``mne\_``. This option changes the prefix to <*tag*> _.

Matlab data structures
----------------------

The Matlab files output by mne_raw2mat can
contain two data structures, <*tag*>_raw and <*tag*>_raw_info .
If ``--info`` option is specifed, the file contains the
latter structure only.

The <*tag*>_raw structure
contains only one field, data which
is a matrix containing the raw data. Each row of this matrix constitutes
the data from one channel in the original file. The data type of
this matrix is the same of the original data (2-byte signed integer,
4-byte signed integer, or single-precision float).

The fields of the <*tag*>_raw_info structure
are listed in :ref:`BEHFDCIH`. Further explanation of the bufs field
is provided in :ref:`BEHJEIHJ`.


.. tabularcolumns:: |p{0.2\linewidth}|p{0.15\linewidth}|p{0.6\linewidth}|
.. _BEHFDCIH:
.. table:: The fields of the raw data info structure.

    +-----------------------+-----------------+------------------------------------------------------------+
    | Variable              | Size            | Description                                                |
    +-----------------------+-----------------+------------------------------------------------------------+
    | orig_file             | string          | The name of the original fif file specified with the       |
    |                       |                 | ``--raw`` option.                                          |
    +-----------------------+-----------------+------------------------------------------------------------+
    | nchan                 | 1               | Number of channels.                                        |
    +-----------------------+-----------------+------------------------------------------------------------+
    | nsamp                 | 1               | Total number of samples                                    |
    +-----------------------+-----------------+------------------------------------------------------------+
    | bufs                  | nbuf x 4        | This field is present if ``--info`` option was specified on|
    |                       |                 | the command line. For details, see :ref:`BEHJEIHJ`.        |
    +-----------------------+-----------------+------------------------------------------------------------+
    | sfreq                 | 1               | The sampling frequency in Hz.                              |
    +-----------------------+-----------------+------------------------------------------------------------+
    | lowpass               | 1               | Lowpass filter frequency (Hz)                              |
    +-----------------------+-----------------+------------------------------------------------------------+
    | highpass              | 1               | Highpass filter frequency (Hz)                             |
    +-----------------------+-----------------+------------------------------------------------------------+
    | ch_names              | nchan (string)  | String array containing the names of the channels included |
    +-----------------------+-----------------+------------------------------------------------------------+
    | ch_types              | nchan x 2       | The column lists the types of the channesl (1 = MEG, 2 =   |
    |                       |                 | EEG). The second column lists the coil types, see          |
    |                       |                 | :ref:`BGBBHGEC` and :ref:`CHDBDFJE`. For EEG electrodes,   |
    |                       |                 | this value equals one.                                     |
    +-----------------------+-----------------+------------------------------------------------------------+
    | ch_lognos             | nchan x 1       | Logical channel numbers as listed in the fiff file         |
    +-----------------------+-----------------+------------------------------------------------------------+
    | ch_units              | nchan x 2       | Units and unit multipliers as listed in the fif file.      |
    |                       |                 | The unit of the data is listed in the first column         |
    |                       |                 | (T = 112, T/m = 201, V = 107). At present, the second      |
    |                       |                 | column will be always zero, *i.e.*, no unit multiplier.    |
    +-----------------------+-----------------+------------------------------------------------------------+
    | ch_pos                | nchan x 12      | The location information for each channel. The first three |
    |                       |                 | values specify the origin of the sensor coordinate system  |
    |                       |                 | or the location of the electrode. For MEG channels, the    |
    |                       |                 | following nine number specify the *x*, *y*, and            |
    |                       |                 | *z*-direction unit vectors of the sensor coordinate system.|
    |                       |                 | For EEG electrodes the first vector after the electrode    |
    |                       |                 | location specifies the location of the reference electrode.|
    |                       |                 | If the reference is not specified this value is all zeroes.|
    |                       |                 | The remaining unit vectors are irrelevant for EEG          |
    |                       |                 | electrodes.                                                |
    +-----------------------+-----------------+------------------------------------------------------------+
    | ch_cals               | nchan x 2       | The raw data output by mne_raw2mat is uncalibrated.        |
    |                       |                 | The first column is the range member of the fiff data      |
    |                       |                 | structures and while the second is the cal member. To get  |
    |                       |                 | calibrared data values in the units given in ch_units from |
    |                       |                 | the raw data, the data must be multiplied with the product |
    |                       |                 | of range and cal .                                         |
    +-----------------------+-----------------+------------------------------------------------------------+
    | meg_head_trans        | 4 x 4           | The coordinate frame transformation from the MEG device    |
    |                       |                 | coordinates to the MEG head coordinates.                   |
    +-----------------------+-----------------+------------------------------------------------------------+


.. tabularcolumns:: |p{0.1\linewidth}|p{0.6\linewidth}|
.. _BEHJEIHJ:
.. table:: The bufs member of the raw data info structure.

    +-----------------------+-------------------------------------------------------------------------+
    | Column                | Contents                                                                |
    +-----------------------+-------------------------------------------------------------------------+
    | 1                     | The raw data type (2 or 16 = 2-byte signed integer, 3 = 4-byte signed   |
    |                       | integer, 4 = single-precision float). All data in the fif file are      |
    |                       | written in the big-endian byte order. The raw data are stored sample by |
    |                       | sample.                                                                 |
    +-----------------------+-------------------------------------------------------------------------+
    | 2                     | Byte location of this buffer in the original fif file.                  |
    +-----------------------+-------------------------------------------------------------------------+
    | 3                     | First sample of this buffer. Since raw data storing can be switched on  |
    |                       | and off during the acquisition, there might be gaps between the end of  |
    |                       | one buffer and the beginning of the next.                               |
    +-----------------------+-------------------------------------------------------------------------+
    | 4                     | Number of samples in the buffer.                                        |
    +-----------------------+-------------------------------------------------------------------------+


.. _mne_rename_channels:

mne_rename_channels
===================

Sometimes it is necessary to change the names types of channels
in MEG/EEG data files. Such situations include:

- Designating an EEG as an EOG channel.
  For example, the EOG channels are not recognized as such in the
  fif files converted from CTF data files.

- Changing the name of the digital trigger channel of interest
  to STI 014 so that mne_browse_raw and mne_process_raw will
  recognize the correct channel without the need to specify the ``--digtrig``
  option or the MNE_TRIGGER_CH_NAME environment variable every time a
  data file is loaded.

The utility mne_rename_channels was
designed to meet the above needs. It recognizes the following command-line
options:

``--fif  <*name*>``

    Specifies the name of the data file to modify.

``--alias  <*name*>``

    Specifies the text file which contains the modifications to be applied,
    see below.

``--revert``

    Reverse the roles of old and new channel names in the alias file.

Each line in the alias file contains the old name and new
name for a channel, separated by a colon. The old name is a name
of one of the channels presently in the file and the new name is
the name to be assigned to it. The old name must match an existing
channel name in the file exactly. The new name may be followed by
another colon and a number which is the channel type to be assigned
to this channel. The channel type options are listed below.

.. table:: Channel types.

    ==============  ======================
    Channel type    Corresponding number
    ==============  ======================
    MEG             1
    MCG             201
    EEG             2
    EOG             202
    EMG             302
    ECG             402
    MISC            502
    STIM            3
    ==============  ======================

.. warning:: Do not attempt to designate MEG channels    to EEG channels or vice versa. This may result in strange errors    during source estimation.

.. note:: You might consider renaming the EEG channels    with descriptive labels related to the standard 10-20 system. This    allows you to use standard EEG channel names when defining derivations,    see :ref:`mne_make_derivations` and :ref:`CACFHAFH`, as well as in the    channel selection files used in mne_browse_raw ,    see :ref:`CACCJEJD`.


.. _mne_sensitivity_map:

mne_sensitivity_map
===================

Purpose
-------

mne_sensitivity_map computes
the size of the columns of the forward operator and outputs the
result in w files.

Command line options
--------------------

mne_sensitivity_map accepts
the following command-line options:

``--fwd  <*name*>``

    Specifies a forward solution file to analyze. By default the MEG
    forward solution is considered.

``--proj  <*name*>``

    Specifies a file containing an SSP operator to be applied. If necessary,
    multiple ``--proj`` options can be specified. For map types 1 - 4 (see
    below), SSP is applied to the forward model data. For map types
    5 and 6, the effects of SSP are evaluated against the unmodified
    forward model.

``--eeg``

    Use the EEG forward solution instead of the MEG one. It does not make
    sense to consider a combination because of the different units of
    measure. For the same reason, gradiometers and magnetometers have
    to be handled separately, see ``--mag`` option below. By
    default MEG gradiometers are included.

``--mag``

    Include MEG magnetometers instead of gradiometers

``--w  <*name*>``

    Specifies the stem of the output w files. To obtain the final output file
    names, ``-lh.w`` and ``-rh.w`` is appended for
    the left and right hemisphere, respectively.

``--smooth  <*number*>``

    Specifies the number of smooth steps to apply to the resulting w files.
    Default: no smoothing.

``--map  <*number*>``

    Select the type of a sensitivity map to compute. At present, valid numbers
    are 1 - 6. For details, see :ref:`CHDCDJIJ`, below.

.. _CHDCDJIJ:

Available sensitivity maps
--------------------------

In the following, let

.. math::    G_k = [g_{xk} g_{yk} g_{zk}]

denote the three consecutive columns of the gain matrix :math:`G` corresponding to
the fields of three orthogonal dipoles at source space location :math:`k`.
Further, lets assume that the source coordinate system has been
selected so that the :math:`z` -axis points
to the cortical normal direction and the :math:`xy` plane
is thus the tangent plane of the cortex at the source space location :math:`k`
Next, compute the SVD

.. math::    G_k = U_k \Lambda_k V_k

and let :math:`g_{1k} = u_{1k} \lambda_{1k}`, where :math:`\lambda_{1k}` and :math:`u_{1k}` are
the largest singular value and the corresponding left singular vector
of :math:`G_k`, respectively. It is easy to see
that :math:`g_{1k}` is has the largest power
among the signal distributions produced by unit dipoles at source
space location :math:`k`.

Furthermore, assume that the colums orthogonal matrix :math:`U_P` (:math:`U_P^T U_P = I`) contain
the orthogonal basis of the noise subspace corresponding to the signal
space projection (SSP) operator :math:`P` specified
with one or more ``--proj`` options so that :math:`P = I - U_P U_P^T`.
For more information on SSP, see :ref:`CACCHABI`.

With these definitions the map selections defined with the ``--map`` option correspond
to the following

``--map 1``

    Compute :math:`\sqrt{g_{1k}^T g_{1k}} = \lambda_{1k}` at each source space point.
    Normalize the result so that the maximum values equals one.

``--map 2``

    Compute :math:`\sqrt{g_z^T g_z}` at each source space point.
    Normalize the result so that the maximum values equals one. This
    is the amplitude of the signals produced by unit dipoles normal
    to the cortical surface.

``--map 3``

    Compute :math:`\sqrt{g_z^T g_z / g_{1k}^T g_{1k}}` at each source space point.

``--map 4``

    Compute :math:`1 - \sqrt{g_z^T g_z / g_{1k}^T g_{1k}}` at each source space point.
    This could be called the *radiality index*.

``--map 5``

    Compute the subspace correlation between :math:`g_z` and :math:`U_P`: :math:`\text{subcorr}^2(g_z , U_P) = (g_z^T U_P U_P^T g_z)/(g_z^T g_z)`.
    This index equals zero, if :math:`g_z` is
    orthogonal to :math:`U_P` and one if :math:`g_z` lies
    in the subspace defined by :math:`U_P`. This
    map shows how close the field pattern of a dipole oriented perpendicular
    to the cortex at each cortical location is to the subspace removed
    by the SSP.

``--map 6``

    Compute :math:`\sqrt{g_z^T P g_z / g_z^T g_z}`, which is the fraction
    of the field pattern of a dipole oriented perpendicular to the cortex
    at each cortical location remaining after applying the SSP a dipole
    remaining


.. _mne_sensor_locations:

mne_sensor_locations
====================

``--meas <name>``

    Measurement file.

``--magonly``

    Magnetometers only.

``--ell``

    Output sensor ellipsoids for MRIlab.

``--dir``

    Output direction info as well.

``--dev``

    Output in MEG device coordinates.

``--out <name>``

    Name output file


.. _mne_show_fiff:

mne_show_fiff
=============

Using the utility mne_show_fiff it
is possible to display information about the contents of a fif file
to the standard output. The command line options for mne_show_fiff are:

``--in  <*name*>``

    Specifies the fif file whose contents will be listed.

``--verbose``

    Produce a verbose output. The data of most tags is included in the output.
    This excludes matrices and vectors. Only the first 80 characters
    of strings are listed unless the ``--long`` option is present.

``--blocks``

    Only list the blocks (the tree structure) of the file. The tags
    within each block are not listed.

``--indent  <*number*>``

    Number of spaces for indentation for each deeper level in the tree structure
    of the fif files. The default indentation is 3 spaces in terse and
    no spaces in verbose listing mode.

``--long``

    List all data from string tags instead of the first 80 characters.
    This options has no effect unless the ``--verbose`` option
    is also present.

``--tag  <*number*>``

    List only tags of this kind. Multiple ``--tag`` options
    can be specified to list several different kinds of data.

mne_show_fiff reads the
explanations of tag kinds, block kinds, and units from ``$MNE_ROOT/share/mne/fiff_explanations.txt`` .


.. _mne_simu:

mne_simu
========

Purpose
-------

The utility mne_simu creates
simulated evoked response data for investigation of the properties
of the inverse solutions. It computes MEG signals generated by dipoles
normal to the cortical mantle at one or several ROIs defined with
label files. Colored noise can be added to the signals.

Command-line options
--------------------

mne_simu has the following
command-line options:

``--fwd  <*name*>``

    Specify a forward solution file to employ in the simulation.

``--label  <*name*>``

    Specify a label

``--meg``

    Provide MEG data in the output file.

``--eeg``

    Provide EEG data in the output file.

``--out  <*name*>``

    Specify the output file. By default, this will be an evoked data
    file in the fif format.

``--raw``

    Output the data as a raw data fif file instead of an evoked one.

``--mat``

    Produce Matlab output of the simulated fields instead of the fif evoked
    file.

``--label  <*name*>``

    Define an ROI. Several label files can be present. By default, the sources
    in the labels will have :math:`\cos^2` -shaped non-overlapping
    timecourses, see below.

``--timecourse  <*name*>``

    Specifies a text file which contains an expression for a source
    time course, see :ref:`CHDCFIBH`. If no --timecourse options
    are present, the standard source time courses described in :ref:`CHDFIIII` are used. Otherwise, the time course expressions
    are read from the files specified. The time course expressions are
    associated with the labels in the order they are specified. If the
    number of expressions is smaller than the number of labels, the
    last expression specified will reused for the remaining labels.

``--sfreq  <*freq/Hz*>``

    Specifies the sampling frequency of the output data (default = 1000 Hz). This
    option is used only with the time course files.

``--tmin  <*time/ms*>``

    Specifies the starting time of the data, used only with time course files
    (default -200 ms).

``--tmax  <*time/ms*>``

    Specifies the ending time of the data, used only with time course files
    (default 500 ms).

``--seed  <*number*>``

    Specifies the seed for random numbers. This seed is used both for adding
    noise, see :ref:`CHDFBJIJ` and for random numbers in source waveform
    expressions, see :ref:`CHDCFIBH`. If no seed is specified, the
    current time in seconds since Epoch (January 1, 1970) is used.

``--all``

    Activate all sources on the cortical surface uniformly. This overrides the ``--label`` options.

.. _CHDFBJIJ:

Noise simulation
----------------

Noise is added to the signals if the ``--senscov`` and ``--nave`` options
are present. If ``--nave`` is omitted the number of averages
is set to :math:`L = 100`. The noise is computed
by first generating vectors of Gaussian random numbers :math:`n(t)` with :math:`n_j(t) \sim N(0,1)`.
Thereafter, the noise-covariance matrix :math:`C` is
used to color the noise:

.. math::    n_c(t) = \frac{1}{\sqrt{L}} \Lambda U^T n(t)\ ,

where we have used the eigenvalue decomposition positive-definite
covariance matrix:

.. math::    C = U \Lambda^2 U^T\ .

Note that it is assumed that the noise-covariance matrix
is given for raw data, *i.e.*, for :math:`L = 1`.

.. _CHDFIIII:

Simulated data
--------------

The default source waveform :math:`q_k` for
the :math:`k^{th}` label is nonzero at times :math:`t_{kp} = (100(k - 1) + p)/f_s`, :math:`p = 0 \dotso 100` with:

.. math::    q_k(t_{kp}) = Q_k \cos^2{(\frac{\pi p}{100} - \frac{\pi}{2})}\ ,

i.e., the source waveforms are non-overlapping 100-samples
wide :math:`\cos^2` pulses. The sampling frequency :math:`f_s = 600` Hz.
The source amplitude :math:`Q_k` is determined
so that the strength of each of the dipoles in a label will be :math:`50 \text{nAm}/N_k`.

Let us denote the sums of the magnetic fields and electric
potentials produced by the dipoles normal to the cortical mantle
at label :math:`k` by :math:`x_k`. The simulated
signals are then:

.. math::    x(t_j) = \sum_{k = 1}^{N_s} {q_k(t_j) x_k + n_c(t_j)}\ ,

where :math:`N_s` is the number of
sources.

.. _CHDCFIBH:

Source waveform expressions
---------------------------

The ``--timecourse`` option provides flexible possibilities
to define the source waveforms in a functional form. The source
waveform expression files consist of lines of the form:

 <*variable*> ``=``  <*arithmetic expression*>

Each file may contain multiple lines. At the end of the evaluation,
only the values in the variable ``y`` (``q`` )
are significant, see :ref:`CHDJBIEE`. They assume the role
of :math:`q_k(t_j)` to compute the simulated signals
as described in :ref:`CHDFIIII`, above.

All expressions are case insensitive. The variables are vectors
with the length equal to the number of samples in the responses,
determined by the ``--tmin`` , ``--tmax`` , and ``--sfreq`` options.
The available variables are listed in :ref:`CHDJBIEE`.

.. _CHDJBIEE:

.. table:: Available variable names in source waveform expressions.

    ================  =======================================
    Variable          Meaning
    ================  =======================================
    x                 time [s]
    t                 current value of x in [ms]
    y                 the source amplitude [Am]
    q                 synonym for y
    a , b , c , d     help variables, initialized to zeros
    ================  =======================================

The arithmetic expressions can use usual arithmetic operations
as well as  mathematical functions listed in :ref:`CHDJIBHA`.
The arguments can be vectors or scalar numbers. In addition, standard
relational operators ( <, >, ==, <=, >=) and their textual
equivalents (lt, gt, eq, le, ge) are available. Table :ref:`CHDDJEHH` gives some
useful examples of source waveform expressions.

.. tabularcolumns:: |p{0.2\linewidth}|p{0.6\linewidth}|
.. _CHDJIBHA:
.. table:: Mathematical functions available for source waveform expressions

    +-----------------------+---------------------------------------------------------------+
    | Function              | Description                                                   |
    +-----------------------+---------------------------------------------------------------+
    | abs(x)                | absolute value                                                |
    +-----------------------+---------------------------------------------------------------+
    | acos(x)               | :math:`\cos^{-1}x`                                            |
    +-----------------------+---------------------------------------------------------------+
    | asin(x)               | :math:`\sin^{-1}x`                                            |
    +-----------------------+---------------------------------------------------------------+
    | atan(x)               | :math:`\tan^{-1}x`                                            |
    +-----------------------+---------------------------------------------------------------+
    | atan2(x,y)            | :math:`\tan^{-1}(^y/_x)`                                      |
    +-----------------------+---------------------------------------------------------------+
    | ceil(x)               | nearest integer larger than :math:`x`                         |
    +-----------------------+---------------------------------------------------------------+
    | cos(x)                | :math:`\cos x`                                                |
    +-----------------------+---------------------------------------------------------------+
    | cosw(x,a,b,c)         | :math:`\cos^2` -shaped window centered at :math:`b` with a    |
    |                       | rising slope of length :math:`a` and a trailing slope of      |
    |                       | length :math:`b`.                                             |
    +-----------------------+---------------------------------------------------------------+
    | deg(x)                | The value of :math:`x` converted to from radians to degrees   |
    +-----------------------+---------------------------------------------------------------+
    | erf(x)                | :math:`\frac{1}{2\pi} \int_0^x{\text{exp}(-t^2)dt}`           |
    +-----------------------+---------------------------------------------------------------+
    | erfc(x)               | :math:`1 - \text{erf}(x)`                                     |
    +-----------------------+---------------------------------------------------------------+
    | exp(x)                | :math:`e^x`                                                   |
    +-----------------------+---------------------------------------------------------------+
    | floor(x)              | Largest integer value not larger than :math:`x`               |
    +-----------------------+---------------------------------------------------------------+
    | hypot(x,y)            | :math:`\sqrt{x^2 + y^2}`                                      |
    +-----------------------+---------------------------------------------------------------+
    | ln(x)                 | :math:`\ln x`                                                 |
    +-----------------------+---------------------------------------------------------------+
    | log(x)                | :math:`\log_{10} x`                                           |
    +-----------------------+---------------------------------------------------------------+
    | maxp(x,y)             | Takes the maximum between :math:`x` and :math:`y`             |
    +-----------------------+---------------------------------------------------------------+
    | minp(x,y)             | Takes the minimum between :math:`x` and :math:`y`             |
    +-----------------------+---------------------------------------------------------------+
    | mod(x,y)              | Gives the remainder of  :math:`x` divided by :math:`y`        |
    +-----------------------+---------------------------------------------------------------+
    | pi                    | Ratio of the circumference of a circle and its diameter.      |
    +-----------------------+---------------------------------------------------------------+
    | rand                  | Gives a vector of uniformly distributed random numbers        |
    |                       | from 0 to 1.                                                  |
    +-----------------------+---------------------------------------------------------------+
    | rnorm(x,y)            | Gives a vector of Gaussian random numbers distributed as      |
    |                       | :math:`N(x,y)`. Note that if :math:`x` and :math:`y` are      |
    |                       | vectors, each number generated will a different mean and      |
    |                       | variance according to the arguments.                          |
    +-----------------------+---------------------------------------------------------------+
    | shift(x,s)            | Shifts the values in the input vector :math:`x` by the number |
    |                       | of positions given by :math:`s`. Note that :math:`s` must be  |
    |                       | a scalar.                                                     |
    +-----------------------+---------------------------------------------------------------+
    | sin(x)                | :math:`\sin x`                                                |
    +-----------------------+---------------------------------------------------------------+
    | sqr(x)                | :math:`x^2`                                                   |
    +-----------------------+---------------------------------------------------------------+
    | sqrt(x)               | :math:`\sqrt{x}`                                              |
    +-----------------------+---------------------------------------------------------------+
    | tan(x)                | :math:`\tan x`                                                |
    +-----------------------+---------------------------------------------------------------+


.. tabularcolumns:: |p{0.4\linewidth}|p{0.4\linewidth}|
.. _CHDDJEHH:
.. table:: Examples of source waveform expressions.

    +---------------------------------------------+-------------------------------------------------------------+
    | Expression                                  | Meaning                                                     |
    +---------------------------------------------+-------------------------------------------------------------+
    | q = 20e-9*sin(2*pi*10*x)                    | A 10-Hz sine wave with 20 nAm amplitude                     |
    +---------------------------------------------+-------------------------------------------------------------+
    | q = 20e-9*sin(2*pi*2*x)*sin(2*pi*10*x)      | A 10-Hz 20-nAm sine wave, amplitude modulated               |
    |                                             | sinusoidally at 2 Hz.                                       |
    +---------------------------------------------+-------------------------------------------------------------+
    | q = 20e-9*cosw(t,100,100,100)               | :math:`\cos^2`-shaped pulse, centered at :math:`t` = 100 ms |
    |                                             | with 100 ms leading and trailing slopes, 20 nAm amplitude   |
    +---------------------------------------------+-------------------------------------------------------------+
    | q = 30e-9*(t > 0)*(t  <* 300)*sin(2*pi*20*x)| 20-Hz sine wave, 30 nAm amplitude, cropped in time to       |
    |                                             | 0...300 ms.                                                 |
    +---------------------------------------------+-------------------------------------------------------------+


.. _mne_smooth:

mne_smooth
==========

Produce a smoothed version of a w or an stc file

``--src <name>``

    The source space file.

``--in <name>``

    The w or stc file to smooth.

``--smooth <val>``

    Number of smoothsteps


.. _mne_surf2bem:

mne_surf2bem
============

``--surf <*name*>``

    Specifies a FreeSurfer binary format surface file. Before specifying the
    next surface (``--surf`` or ``--tri`` options)
    details of the surface specification can be given with the options
    listed in :ref:`BEHCDICC`.

``--tri <*name*>``

    Specifies a text format surface file. Before specifying the next
    surface (``--surf`` or ``--tri`` options) details
    of the surface specification can be given with the options listed
    in :ref:`BEHCDICC`. The format of these files is described
    in :ref:`BEHDEFCD`.

``--check``

    Check that the surfaces are complete and that they do not intersect. This
    is a recommended option. For more information, see :ref:`BEHCBDDE`.

``--checkmore``

    In addition to the checks implied by the ``--check`` option,
    check skull and skull thicknesses. For more information, see :ref:`BEHCBDDE`.

``--fif <*name*>``

    The output fif file containing the BEM. These files normally reside in
    the bem subdirectory under the subject's mri data. A name
    ending with ``-bem.fif`` is recommended.

.. _BEHCDICC:

Surface options
---------------

These options can be specified after each ``--surf`` or ``--tri`` option
to define details for the corresponding surface.

``--swap``

    Swap the ordering or the triangle vertices. The standard convention in
    the MNE software is to have the vertices ordered so that the vector
    cross product of the vectors from vertex 1 to 2 and 1 to 3 gives the
    direction of the outward surface normal. Text format triangle files
    produced by the some software packages have an opposite order. For
    these files, the ``--swap`` . option is required. This option does
    not have any effect on the interpretation of the FreeSurfer surface
    files specified with the ``--surf`` option.

``--sigma <*value*>``

    The conductivity of the compartment inside this surface in S/m.

``--shift <*value/mm*>``

    Shift the vertices of this surface by this amount, given in mm,
    in the outward direction, *i.e.*, in the positive
    vertex normal direction.

``--meters``

    The vertex coordinates of this surface are given in meters instead
    of millimeters. This option applies to text format files only. This
    definition does not affect the units of the shift option.

``--id <*number*>``

    Identification number to assign to this surface. (1 = inner skull, 3
    = outer skull, 4 = scalp).

``--ico <*number*>``

    Downsample the surface to the designated subdivision of an icosahedron.
    This option is relevant (and required) only if the triangulation
    is isomorphic with a recursively subdivided icosahedron. For example,
    the surfaces produced by with mri_watershed are
    isomorphic with the 5th subdivision of a an icosahedron thus containing 20480
    triangles. However, this number of triangles is too large for present
    computers. Therefore, the triangulations have to be decimated. Specifying ``--ico 4`` yields 5120 triangles per surface while ``--ico 3`` results
    in 1280 triangles. The recommended choice is ``--ico 4`` .


.. _mne_toggle_skips:

mne_toggle_skips
================

Toggle skip tags on and off.

``--raw <name>``

    The raw data file to process.


.. _mne_transform_points:

mne_transform_points
====================

Purpose
-------

mne_transform_points applies
the coordinate transformation relating the MEG head coordinates
and the MRI coordinates to a set of locations listed in a text file.

Command line options
--------------------

mne_transform_points accepts
the following command-line options:

``--in  <*name*>``

    Specifies the input file. The file must contain three numbers on
    each line which are the *x*, *y*,
    and *z* coordinates of point in space. By default,
    the input is in millimeters.

``--iso  <*name*>``

    Specifies a name of a fif file containing Isotrak data. If this
    option is present file will be used as the input instead of the
    text file specified with the ``--in`` option.

``--trans  <*name*>``

    Specifies the name of a fif file containing the coordinate transformation
    between the MEG head coordinates and MRI coordinates. If this file
    is not present, the transformation will be replaced by a unit transform.

``--out  <*name*>``

    Specifies the output file. This file has the same format as the
    input file.

``--hpts``

    Output the data in the head points (hpts)
    format accepted by tkmedit . In
    this format, the coordinates are preceded by a point category (hpi,
    cardinal or fiducial, eeg, extra) and a sequence number, see :ref:`CJADJEBH`.

``--meters``

    The coordinates are listed in meters rather than millimeters.

``--tomri``

    By default, the coordinates are transformed from MRI coordinates to
    MEG head coordinates. This option reverses the transformation to
    be from MEG head coordinates to MRI coordinates.


.. _mne_tufts2fiff:

mne_tufts2fiff
==============

``--raw <*filename*>``

    Specifies the name of the raw data file to process.

``--cal <*filename*>``

    The name of the calibration data file. If calibration data are missing, the
    calibration coefficients will be set to unity.

``--elp <*filename*>``

    The name of the electrode location file. If this file is missing,
    the electrode locations will be unspecified. This file is in the "probe" file
    format used by the *Source Signal Imaging, Inc.* software.
    For description of the format, see http://www.sourcesignal.com/formats_probe.html.
    The fiducial marker locations, optional in the "probe" file
    format specification are mandatory for mne_tufts2fiff . Note
    that some other software packages may produce electrode-position
    files with the elp ending not
    conforming to the above specification.

.. note::

    The conversion process includes a transformation from the Tufts head coordinate system convention to that used in    the Neuromag systems.

.. note::

    The fiducial landmark locations, optional in the probe file format, must be present for mne_tufts2fiff .


.. _mne_view_manual:

mne_view_manual
===============

This script shows you the manual in a PDF reader.


.. _mne_volume_data2mri:

mne_volume_data2mri
===================

With help of the :ref:`mne_volume_source_space` utility
it is possible to create a source space which
is defined within a volume rather than a surface. If the ``--mri`` option
was used in :ref:`mne_volume_source_space`, the
source space file contains an interpolator matrix which performs
a trilinear interpolation into the voxel space of the MRI volume
specified.

The purpose of the :ref:`mne_volume_data2mri` is
to produce MRI overlay data compatible with FreeSurfer MRI viewers
(in the mgh or mgz formats) from this type of w or stc files.

The command-line options are:

``--src <*filename*>``

    The name of the volumetric source space file created with mne_volume_source_space .
    The source space must have been created with the ``--mri`` option,
    which adds the appropriate sparse trilinear interpolator matrix
    to the source space.

``--w <*filename*>``

    The name of a w file to convert
    into an MRI overlay.

``--stc <*filename*>``

    The name of the stc file to convert
    into an MRI overlay. If this file has many time frames, the output
    file may be huge. Note: If both ``-w`` and ``--stc`` are
    specified, ``-w`` takes precedence.

``--scale <*number*>``

    Multiply the stc or w by
    this scaling constant before producing the overlay.

``--out <*filename*>``

    Specifies the name of the output MRI overlay file. The name must end
    with either ``.mgh`` or ``.mgz`` identifying the
    uncompressed and compressed FreeSurfer MRI formats, respectively.


.. _mne_volume_source_space:

mne_volume_source_space
=======================

``--surf <*name*>``

    Specifies a FreeSurfer surface file containing the surface which
    will be used as the boundary for the source space.

``--bem <*name*>``

    Specifies a BEM file (ending in ``-bem.fif`` ). The inner
    skull surface will be used as the boundary for the source space.

``--origin <*x/mm*> : <*y/mm*> : <*z/mm*>``

    If neither of the two surface options described above is present,
    the source space will be spherical with the origin at this location,
    given in MRI (RAS) coordinates.

``--rad <*radius/mm*>``

    Specifies the radius of a spherical source space. Default value
    = 90 mm

``--grid <*spacing/mm*>``

    Specifies the grid spacing in the source space.

``--mindist <*distance/mm*>``

    Only points which are further than this distance from the bounding surface
    are included. Default value = 5 mm.

``--exclude <*distance/mm*>``

    Exclude points that are closer than this distance to the center
    of mass of the bounding surface. By default, there will be no exclusion.

``--mri <*name*>``

    Specifies a MRI volume (in mgz or mgh format).
    If this argument is present the output source space file will contain
    a (sparse) interpolation matrix which allows mne_volume_data2mri to
    create an MRI overlay file, see :ref:`mne_volume_data2mri`.

``--pos <*name*>``

    Specifies a name of a text file containing the source locations
    and, optionally, orientations. Each line of the file should contain
    3 or 6 values. If the number of values is 3, they indicate the source
    location, in millimeters. The orientation of the sources will be
    set to the z-direction. If the number of values is 6, the source
    orientation will be parallel to the vector defined by the remaining
    3 numbers on each line. With ``--pos`` , all of the options
    defined above will be ignored. By default, the source position and
    orientation data are assumed to be given in MRI coordinates.

``--head``

    If this option is present, the source locations and orientations
    in the file specified with the ``--pos`` option are assumed
    to be given in the MEG head coordinates.

``--meters``

    Indicates that the source locations in the file defined with the ``--pos`` option
    are give in meters instead of millimeters.

``--src <*name*>``

    Specifies the output file name. Use a name * <*dir*>/ <*name*>*-src.fif

``--all``

    Include all vertices in the output file, not just those in use.
    This option is implied when the ``--mri`` option is present.
    Even with the ``--all`` option, only those vertices actually
    selected will be marked to be "in use" in the
    output source space file.


.. _mne_watershed_bem:

mne_watershed_bem
=================

``--subject  <*subject*>``

    Defines the name of the subject. This can be also accomplished
    by setting the SUBJECT environment variable.

``--overwrite``

    Overwrite the results of previous run of mne_watershed_bem .

``--atlas``

    Makes mri_watershed to employ
    atlas information to correct the segmentation.
