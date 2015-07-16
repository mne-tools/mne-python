

.. _commands_list:

===============
C API Reference
===============

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
    |                            | see :ref:`CBBCGHAH`.                       |
    +----------------------------+--------------------------------------------+
    | `mne_convert_mne_data`_    | Convert MNE data files to other file       |
    |                            | formats, see :ref:`BEHCCEBJ`.              |
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
    |                            | see :ref:`CBBDDBGF`.                       |
    +----------------------------+--------------------------------------------+
    | `mne_make_movie`_          | Make movies in batch mode, see             |
    |                            | :ref:`CBBECEDE`.                           |
    +----------------------------+--------------------------------------------+
    | `mne_make_source_space`_   | Create a *fif* source space description    |
    |                            | file, see :ref:`BEHCGJDD`.                 |
    +----------------------------+--------------------------------------------+
    | `mne_process_raw`_         | A batch-mode version of mne_browse_raw,    |
    |                            | see :ref:`ch_browse`.                      |
    +----------------------------+--------------------------------------------+
    | ``mne_redo_file``          | Many intermediate result files contain a   |
    |                            | description of their                       |
    |                            | 'production environment'. Such files can   |
    |                            | be recreated easily with this utility.     |
    |                            | This is convenient if, for example,        |
    |                            | the selection of bad channels is changed   |
    |                            | and the inverse operator decomposition has |
    |                            | to be recalculated.                        |
    +----------------------------+--------------------------------------------+
    | ``mne_redo_file_nocwd``    | Works like mne_redo_file but does not try  |
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
    |                            | description file, see :ref:`CIHCHDAE`.     |
    +----------------------------+--------------------------------------------+
    | ``mne_show_environment``   | Show information about the production      |
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
    |                                  | to fif format, see :ref:`BEHCCCDC`.        |
    +----------------------------------+--------------------------------------------+
    | `mne_change_baselines`_          | Change the dc offsets according to         |
    |                                  | specifications given in a text file,       |
    |                                  | see :ref:`CHDDIDCC`.                       |
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
    |                                  | several sources into a single fif file,    |
    |                                  | see :ref:`BABBIFIJ`.                       |
    +----------------------------------+--------------------------------------------+
    | `mne_compensate_data`_           | Change the applied software gradient       |
    |                                  | compensation in an evoked-response data    |
    |                                  | file, see :ref:`BEHDDFBI`.                 |
    +----------------------------------+--------------------------------------------+
    | `mne_copy_processing_history`_   | Copy the processing history between files. |
    +----------------------------------+--------------------------------------------+
    | `mne_convert_lspcov`_            | Convert the LISP format noise covariance   |
    |                                  | matrix output by graph into fif,           |
    |                                  | see :ref:`BEHCDBHG`.                       |
    +----------------------------------+--------------------------------------------+
    | `mne_convert_ncov`_              | Convert the ncov format noise covariance   |
    |                                  | file to fif, see :ref:`BEHCHGHD`.          |
    +----------------------------------+--------------------------------------------+
    | `mne_convert_surface`_           | Convert FreeSurfer and text format surface |
    |                                  | files into Matlab mat files,               |
    |                                  | see :ref:`BEHDIAJG`.                       |
    +----------------------------------+--------------------------------------------+
    | `mne_cov2proj`_                  | Pick eigenvectors from a covariance matrix |
    |                                  | and create a signal-space projection (SSP) |
    |                                  | file out of them.     |
    +----------------------------------+--------------------------------------------+
    | `mne_create_comp_data`_          | Create a fif file containing software      |
    |                                  | gradient compensation information from a   |
    |                                  | text file, see :ref:`BEHBIIFF`.            |
    +----------------------------------+--------------------------------------------+
    | `mne_ctf2fiff`_                  | Convert a CTF ds folder into a fif file,   |
    |                                  | see :ref:`BEHDEBCH`.                       |
    +----------------------------------+--------------------------------------------+
    | `mne_ctf_dig2fiff`_              | Convert text format digitization data to   |
    |                                  | fif format, see :ref:`BEHBABFA`.           |
    +----------------------------------+--------------------------------------------+
    | `mne_dicom_essentials`_          | List essential information from a          |
    |                                  | DICOM file.                                |
    |                                  | This utility is used by the script         |
    |                                  | mne_organize_dicom, see :ref:`BABEBJHI`.   |
    +----------------------------------+--------------------------------------------+
    | `mne_edf2fiff`_                  | Convert EEG data from the EDF/EDF+/BDF     |
    |                                  | formats to the fif format,                 |
    |                                  | see :ref:`BEHIAADG`.                       |
    +----------------------------------+--------------------------------------------+
    | `mne_epochs2mat`_                | Apply bandpass filter to raw data and      |
    |                                  | extract epochs for subsequent processing   |
    |                                  | in Matlab, see :ref:`BEHFIDCB`.            |
    +----------------------------------+--------------------------------------------+
    | `mne_evoked_data_summary`_       | List summary of averaged data from a fif   |
    |                                  | file to the standard output.               |
    +----------------------------------+--------------------------------------------+
    | `mne_eximia2fiff`_               | Convert EEG data from the Nexstim eXimia   |
    |                                  | system to fif format, see :ref:`BEHGCEHH`. |
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
    |                                  | from other channels in a fif file, see     |
    |                                  | :ref:`BEHGDDBH`.                           |
    +----------------------------------+--------------------------------------------+
    | `mne_list_bem`_                  | List BEM information in text format,       |
    |                                  | see :ref:`BEHBBEHJ`.                       |
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
    |                                  | Neuromag MRIlab, see :ref:`BEHBHIDH`.      |
    +----------------------------------+--------------------------------------------+
    | `mne_list_versions`_             | List versions and compilation dates of MNE |
    |                                  | software modules.                          |
    +----------------------------------+--------------------------------------------+
    | `mne_make_cor_set`_              | Used by mne_setup_mri to create fif format |
    |                                  | MRI description files from COR or mgh/mgz  |
    |                                  | format MRI data, see :ref:`BABCCEHF`. The  |
    |                                  | mne_make_cor_set utility is described      |
    |                                  | in :ref:`BABBHHHE`.                        |
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
    |                                  | subjects, see :ref:`CHDCEAFC`.             |
    +----------------------------------+--------------------------------------------+
    | `mne_organize_dicom`_            | Organized DICOM MRI image files into       |
    |                                  | directories, see :ref:`BABEBJHI`.          |
    +----------------------------------+--------------------------------------------+
    | `mne_prepare_bem_model`_         | Perform the geometry calculations for      |
    |                                  | BEM forward solutions, see :ref:`CHDJFHEB`.|
    +----------------------------------+--------------------------------------------+
    | ``mne_process_stc``              | Manipulate stc files.                      |
    +----------------------------------+--------------------------------------------+
    | `mne_raw2mat`_                   | Convert raw data into a Matlab file,       |
    |                                  | see :ref:`convert_to_matlab`.              |
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
    | `mne_show_fiff`_                 | List contents of a fif file,               |
    |                                  | see :ref:`CHDHEDEF`.                       |
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
    |                                  | University format to fif format,           |
    |                                  | see :ref:`BEHDGAIJ`.                       |
    +----------------------------------+--------------------------------------------+
    | `mne_view_manual`_               | Starts a PDF reader to show this manual    |
    |                                  | from its standard location.                |
    +----------------------------------+--------------------------------------------+
    | `mne_volume_data2mri`_           | Convert volumetric data defined in a       |
    |                                  | source space created with                  |
    |                                  | mne_volume_source_space into an MRI        |
    |                                  | overlay, see :ref:`BEHDEJEC`.              |
    +----------------------------------+--------------------------------------------+
    | `mne_volume_source_space`_       | Make a volumetric source space,            |
    |                                  | see :ref:`BJEFEHJI`.                       |
    +----------------------------------+--------------------------------------------+
    | `mne_watershed_bem`_             | Do the segmentation for BEM using the      |
    |                                  | watershed algorithm, see :ref:`BABBDHAG`.  |
    +----------------------------------+--------------------------------------------+


Command-line arguments
######################

Most of the programs have two common options to obtain general
information:

``---help``

    Prints concise usage information.

``---version``

    Prints the program module name, version number, and compilation date.


.. _mne_analyze:

mne_analyze
===========

Since mne_analyze is primarily an interactive analysis tool, there are only a
few command-line options:

``\---version``

    Show the program version and compilation date.

``\---help``

    List the command-line options.

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

``---version``

    Show the program version and compilation date.

``---help``

    List the command-line options.

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

``--version``

    Show the program version and compilation date.

``--help``

    List the command-line options.

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
see :ref:`CBBECEDE`. At this time, :ref:`mne_compute_mne` is
still needed to produce time-collapsed w files unless you are willing
to write a Matlab script of your own for this purpose.


``--version``

    Show the program version and compilation date.

``--help``

    List the command-line options.

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

``--version``

    Show the program version and compilation date.

``--help``

    List the command-line options.

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
    see :ref:`CBBHJDAI`. The label files are produced by tksurfer
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
    the label. For more information, see :ref:`CBBHJDAI`. This
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
XXX from IO


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
    see :ref:`CBBDFJIE` and :ref:`CBBDDBGF`.

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
    Suitable values are in the range 0.05...0.2. For details, see :ref:`CBBHEGAB`.

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
    of the weighting is discussed in :ref:`CBBDIJHI`.

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

``--version``

    Show the program version and compilation date.

``--help``

    List the command-line options.

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
    have the patch information computed, see :ref:`CIHCHDAE`.

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
    amount. The value is restricted to the range 0...1. For details, see :ref:`CBBHEGAB`.

``--magreg <value>``

    Regularize the magnetometer and axial gradiometer section (channels
    for which the unit of measurement is T) of the noise-covariance matrix
    by the given amount. The value is restricted to the range 0...1.
    For details, see :ref:`CBBHEGAB`.

``--eegreg <value>``

    Regularize the EEG section of the noise-covariance matrix by the given
    amount. The value is restricted to the range 0...1. For details, see :ref:`CBBHEGAB`.

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

    Employ depth weighting. For details, see :ref:`CBBDFJIE`.

``--weightexp <value>``

    This parameter determines the steepness of the depth weighting function
    (default = 0.8). For details, see :ref:`CBBDFJIE`.

``--weightlimit <value>``

    Maximum relative strength of the depth weighting (default = 10). For
    details, see :ref:`CBBDFJIE`.

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
    of the weighting is discussed in :ref:`CBBDIJHI`.

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
    already when the forward solution is computed, see :ref:`CIHCHDAE` and :ref:`BABCHEJD`.
    For technical details of the patch information, please consult :ref:`CBBDBHDI`. This option is considered experimental at
    the moment.

``--inv <name>``

    Save the inverse operator decomposition here.


.. _mne_make_movie:

mne_make_movie
==============

General options
---------------

``--version``

    Show the program version and compilation date.

``--help``

    List the command-line options.

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
    combine data from different subjects, see :ref:`CHDFDIFE`.
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
    the overlayed color will be equal to the background surface color.
    For currents, the value will be multiplied by :math:`1^{-10}`.
    The default value is 8.

``--fmid <*value*>``

    Specifies the midpoint for the displayed colormaps. At this value, the
    overlayed color will be read (positive values) or blue (negative values).
    For currents, the value will be multiplied by :math:`1^{-10}`.
    The default value is 15.

``--fmax <*value*>``

    Specifies the maximum point for the displayed colormaps. At this value,
    the overlayed color will bright yellow (positive values) or light
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


.. _mne_process_raw:

mne_process_raw
===============

``--version``

    Show the program version and compilation date.

``--help``

    List the command-line options.

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


.. _mne_setup_source_space:

mne_setup_source_space
======================

``--version``

    Show the program version and compilation date.

``--help``

    List the command-line options.

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



----

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

``--version``

    Show the program version and compilation date.

``--help``

    List the command-line options.

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

``--version``

    Show the program version and compilation date.

``--help``

    List the command-line options.

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

``--version``

    Show the program version and compilation date.

``--help``

    List the command-line options.

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

``--version``

    Show the program version and compilation date.

``--help``

    List the command-line options.

``--his``

    Remove the FIFF_SUBJ_HIS_ID tag as well, see above.

``--file  <*name*>``

    Specifies the name of the file to be modified.

.. note:: You need write permission to the file to be    processed.


.. _mne_average_forward_solutions:

mne_average_forward_solutions
=============================

``--version``

    Show the program version and compilation date.

``--help``

    List the command-line options.

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


.. _mne_change_baselines:

mne_change_baselines
====================


.. _mne_change_nave:

mne_change_nave
===============


.. _mne_check_eeg_locations:

mne_check_eeg_locations
=======================

Some versions of the Neuromag acquisition software did not
copy the EEG channel location information properly from the Polhemus
digitizer information data block to the EEG channel information
records if the number of EEG channels exceeds 60. The purpose of mne_check_eeg_locations is
to detect this problem and fix it, if requested. The command-line
options are:

``--version``

    Show the program version and compilation date.

``--help``

    List the command-line options.

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


.. _mne_collect_transforms:

mne_collect_transforms
======================


.. _mne_compensate_data:

mne_compensate_data
===================


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


.. _mne_convert_lspcov:

mne_convert_lspcov
==================


.. _mne_convert_ncov:

mne_convert_ncov
================


.. _mne_convert_surface:

mne_convert_surface
===================


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

``--version``

    Show the program version and compilation date.

``--help``

    List the command-line options.

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


.. _mne_ctf2fiff:

mne_ctf2fiff
============


.. _mne_ctf_dig2fiff:

mne_ctf_dig2fiff
================


.. _mne_dicom_essentials:

mne_dicom_essentials
====================


.. _mne_edf2fiff:

mne_edf2fiff
============


.. _mne_epochs2mat:

mne_epochs2mat
==============


.. _mne_evoked_data_summary:

mne_evoked_data_summary
=======================


.. _mne_eximia2fiff:

mne_eximia2fiff
===============


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

``--version``

    Show the program version and compilation date.

``--help``

    List the command-line options.

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


.. _mne_list_bem:

mne_list_bem
============


.. _mne_list_coil_def:

mne_list_coil_def
=================


.. _mne_list_proj:

mne_list_proj
=============


.. _mne_list_source_space:

mne_list_source_space
=====================


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

``--version``

    Show the program version and compilation date.

``--help``

    List the command-line options.

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

.. note:: You might consider renaming the EEG channels    with descriptive labels related to the standard 10-20 system using    the mne_rename_channels utility,    see :ref:`CHDCFEAJ`. This allows you to use standard EEG    channel names in the derivations you define as well as in the channel    selection files used in mne_browse_raw ,    see :ref:`CACCJEJD`.


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

``--version``

    Show the program version and compilation date.

``--help``

    List the command-line options.

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

``--version``

    Show the program version and compilation date.

``--help``

    List the command-line options.

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


.. _mne_mark_bad_channels:

mne_mark_bad_channels
=====================

This utility adds or replaces information about unusable
(bad) channels. The command line options are:

``--version``

    Show the program version and compilation date.

``--help``

    List the command-line options.

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

``--version``

    Show the program version and compilation date.

``--help``

    List the command-line options.

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


.. _mne_raw2mat:

mne_raw2mat
===========


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

``--version``

    Show the program version and compilation date.

``--help``

    List the command-line options.

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

``--version``

    Show the program version and compilation date.

``--help``

    List the command-line options.

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


.. _mne_show_fiff:

mne_show_fiff
=============

Using the utility mne_show_fiff it
is possible to display information about the contents of a fif file
to the standard output. The command line options for mne_show_fiff are:

``--version``

    Show the program version and compilation date.

``--help``

    List the command-line options.

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

``--version``

    Show the program version and compilation date.

``--help``

    List the command-line options.

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


.. _mne_surf2bem:

mne_surf2bem
============

``--version``

    Show the program version and compilation date.

``--help``

    List the command-line options.

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

``--version``

    Show the program version and compilation date.

``--help``

    List the command-line options.

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

.. _CHDDIDCC:

Inquiring and changing baselines
################################

The utility mne_change_baselines computes
baseline values and applies them to an evoked-response data file.
The command-line options are:

``--version``

    Show the program version and compilation date.

``--help``

    List the command-line options.

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


.. _mne_tufts2fiff:

mne_tufts2fiff
==============


.. _mne_view_manual:

mne_view_manual
===============


.. _mne_volume_data2mri:

mne_volume_data2mri
===================


.. _mne_volume_source_space:

mne_volume_source_space
=======================

``--version``

    Show the program version and compilation date.

``--help``

    List the command-line options.

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
    create an MRI overlay file, see :ref:`BEHDEJEC`.

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
