

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
    |                            | see :ref:`BABDABHI`. Most of the           |
    |                            | functionality of mne_compute_mne is        |
    |                            | included in mne_make_movie.                |
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


Utilities
=========

.. tabularcolumns:: |p{0.3\linewidth}|p{0.65\linewidth}|
.. _BABDJHGH:
.. table::

    +----------------------------------+--------------------------------------------+
    | Name                             |   Purpose                                  |
    +==================================+============================================+
    | `mne_add_patch_info`_            | Add neighborhood information to a source   |
    |                                  | space file, see :ref:`BEHCBCGG`.           |
    +----------------------------------+--------------------------------------------+
    | `mne_add_to_meas_info`_          | Utility to add new information to the      |
    |                                  | measurement info block of a fif file. The  |
    |                                  | source of information is another fif file. |
    +----------------------------------+--------------------------------------------+
    | `mne_add_triggers`_              | Modify the trigger channel STI 014 in a raw|
    |                                  | data file, see :ref:`CHDBDDDF`. The same   |
    |                                  | effect can be reached by using an event    |
    |                                  | file for averaging in mne_process_raw and  |
    |                                  | mne_browse_raw.                            |
    +----------------------------------+--------------------------------------------+
    | `mne_annot2labels`_              | Convert parcellation data into label files,|
    |                                  | see :ref:`CHDEDHCG`.                       |
    +----------------------------------+--------------------------------------------+
    | `mne_anonymize`_                 | Remove subject-specific information from a |
    |                                  | fif data file, see :ref:`CHDIJHIC`.        |
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
    |                                  | information tags, see :ref:`CHDJGGGC`.     |
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
    |                                  | file out of them, see :ref:`CHDECHBF`.     |
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
    |                                  | or FreeSurfer format, see :ref:`CHDECHBF`. |
    +----------------------------------+--------------------------------------------+
    | `mne_fix_mag_coil_types`_        | Update the coil types for magnetometers    |
    |                                  | in a fif file, see :ref:`CHDGAAJC`.        |
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
    |                                  | software modules, see :ref:`CHDFIGBG`.     |
    +----------------------------------+--------------------------------------------+
    | `mne_make_cor_set`_              | Used by mne_setup_mri to create fif format |
    |                                  | MRI description files from COR or mgh/mgz  |
    |                                  | format MRI data, see :ref:`BABCCEHF`. The  |
    |                                  | mne_make_cor_set utility is described      |
    |                                  | in :ref:`BABBHHHE`.                        |
    +----------------------------------+--------------------------------------------+
    | `mne_make_derivations`_          | Create a channel derivation data file, see |
    |                                  | :ref:`CHDHJABJ`.                           |
    +----------------------------------+--------------------------------------------+
    | `mne_make_eeg_layout`_           | Make a topographical trace layout file     |
    |                                  | using the EEG electrode locations from     |
    |                                  | an actual measurement, see :ref:`CHDDGDJA`.|
    +----------------------------------+--------------------------------------------+
    | `mne_make_morph_maps`_           | Precompute the mapping data needed for     |
    |                                  | morphing between subjects, see             |
    |                                  | :ref:`CHDBBHDH`.                           |
    +----------------------------------+--------------------------------------------+
    | `mne_make_uniform_stc`_          | Create a spatially uniform stc file for    |
    |                                  | testing purposes.                          |
    +----------------------------------+--------------------------------------------+
    | `mne_mark_bad_channels`_         | Update the list of unusable channels in    |
    |                                  | a data file, see :ref:`CHDDHBEE`.          |
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
    |                                  | in a fif file, see :ref:`CHDCFEAJ`.        |
    +----------------------------------+--------------------------------------------+
    | `mne_sensitivity_map`_           | Compute a sensitivity map and output       |
    |                                  | the result in a w-file,                    |
    |                                  | see :ref:`CHDDCBGI`.                       |
    +----------------------------------+--------------------------------------------+
    | `mne_sensor_locations`_          | Create a file containing the sensor        |
    |                                  | locations in text format.                  |
    +----------------------------------+--------------------------------------------+
    | `mne_show_fiff`_                 | List contents of a fif file,               |
    |                                  | see :ref:`CHDHEDEF`.                       |
    +----------------------------------+--------------------------------------------+
    | `mne_simu`_                      | Simulate MEG and EEG data,                 |
    |                                  | see :ref:`CHDECAFD`.                       |
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
    |                                  | coordinate frames, see :ref:`CHDDDJCA`.    |
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


.. _mne_compute_mne:

mne_compute_mne
===============


.. _mne_compute_raw_inverse:

mne_compute_raw_inverse
=======================


.. _mne_convert_mne_data:

mne_convert_mne_data
====================

.. _mne_do_forward_solution:

mne_do_forward_solution
=======================


.. _mne_do_inverse_operator:

mne_do_inverse_operator
=======================


.. _mne_forward_solution:

mne_forward_solution
====================


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
    file contains a projection operator, attached by mne_browse_raw and mne_process_raw ,
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
    one produced by mne_browse_raw or mne_process_raw , the
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


.. _mne_setup_forward_model:

mne_setup_forward_model
=======================


.. _mne_setup_mri:

mne_setup_mri
=============


.. _mne_setup_source_space:

mne_setup_source_space
======================



----

mne_add_patch_info
==================


mne_add_to_meas_info
====================


mne_add_triggers
================


mne_annot2labels
================


mne_anonymize
=============


mne_average_forward_solutions
=============================


mne_brain_vision2fiff
=====================


mne_change_baselines
====================


mne_change_nave
===============


mne_check_eeg_locations
=======================


mne_check_surface
=================


mne_collect_transforms
======================


mne_compensate_data
===================


mne_convert_lspcov
==================


mne_convert_ncov
================


mne_convert_surface
===================


mne_cov2proj
============


mne_create_comp_data
====================


mne_ctf2fiff
============


mne_ctf_dig2fiff
================


mne_dicom_essentials
====================


mne_edf2fiff
============


mne_epochs2mat
==============


mne_evoked_data_summary
=======================


mne_eximia2fiff
===============


mne_fit_sphere_to_surf
======================


mne_fix_mag_coil_types
======================


mne_fix_stim14
==============


mne_flash_bem
=============


mne_insert_4D_comp
==================


mne_list_bem
============


mne_list_coil_def
=================


mne_list_proj
=============


mne_list_source_space
=====================


mne_list_versions
=================


mne_make_cor_set
================


mne_make_derivations
====================


mne_make_eeg_layout
===================


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


mne_make_uniform_stc
====================


mne_mark_bad_channels
=====================


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


mne_organize_dicom
==================


mne_prepare_bem_model
=====================


mne_raw2mat
===========


mne_rename_channels
===================


mne_sensitivity_map
===================


mne_sensor_locations
====================


mne_show_fiff
=============


mne_simu
========


mne_smooth
==========


mne_surf2bem
============


mne_toggle_skips
================


mne_transform_points
====================


mne_tufts2fiff
==============


mne_view_manual
===============


mne_volume_data2mri
===================


mne_volume_source_space
=======================


mne_watershed_bem
=================
