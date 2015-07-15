

.. _commands_list:

========
Overview
========

List of components
##################

The principal components of the MNE Software and their functions
are listed in :ref:`CHDDJIDB`. Documented software is listed
in italics. :ref:`BABDJHGH` lists various supplementary utilities.

.. tabularcolumns:: |p{0.3\linewidth}|p{0.65\linewidth}|
.. _CHDDJIDB:
.. table:: The software components.

    +----------------------------+--------------------------------------------+
    | Name                       |   Purpose                                  |
    +============================+============================================+
    | *mne_analyze*              | An interactive analysis tool for computing |
    |                            | source estimates, see                      |
    |                            | :ref:`ch_interactive_analysis`.            |
    +----------------------------+--------------------------------------------+
    | *mne_average_estimates*    | Average data across subjects,              |
    |                            | see :ref:`CHDEHFGD`.                       |
    +----------------------------+--------------------------------------------+
    | *mne_browse_raw*           | Interactive raw data browser. Includes     |
    |                            | filtering, offline averaging, and          |
    |                            | computation of covariance matrices,        |
    |                            | see :ref:`ch_browse`.                      |
    +----------------------------+--------------------------------------------+
    | *mne_compute_mne*          | Computes the minimum-norm estimates,       |
    |                            | see :ref:`BABDABHI`. Most of the           |
    |                            | functionality of mne_compute_mne is        |
    |                            | included in mne_make_movie.                |
    +----------------------------+--------------------------------------------+
    | *mne_compute_raw_inverse*  | Compute the inverse solution from raw data |
    |                            | see :ref:`CBBCGHAH`.                       |
    +----------------------------+--------------------------------------------+
    | *mne_convert_mne_data*     | Convert MNE data files to other file       |
    |                            | formats, see :ref:`BEHCCEBJ`.              |
    +----------------------------+--------------------------------------------+
    | *mne_do_forward_solution*  | Convenience script to calculate the forward|
    |                            | solution matrix, see :ref:`BABCHEJD`.      |
    +----------------------------+--------------------------------------------+
    | *mne_do_inverse_operator*  | Convenience script for inverse operator    |
    |                            | decomposition, see :ref:`CIHCFJEI`.        |
    +----------------------------+--------------------------------------------+
    | *mne_forward_solution*     | Calculate the forward solution matrix, see |
    |                            | :ref:`CHDDIBAH`.                           |
    +----------------------------+--------------------------------------------+
    | mne_inverse_operator       | Compute the inverse operator decomposition |
    |                            | see :ref:`CBBDDBGF`.                       |
    +----------------------------+--------------------------------------------+
    | *mne_make_movie*           | Make movies in batch mode, see             |
    |                            | :ref:`CBBECEDE`.                           |
    +----------------------------+--------------------------------------------+
    | *mne_make_source_space*    | Create a *fif* source space description    |
    |                            | file, see :ref:`BEHCGJDD`.                 |
    +----------------------------+--------------------------------------------+
    | *mne_process_raw*          | A batch-mode version of mne_browse_raw,    |
    |                            | see :ref:`ch_browse`.                      |
    +----------------------------+--------------------------------------------+
    | mne_redo_file              | Many intermediate result files contain a   |
    |                            | description of their                       |
    |                            | 'production environment'. Such files can   |
    |                            | be recreated easily with this utility.     |
    |                            | This is convenient if, for example,        |
    |                            | the selection of bad channels is changed   |
    |                            | and the inverse operator decomposition has |
    |                            | to be recalculated.                        |
    +----------------------------+--------------------------------------------+
    | mne_redo_file_nocwd        | Works like mne_redo_file but does not try  |
    |                            | to change in to the working directory      |
    |                            | specified in the 'production environment'. |
    +----------------------------+--------------------------------------------+
    | *mne_setup_forward_model*  | Set up the BEM-related fif files,          |
    |                            | see :ref:`CIHDBFEG`.                       |
    +----------------------------+--------------------------------------------+
    | *mne_setup_mri*            | A convenience script to create the fif     |
    |                            | files describing the anatomical MRI data,  |
    |                            | see :ref:`BABCCEHF`                        |
    +----------------------------+--------------------------------------------+
    | *mne_setup_source_space*   | A convenience script to create source space|
    |                            | description file, see :ref:`CIHCHDAE`.     |
    +----------------------------+--------------------------------------------+
    | mne_show_environment       | Show information about the production      |
    |                            | environment of a file.                     |
    +----------------------------+--------------------------------------------+


.. tabularcolumns:: |p{0.3\linewidth}|p{0.65\linewidth}|
.. _BABDJHGH:
.. table:: Utility programs.

    +---------------------------------+--------------------------------------------+
    | Name                            |   Purpose                                  |
    +=================================+============================================+
    | *mne_add_patch_info*            | Add neighborhood information to a source   |
    |                                 | space file, see :ref:`BEHCBCGG`.           |
    +---------------------------------+--------------------------------------------+
    | *mne_add_to_meas_info*          | Utility to add new information to the      |
    |                                 | measurement info block of a fif file. The  |
    |                                 | source of information is another fif file. |
    +---------------------------------+--------------------------------------------+
    | *mne_add_triggers*              | Modify the trigger channel STI 014 in a raw|
    |                                 | data file, see :ref:`CHDBDDDF`. The same   |
    |                                 | effect can be reached by using an event    |
    |                                 | file for averaging in mne_process_raw and  |
    |                                 | mne_browse_raw.                            |
    +---------------------------------+--------------------------------------------+
    | *mne_annot2labels*              | Convert parcellation data into label files,|
    |                                 | see :ref:`CHDEDHCG`.                       |
    +---------------------------------+--------------------------------------------+
    | *mne_anonymize*                 | Remove subject-specific information from a |
    |                                 | fif data file, see :ref:`CHDIJHIC`.        |
    +---------------------------------+--------------------------------------------+
    | *mne_average_forward_solutions* | Calculate an average of forward solutions, |
    |                                 | see :ref:`CHDBBFCA`.                       |
    +---------------------------------+--------------------------------------------+
    | *mne_brain_vision2fiff*         | Convert EEG data from BrainVision format   |
    |                                 | to fif format, see :ref:`BEHCCCDC`.        |
    +---------------------------------+--------------------------------------------+
    | *mne_change_baselines*          | Change the dc offsets according to         |
    |                                 | specifications given in a text file,       |
    |                                 | see :ref:`CHDDIDCC`.                       |
    +---------------------------------+--------------------------------------------+
    | *mne_change_nave*               | Change the number of averages in an        |
    |                                 | evoked-response data file. This is often   |
    |                                 | necessary if the file was derived from     |
    |                                 | several files.                             |
    +---------------------------------+--------------------------------------------+
    | *mne_check_eeg_locations*       | Checks that the EEG electrode locations    |
    |                                 | have been correctly transferred from the   |
    |                                 | Polhemus data block to the channel         |
    |                                 | information tags, see :ref:`CHDJGGGC`.     |
    +---------------------------------+--------------------------------------------+
    | *mne_check_surface*             | Check the validity of a FreeSurfer surface |
    |                                 | file or one of the surfaces within a BEM   |
    |                                 | file. This program simply checks for       |
    |                                 | topological errors in surface files.       |
    +---------------------------------+--------------------------------------------+
    | *mne_collect_transforms*        | Collect coordinate transformations from    |
    |                                 | several sources into a single fif file,    |
    |                                 | see :ref:`BABBIFIJ`.                       |
    +---------------------------------+--------------------------------------------+
    | *mne_compensate_data*           | Change the applied software gradient       |
    |                                 | compensation in an evoked-response data    |
    |                                 | file, see :ref:`BEHDDFBI`.                 |
    +---------------------------------+--------------------------------------------+
    | *mne_convert_lspcov*            | Convert the LISP format noise covariance   |
    |                                 | matrix output by graph into fif,           |
    |                                 | see :ref:`BEHCDBHG`.                       |
    +---------------------------------+--------------------------------------------+
    | *mne_convert_ncov*              | Convert the ncov format noise covariance   |
    |                                 | file to fif, see :ref:`BEHCHGHD`.          |
    +---------------------------------+--------------------------------------------+
    | *mne_convert_surface*           | Convert FreeSurfer and text format surface |
    |                                 | files into Matlab mat files,               |
    |                                 | see :ref:`BEHDIAJG`.                       |
    +---------------------------------+--------------------------------------------+
    | *mne_cov2proj*                  | Pick eigenvectors from a covariance matrix |
    |                                 | and create a signal-space projection (SSP) |
    |                                 | file out of them, see :ref:`CHDECHBF`.     |
    +---------------------------------+--------------------------------------------+
    | *mne_create_comp_data*          | Create a fif file containing software      |
    |                                 | gradient compensation information from a   |
    |                                 | text file, see :ref:`BEHBIIFF`.            |
    +---------------------------------+--------------------------------------------+
    | *mne_ctf2fiff*                  | Convert a CTF ds folder into a fif file,   |
    |                                 | see :ref:`BEHDEBCH`.                       |
    +---------------------------------+--------------------------------------------+
    | *mne_ctf_dig2fiff*              | Convert text format digitization data to   |
    |                                 | fif format, see :ref:`BEHBABFA`.           |
    +---------------------------------+--------------------------------------------+
    | *mne_dicom_essentials*          | List essential information from a          |
    |                                 | DICOM file.                                |
    |                                 | This utility is used by the script         |
    |                                 | mne_organize_dicom, see :ref:`BABEBJHI`.   |
    +---------------------------------+--------------------------------------------+
    | *mne_edf2fiff*                  | Convert EEG data from the EDF/EDF+/BDF     |
    |                                 | formats to the fif format,                 |
    |                                 | see :ref:`BEHIAADG`.                       |
    +---------------------------------+--------------------------------------------+
    | *mne_epochs2mat*                | Apply bandpass filter to raw data and      |
    |                                 | extract epochs for subsequent processing   |
    |                                 | in Matlab, see :ref:`BEHFIDCB`.            |
    +---------------------------------+--------------------------------------------+
    | *mne_evoked_data_summary*       | List summary of averaged data from a fif   |
    |                                 | file to the standard output.               |
    +---------------------------------+--------------------------------------------+
    | *mne_eximia2fiff*               | Convert EEG data from the Nexstim eXimia   |
    |                                 | system to fif format, see :ref:`BEHGCEHH`. |
    +---------------------------------+--------------------------------------------+
    | *mne_fit_sphere_to_surf*        | Fit a sphere to a surface given in fif     |
    |                                 | or FreeSurfer format, see :ref:`CHDECHBF`. |
    +---------------------------------+--------------------------------------------+
    | *mne_fix_mag_coil_types*        | Update the coil types for magnetometers    |
    |                                 | in a fif file, see :ref:`CHDGAAJC`.        |
    +---------------------------------+--------------------------------------------+
    | *mne_fix_stim14*                | Fix coding errors of trigger channel       |
    |                                 | STI 014, see :ref:`BABCDBDI`.              |
    +---------------------------------+--------------------------------------------+
    | *mne_flash_bem*                 | Create BEM tessellation using multi-echo   |
    |                                 | FLASH MRI data, see :ref:`BABFCDJH`.       |
    +---------------------------------+--------------------------------------------+
    | *mne_insert_4D_comp*            | Read Magnes compensation channel data from |
    |                                 | a text file and merge it with raw data     |
    |                                 | from other channels in a fif file, see     |
    |                                 | :ref:`BEHGDDBH`.                           |
    +---------------------------------+--------------------------------------------+
    | *mne_list_bem*                  | List BEM information in text format,       |
    |                                 | see :ref:`BEHBBEHJ`.                       |
    +---------------------------------+--------------------------------------------+
    | *mne_list_coil_def*             | Create the coil description file. This     |
    |                                 | is run automatically at when the software  |
    |                                 | is set up, see :ref:`BJEHHJIJ`.            |
    +---------------------------------+--------------------------------------------+
    | *mne_list_proj*                 | List signal-space projection data from a   |
    |                                 | fif file.                                  |
    +---------------------------------+--------------------------------------------+
    | *mne_list_source_space*         | List source space information in text      |
    |                                 | format suitable for importing into         |
    |                                 | Neuromag MRIlab, see :ref:`BEHBHIDH`.      |
    +---------------------------------+--------------------------------------------+
    | *mne_list_versions*             | List versions and compilation dates of MNE |
    |                                 | software modules, see :ref:`CHDFIGBG`.     |
    +---------------------------------+--------------------------------------------+
    | *mne_make_cor_set*              | Used by mne_setup_mri to create fif format |
    |                                 | MRI description files from COR or mgh/mgz  |
    |                                 | format MRI data, see :ref:`BABCCEHF`. The  |
    |                                 | mne_make_cor_set utility is described      |
    |                                 | in :ref:`BABBHHHE`.                        |
    +---------------------------------+--------------------------------------------+
    | *mne_make_derivations*          | Create a channel derivation data file, see |
    |                                 | :ref:`CHDHJABJ`.                           |
    +---------------------------------+--------------------------------------------+
    | *mne_make_eeg_layout*           | Make a topographical trace layout file     |
    |                                 | using the EEG electrode locations from     |
    |                                 | an actual measurement, see :ref:`CHDDGDJA`.|
    +---------------------------------+--------------------------------------------+
    | *mne_make_morph_maps*           | Precompute the mapping data needed for     |
    |                                 | morphing between subjects, see             |
    |                                 | :ref:`CHDBBHDH`.                           |
    +---------------------------------+--------------------------------------------+
    | *mne_make_uniform_stc*          | Create a spatially uniform stc file for    |
    |                                 | testing purposes.                          |
    +---------------------------------+--------------------------------------------+
    | *mne_mark_bad_channels*         | Update the list of unusable channels in    |
    |                                 | a data file, see :ref:`CHDDHBEE`.          |
    +---------------------------------+--------------------------------------------+
    | *mne_morph_labels*              | Morph label file definitions between       |
    |                                 | subjects, see :ref:`CHDCEAFC`.             |
    +---------------------------------+--------------------------------------------+
    | *mne_organize_dicom*            | Organized DICOM MRI image files into       |
    |                                 | directories, see :ref:`BABEBJHI`.          |
    +---------------------------------+--------------------------------------------+
    | *mne_prepare_bem_model*         | Perform the geometry calculations for      |
    |                                 | BEM forward solutions, see :ref:`CHDJFHEB`.|
    +---------------------------------+--------------------------------------------+
    | mne_process_stc                 | Manipulate stc files.                      |
    +---------------------------------+--------------------------------------------+
    | *mne_raw2mat*                   | Convert raw data into a Matlab file,       |
    |                                 | see :ref:`convert_to_matlab`.              |
    +---------------------------------+--------------------------------------------+
    | *mne_rename_channels*           | Change the names and types of channels     |
    |                                 | in a fif file, see :ref:`CHDCFEAJ`.        |
    +---------------------------------+--------------------------------------------+
    | *mne_sensitivity_map*           | Compute a sensitivity map and output       |
    |                                 | the result in a w-file,                    |
    |                                 | see :ref:`CHDDCBGI`.                       |
    +---------------------------------+--------------------------------------------+
    | *mne_sensor_locations*          | Create a file containing the sensor        |
    |                                 | locations in text format.                  |
    +---------------------------------+--------------------------------------------+
    | *mne_show_fiff*                 | List contents of a fif file,               |
    |                                 | see :ref:`CHDHEDEF`.                       |
    +---------------------------------+--------------------------------------------+
    | *mne_simu*                      | Simulate MEG and EEG data,                 |
    |                                 | see :ref:`CHDECAFD`.                       |
    +---------------------------------+--------------------------------------------+
    | *mne_smooth*                    | Smooth a w or stc file.                    |
    +---------------------------------+--------------------------------------------+
    | *mne_surf2bem*                  | Create a *fif* file describing the         |
    |                                 | triangulated compartment boundaries for    |
    |                                 | the boundary-element model (BEM),          |
    |                                 | see :ref:`BEHCACCJ`.                       |
    +---------------------------------+--------------------------------------------+
    | *mne_toggle_skips*              | Change data skip tags in a raw file into   |
    |                                 | ignored skips or vice versa.               |
    +---------------------------------+--------------------------------------------+
    | *mne_transform_points*          | Transform between MRI and MEG head         |
    |                                 | coordinate frames, see :ref:`CHDDDJCA`.    |
    +---------------------------------+--------------------------------------------+
    | *mne_tufts2fiff*                | Convert EEG data from the Tufts            |
    |                                 | University format to fif format,           |
    |                                 | see :ref:`BEHDGAIJ`.                       |
    +---------------------------------+--------------------------------------------+
    | *mne_view_manual*               | Starts a PDF reader to show this manual    |
    |                                 | from its standard location.                |
    +---------------------------------+--------------------------------------------+
    | *mne_volume_data2mri*           | Convert volumetric data defined in a       |
    |                                 | source space created with                  |
    |                                 | mne_volume_source_space into an MRI        |
    |                                 | overlay, see :ref:`BEHDEJEC`.              |
    +---------------------------------+--------------------------------------------+
    | *mne_volume_source_space*       | Make a volumetric source space,            |
    |                                 | see :ref:`BJEFEHJI`.                       |
    +---------------------------------+--------------------------------------------+
    | *mne_watershed_bem*             | Do the segmentation for BEM using the      |
    |                                 | watershed algorithm, see :ref:`BABBDHAG`.  |
    +---------------------------------+--------------------------------------------+


File formats
############

The MNE software employs the fif file format whenever possible.
New tags have been added to incorporate information specific to
the calculation of cortically contained source estimates. FreeSurfer
file formats are also employed when needed to represent cortical
surface geometry data as well as spatiotemporal distribution of
quantities on the surfaces. Of particular interest are the w files,
which contain static overlay data on the cortical surface and stc files,
which contain dynamic overlays (movies).

Conventions
###########

When command line examples are shown, the backslash character
(\\) indicates a continuation line. It is also valid in the shells.
In most cases, however, you can easily fit the commands listed in
this manual on one line and thus omit the backslashes. The order
of options  is irrelevant. Entries to be typed literally are shown
like ``this`` . *Italicized* text indicates
conceptual entries. For example, *<*dir*>* indicates a directory
name.

In the description of interactive software modules the notation <*menu*>/<*item*> is
often used to denotes menu selections. For example, File/Quit stands
for the Quit button in the File menu.

All software modules employ the double-dash (``--``) option convention, *i.e.*, the
option names are preceded by two dashes.

Most of the programs have two common options to obtain general
information:

**\---help**

    Prints concise usage information.

**\---version**

    Prints the program module name, version number, and compilation date.
