

.. _release_notes:

=============
Release notes
=============

.. contents:: Contents
   :local:
   :depth: 2


This appendix contains a brief description of the changes
in MNE software in each major release.

Release notes for MNE software 2.4
##################################

Manual
======

The manual has been significantly expanded and reorganized.
:ref:`ch_interactive_analysis` describing mne_analyze has
been added. :ref:`ch_sample_data` contains instructions for analyzing
the sample data set provided with the software. Useful background
material is listed in :ref:`ch_reading`. Almost all utility programs
are now covered in the manual.

General software changes
========================

The following overall changes have been made:

- A forward solution library independent
  of Neuromag software was written.

- The MEG sensor information is now imported from the coil definition file
  instead of being hardcoded in the software.

- CTF and 4D Neuroimaging sensors are now supported.

- The number of Neuromag-based utilities was minimized.

- The LINUX port of Neuromag software modules was completely
  separated from the MNE software and now resides under a separate
  directory tree.

- Support for topologically connected source spaces was added.

- A lot of bugs were fixed.

File conversion utilities
=========================

The following import utilities were added:

- mne_ctf2fiff to convert CTF data to the fif format.

- mne_tufts2fiff to convert
  EEG data from Tufts university to fif format.

The output of the Matlab conversion utilities was changed
to use structures.

Matlab tools to import and export w and stc files were added.

mne_browse_raw
==============

Output of decimated and filtered data is now available. mne_analyze now fully
supports 32-bit integer data found in CTF and new Neuromag raw data
files.

mne_analyze
===========

The following changes have been made in mne_analyze :

- Curved and flat surface patches are
  now supported.

- An iterative coordinate alignment procedure was added, see
  :ref:`CACEHGCD`.

- Utility to view continuous HPI information was added.

- Several small changes and bug fixes were done.

mne_make_movie
==============

The only major change in mne_make_movie is
the addition of support for curved and surface patches.

Averaging
=========

The highly inefficient program mne_grand_average has
been removed from the distribution and replaced with the combined
use of mne_make_movie and a new
averaging program mne_average_estimates.

Release notes for MNE software 2.5
##################################

Manual
======

The MNE Matlab toolbox is now covered in a separate chapter.
Change bars are employed to indicate changes in the chapters that
existed in the previous version of the manual. Note that :ref:`ch_matlab` describing
the Matlab toolbox is totally new and change bars have not been
used there. Furthermore, :ref:`setup_martinos` now contains all the
information specific to the Martinos Center.

mne_browse_raw
==============

There are several improvements in the raw data processor mne_browse_raw/mne_process_raw :

- Possibility to delete and add channel
  selections interactively has been added. A nonstandard channel selection
  file can be now specified on the command line.

- Handling of CTF software gradient compensation has been added.

- The vertical scale of the digital trigger channel is now automatically
  set to accommodate the largest trigger value.

- It is now possible to load evoked-response data sets from
  files. Time scales of the evoked-response data and data averaged
  in mne_browse_raw can be now
  set from the scales dialog. :ref:`CHDHBGGH` has
  been updated to employ mne_browse_raw in
  viewing the averages computed from the sample raw data set.

- It is now possible to create new SSP operators in mne_browse_raw.

- Listing of amplitude values have been added to both the strip-chart
  and topographical displays.

- Text format event files can now be loaded for easy inspection
  of rejected epochs, for example.

- Handling of derived channels has been added.

- SSS information is now transferred to the covariance matrix
  output files.

- Neuromag processing history is included with the output files.

mne_epochs2mat
==============

This new utility extracts epochs from a raw data file, applies
a bandpass filter to them and outputs them in a format convenient
for processing in Matlab.

mne_analyze
===========

The following new features have been added:

- Processing of raw data segment and easy
  switching between multiple evoked data sets (not in the manual yet).

- Sketchy surface display mode for source spaces with selection
  triangulation information created with the ``--ico`` option
  to mne_setup_source_space.

- Rotation of the coordinate frame in the coordinate system
  alignment dialog.

- Several new graphics output file formats as well as automatic
  and snapshot output modes.

- It is now possible to inquire timecourses from stc overlays.
  Both labels and surface picking are supported.

- Added an option to include surface vertex numbers to the timecourse output.

- Overlays matching the scalp surface can now be loaded.

- The dipole display dialog has now control over the dipole
  display properties. Multiple dipoles can be now displayed.

- Time stepping with cursor keys has been added.

- Dynamic cursors have been added to the full view display.

- The viewer display now automatically rotates to facilitate
  fiducial picking from the head surface.

mne_ctf2fiff
============

Correct errors in compensation channel information and compensation data
output. The transformation between the CTF and Neuromag coordinate
frames is now included in the output file.

mne_make_movie
==============

Added the ``--labelverts`` option.

mne_surf2bem
============

Added the ``--shift`` option to move surface vertices
outwards. Fixed some loopholes in topology checks. Also added the ``--innershift`` option
to mne_setup_forward_model.

mne_forward_solution
====================

Added code to compute forward solutions for CTF data with
software gradient compensation on.

mne_inverse_operator
====================

The following changes have been made in mne_inverse_operator :

- Added options to regularize the noise-covariance
  matrix.

- Added correct handling of the rank-deficient covariance matrix
  resulting from the use of SSS.

- Additional projections cannot be specified if the noise covariance matrix
  was computed with projections on.

- Bad channels can be added only in special circumstances if
  the noise covariance matrix was computed with projections on.

mne_compute_raw_inverse
=======================

This utility is now documented in :ref:`computing_inverse`. The
utility mne_make_raw_inverse_operator has been removed from the software.

Time range settings
===================

The tools mne_compute_raw_inverse , mne_convert_mne_data ,
and mne_compute_mne no longer have command-line options to restrict
the time range of evoked data input.

mne_change_baselines
====================

It is now possible to process all data sets in a file at
once. All processed data are stored in a single output file.

New utilities
=============

mne_show_fiff
-------------

Replacement for the Neuromag utility show_fiff .
This utility conforms to the standard command-line option conventions
in MNE software.

mne_make_cor_set
----------------

Replaces the functionality of the Neuromag utility create_mri_set_simple to
create a fif format description file for the FreeSurfer MRI data.
This utility is called by the mne_setup_mri script.

mne_compensate_data
-------------------

This utility applies or removes CTF software gradient compensation
from evoked-response data.

mne_insert_4D_comp
------------------

This utility merges 4D Magnes compensation data from a text
file and the main helmet sensor data from a fif file and creates
a new fif file :ref:`mne_insert_4D_comp`.

mne_ctf_dig2fiff
----------------

This utility reads a text format Polhemus data file, transforms
the data into the Neuromag head coordinate system, and outputs the
data in fif or hpts format.

mne_kit2fiff
------------

The purpose of this new utility is to import data from the
KIT MEG system.

mne_make_derivations
--------------------

This new utility will take derivation data from a text file
and convert it to fif format for use with mne_browse_raw.

BEM mesh generation
===================

All information concerning BEM mesh generation has been moved
to :ref:`create_bem_model`. Utilities for BEM mesh generation using
FLASH images have been added.

Matlab toolbox
==============

The MNE Matlab toolbox has been significantly enhanced. New
features include:

- Basic routines for reading and writing
  fif files.

- High-level functions to read and write evoked-response fif
  data.

- High-level functions to read raw data.

- High-level routines to read source space information, covariance
  matrices, forward solutions, and inverse operator decompositions
  directly from fif files.

The Matlab toolbox is documented in :ref:`ch_matlab`.

The mne_div_w utility
has been removed because it is now easy to perform its function
and much more using the Matlab Toolbox.

Release notes for MNE software 2.6
##################################

Manual
======

The changes described below briefly are documented in the
relevant sections of the manual. Change bars are employed to indicate
changes with respect to manual version 2.5. :ref:`ch_forward` now
contains a comprehensive discussion of the various coordinate systems
used in MEG/EEG data.

Command-line options
====================

All compiled C programs now check that the command line does
not contain any unknown options. Consequently, scripts that have
inadvertently specified some options which are invalid will now
fail.

Changes to existing software
============================

mne_add_patch_info
------------------

- Changed option ``--in`` to ``--src`` and ``--out`` to ``--srcp`` .

- Added ``--labeldir`` option.

mne_analyze
-----------

New features include:

- The name of the digital trigger channel
  can be specified with the MNE_TRIGGER_CH_NAME environment variable.

- Using information from the fif data files, the wall clock
  time corresponding to the current file position is shown on the
  status line

- mne_analyze can now be
  controlled by mne_browse_raw to
  facilitate interactive analysis of clinical data.

- Added compatibility with Elekta-Neuromag Report Composer (cliplab and
  improved the quality of hardcopies.

- Both in mne_browse_raw and
  in mne_analyze , a non-standard
  default layout can be set on a user-by-user basis.

- Added the ``--digtrigmask`` option.

- Added new image rotation functionality using the mouse wheel
  or trackball.

- Added remote control of the FreeSurfer MRI
  viewer (tkmedit ).

- Added fitting of single equivalent current dipoles and channel
  selections.

- Added loading of FreeSurfer cortical
  parcellation data as labels.

- Added support for using the FreeSurfer average
  brain (fsaverage) as a surrogate.

- The surface selection dialog was redesigned for faster access
  to the files and to remove problems with a large number of subjects.

- A shortcut button to direct a file selector to the appropriate
  default directory was added to several file loading dialogs.

- The vertex coordinates can now be displayed.

mne_average_forward_solutions
-----------------------------

EEG forward solutions are now averaged as well.

mne_browse_raw and mne_process_raw
----------------------------------

Improvements in the raw data processor mne_browse_raw /mne_process_raw include:

- The name of the digital trigger channel
  can be specified with the MNE_TRIGGER_CH_NAME environment variable.

- The format of the text event files was slightly changed. The
  sample numbers are now "absolute" sample numbers
  taking into account the initial skip in the event files. The new
  format is indicated by an additional "pseudoevent" in
  the beginning of the file. mne_browse_raw and mne_process_raw are
  still compatible with the old event file format.

- Using information from the fif data files, the wall clock
  time corresponding to the current file position is shown on the
  status line

- mne_browse_raw can now
  control mne_analyze to facilitate
  interactive analysis of clinical data.

- If the length of an output raw data file exceeds the 2-Gbyte
  fif file size limit, the output is split into multiple files.

- ``-split`` and ``--events`` options was
  added to mne_process_raw .

- The ``--allowmaxshield`` option was added to mne_browse_raw to allow
  loading of unprocessed data with MaxShield in the Elekta-Neuromag
  systems. These kind of data should never be used as an input for source
  localization.

- The ``--savehere`` option was added.

- The stderr parameter was
  added to the averaging definition files.

- Added compatibility with Elekta-Neuromag Report Composer (cliplab and
  improved the quality of hardcopies.

- Both in mne_browse_raw and
  in mne_analyze , a non-standard
  default layout can be set on a user-by-user basis.

- mne_browse_raw now includes
  an interactive editor to create derived channels.

- The menus in mne_browse_raw were
  reorganized and an time point specification text field was added

- Possibility to keep the old projection items added to the
  new projection definition dialog.

- Added ``--cd`` option.

- Added filter buttons for raw files and Maxfilter (TM) output
  to the open dialog.

- Added possibility to create a graph-compatible projection
  to the Save projection dialog

- Added possibility to compute a projection operator from epochs
  specified by events.

- Added the ``--keepsamplemean`` option
  to the covariance matrix computation files.

- Added the ``--digtrigmask`` option.

- Added Load channel selections... item
  to the File menu.

- Added new browsing functionality using the mouse wheel or
  trackball.

- Added optional items to the topographical data displays.

- Added an event list window.

- Added an annotator window.

- Keep events sorted by time.

- User-defined events are automatically kept in a fif-format
  annotation file.

- Added the delay parameter
  to the averaging and covariance matrix estimation description files.

Detailed information on these changes can be found in :ref:`ch_browse`.

mne_compute_raw_inverse
-----------------------

The ``--digtrig`` , ``--extra`` , ``--noextra`` , ``--split`` , ``--labeldir`` , and ``--out`` options
were added.

mne_convert_surface
-------------------

The functionality of mne_convert_dfs was
integrated into mne_convert_surface .
Text output as a triangle file and and file file containing the
list of vertex points was added. The Matlab output option was removed.
Consequently,  mne_convert_dfs , mne_surface2mat ,
and mne_list_surface_nodes were
deleted from the distribution.

mne_dump_triggers
-----------------

This obsolete utility was deleted from the distribution.

mne_epochs2mat
--------------

The name of the digital trigger channel can be specified
with the MNE_TRIGGER_CH_NAME environment variable. Added
the ``--digtrigmask`` option.

mne_forward_solution
--------------------

Added code to compute the derivatives of with respect to
the dipole position coordinates.

mne_list_bem
------------

The ``--surfno`` option is replaced with the ``--id`` option.

mne_make_cor_set
----------------

Include data from mgh/mgz files to the output automatically.
Include the Talairach transformations from the FreeSurfer data to
the output file if possible.

mne_make_movie
--------------

Added the ``--noscalebar``, ``--nocomments``, ``--morphgrade``, ``--rate``,
and ``--pickrange`` options.

mne_make_source_space
---------------------

The ``--spacing`` option is now implemented in this
program, which means mne_mris_trix is
now obsolete. The mne_setup_source_space script
was modified accordingly. Support for tri, dec, and dip files was dropped.

mne_mdip2stc
------------

This utility is obsolete and was removed from the distribution.

mne_project_raw
---------------

This is utility is obsolete and was removed from the distribution.
The functionality is included in mne_process_raw .

mne_rename_channels
-------------------

Added the ``--revert`` option.

mne_setup_forward_model
-----------------------

Added the ``--outershift`` and ``--scalpshift`` options.

mne_simu
--------

Added source waveform expressions and the ``--raw`` option.

mne_transform_points
--------------------

Removed the ``--tomrivol`` option.

Matlab toolbox
--------------

Several new functions were added, see :ref:`ch_matlab`.

.. note:: The matlab function fiff_setup_read_raw has    a significant change. The sample numbers now take into account possible    initial skip in the file, *i.e.*, the time between    the start of the data acquisition and the start of saving the data    to disk. The first_samp member    of the returned structure indicates the initial skip in samples.    If you want your own routines, which assume that initial skip has    been removed, perform indentically with the previous version, subtract first_samp from    the sample numbers you specify to fiff_read_raw_segment .    Furthermore, fiff_setup_read_raw has    an optional argument to allow reading of unprocessed MaxShield data acquired    with the Elekta MEG systems.

New utilities
=============

mne_collect_transforms
----------------------

This utility collects coordinate transformation information
from several sources into a single file.

mne_convert_dig_data
--------------------

This new utility convertes digitization (Polhemus) data between
different file formats.

mne_edf2fiff
------------

This is a new utility to convert EEG data from EDF, EDF+,
and BDF formats to the fif format.

mne_brain_vision2fiff
---------------------

This is a new utility to convert BrainVision EEG data to
the fif format. This utility is also
used by the mne_eximia_2fiff script
to convert EEG data from the Nexstim eXimia EEG system to the fif
format.

mne_anonymize
-------------

New utility to remove subject identifying information from
measurement files.

mne_opengl_test
---------------

New utility for testing the OpenGL graphics performance.

mne_volume_data2mri
-------------------

Convert data defined in a volume created with mne_volume_source_space to
an MRI overlay.

mne_volume_source_space
-----------------------

Create a a grid of source points within a volume. mne_volume_source_space also
optionally creates a trilinear interpolator matrix to facilitate
converting values a distribution in the volume grid into an MRI
overlay using mne_volume_data2mri.

mne_copy_processing_history
---------------------------

This new utility copies the processing history block from
one data file to another.

Release notes for MNE software 2.7
##################################

Software engineering
====================

There have been two significant changes in the software engineering
since MNE Version 2.6:

- CMake is now used in building the software
  package and

- Subversion (SVN) is now used for revision control instead
  of Concurrent Versions System (CVS).

These changes have the effects on the distribution of the
MNE software and setup for individual users:

- There is now a separate software package
  for each of the platforms supported.

- The software is now organized completely under standard directories (bin,
  lib, and share). In particular, the directory setup/mne has been moved
  to share/mne and the directories app-defaults and doc are now under
  share. All files under share are platform independent.

- The use of shared libraries has been minimized. This alleviates
  compatibility problems across operating system versions.

- The setup scripts have changed.

The installation and user-level effects of the new software
organization are discussed in :ref:`install_mne_c`.

In addition, several minor bugs have been fixed in the source
code. Most relevant changes visible to the user are listed below.

Matlab tools
============

- The performance of the fiff I/O routines
  has been significantly improved thanks to the contributions of François
  Tadel at USC.

- Label file I/O routines mne_read_label_file and mne_write_label_file as
  well as a routine to extract time courses corresponding to a label from
  an stc file (mne_label_time_courses) have been added.

- The patch information is now read from the source space file
  and included in the source space data structure.

mne_browse_raw
==============

- Rejection criteria to detect flat channels
  have been added.

- Possibility to detect temporal skew between trigger input
  lines has been added.

- ``--allowmaxshield`` option now works in the batch mode as well.

- Added the ``--projevent`` option to batch mode.

- It is now possible to compute an SSP operator for EEG.

mne_analyze
===========

- Both hemispheres can now be displayed
  simultaneously.

- If the source space was created with mne_make_source_space version 2.3
  or later, the subject's surface data are automatically
  loaded after loading the data and the inverse operator.

Miscellaneous
=============

- mne_smooth_w was
  renamed to mne_smooth and can
  now handle both w and stc files. Say ``mne_smooth --help`` to
  find the options.

- All binaries now reside in $MNE_ROOT/bin. There are no separate bin/mne
  and bin/admin directories.

- mne_anonymize now has the
  ``--his`` option to remove the HIS ID of the subject.

- mne_check_surface now has
  the ``--bem`` and ``--id`` options to check surfaces from a BEM fif file.

- mne_compute_raw_inverse now has the ``--orignames`` option.

- Added ``--headcoord`` option to mne_convert_dig_data.

- Added ``--talairach`` option to mne_make_cor_set.

- Added the ``--morph`` option to mne_setup_source_space and mne_make_source_space.

- Added the ``--prefix`` option to mne_morph_labels.

- Added the ``--blocks`` and ``--indent`` options to mne_show_fiff.

- Added the ``--proj`` option as well as map types 5 and 6 to mne_sensitivity_map.

- Fixed a bug in mne_inverse_operator which
  caused erroneous calculation of EEG-only source estimates if the
  data were processed with Maxfilter software and sometimes caused
  similar behavior on MEG/EEG source estimates.

Release notes for MNE software 2.7.1
####################################

mne_analyze
===========

- Added a new restricted mode for visualizing
  head position within the helmet.

- Added information about mne_make_scalp_surfaces to :ref:`CHDCGHIF`.

mne_browse_raw
==============

- Added possibility for multiple event
  parameters and the mask parameter in averaging and noise covariance
  calculation.

- Added simple conditional averaging.

Release notes for MNE software 2.7.2
####################################

mne_add_patch_info
==================

Added the capability to compute distances between source
space vertices.

Matlab toolbox
==============

- Added new functions to for stc and w
  file I/O to employ 1-based vertex numbering inside Matlab, see Table 10.11.

- mne_read_source_spaces.m now reads the inter-vertex distance
  information now optionally produced by mne_add_patch_info.

Miscellaneous
=============

- Added ``--shift`` option to mne_convert_surface.

- Added ``--alpha`` option to mne_make_movie.

- Added ``--noiserank`` option to mne_inverse_operator and mne_do_inverse_operator.

- The fif output from mne_convert_dig_data now
  includes the transformation between the digitizer and MNE head coordinate
  systems if such a transformation has been requested.
  This also affects the output from mne_eximia2fiff.

- Added ``--noflash30``, ``--noconvert``, and ``--unwarp`` options to mne_flash_bem.

Release notes for MNE software 2.7.3
####################################

Miscellaneous
=============

- Added preservation of the volume geometry
  information in the FreeSurfer surface files.

- The ``--mghmri`` option in combination with ``--surfout`` inserts
  the volume geometry information to the output of mne_convert_surface.

- Added ``--replacegeom`` option to mne_convert_surface.

- Modified mne_watershed_bem and mne_flash_bem to
  include the volume geometry information to the output. This allows
  viewing of the output surfaces in the FreeSurfer freeview utility.
