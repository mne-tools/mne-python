Miscellaneous C functionality
=============================

.. _BABCCEHF:

Setting up anatomical MR images for MRIlab
------------------------------------------

If you have the Neuromag software installed, the Neuromag
MRI viewer, MRIlab, can be used to access the MRI slice data created
by FreeSurfer . In addition, the
Neuromag MRI directories can be used for storing the MEG/MRI coordinate
transformations created with mne_analyze ,
see :ref:`CACEHGCD`.  During the computation of the forward
solution, mne_do_forwand_solution searches
for the MEG/MRI coordinate in the Neuromag MRI directories, see :ref:`BABCHEJD`. The fif files created by mne_setup_mri can
be loaded into Matlab with the fiff_read_mri function,
see :ref:`ch_matlab`.

These functions require running the script mne_setup_mri which
requires that the subject is set with the ``--subject`` option
or by the SUBJECT environment variable. The script processes one
or more MRI data sets from ``$SUBJECTS_DIR/$SUBJECT/mri`` ,
by default they are T1 and brain. This default can be changed by
specifying the sets by one or more ``--mri`` options.

The script creates the directories ``mri/`` <*name*> ``-neuromag/slices`` and ``mri/`` <*name*> ``-neuromag/sets`` .
If the input data set is in COR format, mne_setup_mri makes
symbolic links from the COR files in the directory ``mri/`` <*name*> into ``mri/`` <*name*> ``-neuromag/slices`` ,
and creates a corresponding fif file COR.fif in ``mri/`` <*name*> ``-neuromag/sets`` ..
This "description file" contains references to
the actual MRI slices.

If the input MRI data are stored in the newer mgz format,
the file created in the ``mri/`` <*name*> ``-neuromag/sets`` directory
will include the MRI pixel data as well. If available, the coordinate
transformations to allow conversion between the MRI (surface RAS)
coordinates and MNI and FreeSurfer Talairach coordinates are copied
to the MRI description file. mne_setup_mri invokes mne_make_cor_set ,
described in :ref:`mne_make_cor_set` to convert the data.

For example:

``mne_setup_mri --subject duck_donald --mri T1``

This command processes the MRI data set T1 for subject duck_donald.

.. note:: If the SUBJECT environment variable is set it    is usually sufficient to run mne_setup_mri without    any options.

.. note:: If the name specified with the ``--mri`` option    contains a slash, the MRI data are accessed from the directory specified    and the ``SUBJECT`` and ``SUBJECTS_DIR`` environment    variables as well as the ``--subject`` option are ignored.

MRIlab can also be used for coordinate frame alignment.
Section 3.3.1 of the MRIlab User's Guide,
Neuromag P/N NM20419A-A contains a detailed description of
this task. Employ the images in the set ``mri/T1-neuromag/sets/COR.fif`` for
the alignment. Check the alignment carefully using the digitization
data included in the measurement file as described in Section 5.3.1
of the above manual. Save the aligned description file in the same
directory as the original description file without the alignment
information but under a different name.


.. _BABCDBDI:

Cleaning the digital trigger channel
------------------------------------

The calibration factor of the digital trigger channel used
to be set to a value much smaller than one by the Neuromag data
acquisition software. Especially to facilitate viewing of raw data
in graph it is advisable to change the calibration factor to one.
Furthermore, the eighth bit of the trigger word is coded incorrectly
in the original raw files. Both problems can be corrected by saying:

``mne_fix_stim14`` <*raw file*>

More information about mne_fix_stim14 is
available in :ref:`mne_fix_stim14`. It is recommended that this
fix is included as the first raw data processing step. Note, however,
the mne_browse_raw and mne_process_raw always sets
the calibration factor to one internally.

.. note:: If your data file was acquired on or after November 10, 2005 on the Martinos center Vectorview system, it is not necessary to use mne_fix_stim14 .

.. _BABCDFJH:

Fixing channel information
--------------------------

There are two potential discrepancies in the channel information
which need to be fixed before proceeding:

- EEG electrode locations may be incorrect
  if more than 60 EEG channels are acquired.

- The magnetometer coil identifiers are not always correct.

These potential problems can be fixed with the utilities mne_check_eeg_locations and mne_fix_mag_coil_types,
see :ref:`mne_check_eeg_locations` and :ref:`mne_fix_mag_coil_types`.
