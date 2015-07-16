

.. _ch_cookbook:

==================
The MNE-C Cookbook
==================

.. contents:: Contents
   :local:
   :depth: 2

Selecting the subject
#####################

Before starting the data analysis, setup the environment
variable SUBJECTS_DIR to select the directory under which the anatomical
MRI data are stored. Optionally, set SUBJECT as the name of the
subject's MRI data directory under SUBJECTS_DIR. With this
setting you can avoid entering the ``--subject`` option common to many
MNE programs and scripts. In the following sections, files in the
FreeSurfer directory hierarchy are usually referred to without specifying
the leading directories. Thus, bem/msh-7-src.fif is used to refer
to the file $SUBJECTS_DIR/$SUBJECT/bem/msh-7-src.fif.

It is also recommended that the FreeSurfer environment
is set up before using the MNE software.

.. _CHDBBCEJ:

Cortical surface reconstruction with FreeSurfer
###############################################

The first processing stage is the creation of various surface
reconstructions with FreeSurfer .
The recommended FreeSurfer workflow
is summarized on the FreeSurfer wiki pages: https://surfer.nmr.mgh.harvard.edu/fswiki/RecommendedReconstruction.
Please refer to the FreeSurfer wiki pages
(https://surfer.nmr.mgh.harvard.edu/fswiki/) and other FreeSurfer documentation
for more information.

.. note:: Only the latest (4.0.X and later) FreeSurfer distributions    contain a version of tkmedit which    is compatible with mne_analyze, see :ref:`CACCHCBF`.

.. _BABCCEHF:

Setting up the anatomical MR images for MRIlab
##############################################

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
described in :ref:`BABBHHHE` to convert the data.

For example:

``mne_setup_mri --subject duck_donald --mri T1``

This command processes the MRI data set T1 for subject duck_donald.

.. note:: If the SUBJECT environment variable is set it    is usually sufficient to run mne_setup_mri without    any options.

.. note:: If the name specified with the ``--mri`` option    contains a slash, the MRI data are accessed from the directory specified    and the ``SUBJECT`` and ``SUBJECTS_DIR`` environment    variables as well as the ``--subject`` option are ignored.

.. _CIHCHDAE:

Setting up the source space
###########################

This stage consists of the following:

- Creating a suitable decimated dipole
  grid on the white matter surface.

- Creating the source space file in fif format.

- Creating ascii versions of the source space file for viewing
  with MRIlab.

All of the above is accomplished with the convenience script :ref:`mne_setup_source_space`. This
script assumes that:

- The anatomical MRI processing has been
  completed as described in :ref:`CHDBBCEJ`.

- The environment variable SUBJECTS_DIR is set correctly.

See :ref:`mne_setup_source_space` for command-line options.

.. _BABGCDHA:

.. table:: Recommended subdivisions of an icosahedron and an octahedron for the creation of source spaces. The approximate source spacing and corresponding surface area have been calculated assuming a 1000-cm2 surface area per hemisphere.

    ==========  ========================  =====================  ===============================
    <*number*>  Sources per hemisphere    Source spacing / mm    Surface area per source / mm2
    ==========  ========================  =====================  ===============================
    -5          1026                      9.9                    97
    4           2562                      6.2                    39
    -6          4098                      4.9                    24
    5           10242                     3.1                    9.8
    ==========  ========================  =====================  ===============================

For example, to create the reconstruction geometry for Donald
Duck with a 5-mm spacing between the grid points, say

``mne_setup_source_space --subject duck_donald --spacing 5``

As a result, the following files are created into the ``bem`` directory:

- <*subject*>-<*spacing*>- ``src.fif`` containing
  the source space description in fif format.

- <*subject*>-<*spacing*>- ``lh.pnt`` and <*subject*>-<*spacing*>- ``rh.pnt`` containing
  the source space points in MRIlab compatible ascii format.

- <*subject*>-<*spacing*>- ``lh.dip`` and <*subject*>-<*spacing*>- ``rh.dip`` containing
  the source space points in MRIlab compatible ascii format. These
  files contain 'dipoles', *i.e.*,
  both source space points and cortex normal directions.

- If cortical patch statistics is requested, another source
  space file called <*subject*>-<*spacing*> ``p-src.fif`` will
  be created.

.. note:: <*spacing*> will    be the suggested source spacing in millimeters if the ``--spacing`` option    is used. For source spaces based on *k*th subdivision    of an icosahedron, <*spacing*> will    be replaced by ``ico-`` k or ``oct-`` k , respectively.

.. note:: After the geometry is set up it is possible to    check that the source space points are located on the cortical surface.    This can be easily done with by loading the ``COR.fif`` file    from ``mri/T1/neuromag/sets`` into MRIlab and by subsequently    overlaying the corresponding pnt or dip files using Import/Strings or Import/Dipoles from    the File menu, respectively.

.. note:: If the SUBJECT environment variable is set correctly    it is usually sufficient to run ``mne_setup_source_space`` without    any options.

.. _CHDBJCIA:

Creating the BEM model meshes
#############################

Calculation of the forward solution using the boundary-element
model (BEM) requires that the surfaces separating regions of different
electrical conductivities are tessellated with suitable surface
elements. Our BEM software employs triangular tessellations. Therefore,
prerequisites for BEM calculations are the segmentation of the MRI
data and the triangulation of the relevant surfaces.

For MEG computations, a reasonably accurate solution can
be obtained by using a single-compartment BEM assuming the shape
of the intracranial volume. For EEG, the standard model contains
the intracranial space, the skull, and the scalp.

At present, no bulletproof method exists for creating the
triangulations. Feasible approaches are described in :ref:`create_bem_model`.

.. _BABDBBFC:

Setting up the triangulation files
==================================

The segmentation algorithms described in :ref:`create_bem_model` produce
either FreeSurfer surfaces or triangulation
data in text. Before proceeding to the creation of the boundary
element model, standard files (or symbolic links created with the ``ln -s`` command) have to be present in the subject's ``bem`` directory.
If you are employing ASCII triangle files the standard file names
are:

**inner_skull.tri**

    Contains the inner skull triangulation.

**outer_skull.tri**

    Contains the outer skull triangulation.

**outer_skin.tri**

    Contains the head surface triangulation.

The corresponding names for FreeSurfer surfaces
are:

**inner_skull.surf**

    Contains the inner skull triangulation.

**outer_skull.surf**

    Contains the outer skull triangulation.

**outer_skin.surf**

    Contains the head surface triangulation.

.. note:: Different methods can be employed for the creation    of the individual surfaces. For example, it may turn out that the    watershed algorithm produces are better quality skin surface than    the segmentation approach based on the FLASH images. If this is    the case, ``outer_skin.surf`` can set to point to the corresponding    watershed output file while the other surfaces can be picked from    the FLASH segmentation data.

.. note:: The triangulation files can include name of the    subject as a prefix ``<*subject name*>-`` , *e.g.*, ``duck-inner_skull.surf`` .

.. note:: The mne_convert_surface utility    described in :ref:`BEHDIAJG` can be used to convert text format    triangulation files into the FreeSurfer surface format.

.. note:: "Aliases" created with    the Mac OSX finder are not equivalent to symbolic links and do not    work as such for the UNIX shells and MNE programs.

.. _CIHDBFEG:

Setting up the boundary-element model
#####################################

This stage sets up the subject-dependent data for computing
the forward solutions:

- The fif format boundary-element model
  geometry file is created. This step also checks that the input surfaces
  are complete and that they are topologically correct, *i.e.*,
  that the surfaces do not intersect and that the surfaces are correctly
  ordered (outer skull surface inside the scalp and inner skull surface
  inside the outer skull). Furthermore, the range of triangle sizes
  on each surface is reported. For the three-layer model, the minimum
  distance between the surfaces is also computed.

- Text files containing the boundary surface vertex coordinates are
  created.

- The the geometry-dependent BEM solution data are computed. This step
  can be optionally omitted. This step takes several minutes to complete.

This step assigns the conductivity values to the BEM compartments.
For the scalp and the brain compartments, the default is 0.3 S/m.
The default skull conductivity is 50 times smaller, *i.e.*,
0.006 S/m. Recent publications, see :ref:`CEGEGDEI`, report
a range of skull conductivity ratios ranging from 1:15 (Oostendorp *et
al.*, 2000) to 1:25 - 1:50 (Slew *et al.*,
2009, Conçalves *et al.*, 2003). The
MNE default ratio 1:50 is based on the typical values reported in
(Conçalves *et al.*, 2003), since their
approach is based comparison of SEF/SEP measurements in a BEM model.
The variability across publications may depend on individual variations
but, more importantly, on the precision of the skull compartment
segmentation.

This processing stage is automated with the script mne_setup_forward_model . This
script assumes that:

- The anatomical MRI processing has been
  completed as described in :ref:`CHDBBCEJ`.

- The BEM model meshes have been created as outlined in :ref:`CHDBJCIA`.

- The environment variable SUBJECTS_DIR is set correctly.

See :ref:`mne_setup_forward_model` for command-line options.

As a result of running the :ref:`mne_setup_foward_model` script, the
following files are created into the ``bem`` directory:

- BEM model geometry specifications <*subject*>-<*ntri-scalp*>-<*ntri-outer_skull*>-<*ntri-inner_skull*>- ``bem.fif`` or <*subject*>-<*ntri-inner_skull*> ``-bem.fif`` containing
  the BEM geometry in fif format. The latter file is created if ``--homog``
  option is specified. Here, <*ntri-xxx*> indicates
  the number of triangles on the corresponding surface.

- <*subject*>-<*surface name*>-<*ntri*> ``.pnt`` files
  are created for each of the surfaces present in the BEM model. These
  can be loaded to MRIlab to check the location of the surfaces.

- <*subject*>-<*surface name*>-<*ntri*> ``.surf`` files
  are created for each of the surfaces present in the BEM model. These
  can be loaded to tkmedit to check
  the location of the surfaces.

- The BEM 'solution' file containing the geometry
  dependent solution data will be produced with the same name as the
  BEM geometry specifications with the ending ``-bem-sol.fif`` .
  These files also contain all the information in the ``-bem.fif`` files.

After the BEM is set up it is advisable to check that the
BEM model meshes are correctly positioned. This can be easily done
with by loading the COR.fif file
from mri/T1-neuromag/sets into
MRIlab and by subsequently overlaying the corresponding pnt files
using Import/Strings from the File menu.

.. note:: The FreeSurfer format    BEM surfaces can be also viewed with the tkmedit program    which is part of the FreeSurfer distribution.

.. note:: If the SUBJECT environment variable is set, it    is usually sufficient to run ``mne_setup_forward_model`` without    any options for the three-layer model and with the ``--homog`` option    for the single-layer model. If the input files are FreeSurfer surfaces, ``--surf`` and ``--ico 4`` are required as well.

.. note:: With help of the ``--nosol`` option    it is possible to create candidate BEM geometry data files quickly    and do the checking with respect to the anatomical MRI data. When    the result is satisfactory, mne_setup_forward_model can be run without ``--nosol`` to    invoke the time-consuming calculation of the solution file as well.

.. note:: The triangle meshes created by the seglab program    have counterclockwise vertex ordering and thus require the ``--noswap``    option.

.. note:: Up to this point all processing stages depend    on the anatomical (geometrical) information only and thus remain    identical across different MEG studies.

Setting up the MEG/EEG analysis directory
#########################################

The remaining steps require that the actual MEG/EEG data
are available. It is recommended that a new directory is created
for the MEG/EEG data processing. The raw data files collected should not be
copied there but rather referred to with symbolic links created
with the ``ln -s`` command. Averages calculated
on-line can be either copied or referred to with links.

.. note:: If you don't know how to create a directory,    how to make symbolic links, or how to copy files from the shell    command line, this is a perfect time to learn about this basic skills    from other users or from a suitable elementary book before proceeding.

Preprocessing the raw data
##########################

The following MEG and EEG data preprocessing steps are recommended:

- The coding problems on the trigger channel
  STI 014 may have to fixed, see :ref:`BABCDBDI`.

- EEG electrode location information and MEG coil types may
  need to be fixed, see :ref:`BABCDFJH`.

- The data may be optionally downsampled to facilitate subsequent
  processing, see :ref:`BABDGFFG`.

- Bad channels in the MEG and EEG data must be identified, see :ref:`BABBHCFG`.

- The data has to be filtered to the desired passband. If mne_browse_raw or mne_process_raw is
  employed to calculate the offline averages and covariance matrices,
  this step is unnecessary since the data are filtered on the fly.
  For information on these programs, please consult :ref:`ch_browse`.

- For evoked-response analysis, the data has to be re-averaged
  off line, see :ref:`BABEAEDF`.

.. _BABCDBDI:

Cleaning the digital trigger channel
====================================

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
==========================

There are two potential discrepancies in the channel information
which need to be fixed before proceeding:

- EEG electrode locations may be incorrect
  if more than 60 EEG channels are acquired.

- The magnetometer coil identifiers are not always correct.

These potential problems can be fixed with the utilities mne_check_eeg_locations and mne_fix_mag_coil_types,
see :ref:`mne_check_eeg_locations` and :ref:`mne_fix_mag_coil_types`.

.. _BABBHCFG:

Designating bad channels
========================

Sometimes some MEG or EEG channels are not functioning properly
for various reasons. These channels should be excluded from the
analysis by marking them bad using the mne_mark_bad_channels utility,
see :ref:`mne_mark_bad_channels`. Especially if a channel does not show
a signal at all (flat) it is most important to exclude it from the
analysis, since its noise estimate will be unrealistically low and
thus the current estimate calculations will give a strong weight
to the zero signal on the flat channels and will essentially vanish.
It is also important to exclude noisy channels because they can
possibly affect others when signal-space projections or EEG average electrode
reference is employed. Noisy bad channels can also adversely affect
off-line averaging and noise-covariance matrix estimation by causing
unnecessary rejections of epochs.

Recommended ways to identify bad channels are:

- Observe the quality of data during data
  acquisition and make notes of observed malfunctioning channels to
  your measurement protocol sheet.

- View the on-line averages and check the condition of the channels.

- Compute preliminary off-line averages with artefact rejection,
  signal-space projection, and EEG average electrode reference computation
  off and check the condition of the channels.

- View raw data in mne_process_raw or
  the Neuromag signal processor graph without
  signal-space projection or EEG average electrode reference computation
  and identify bad channels.

.. note:: It is strongly recommended that bad channels    are identified and marked in the original raw data files. If present    in the raw data files, the bad channel selections will be automatically    transferred to averaged files, noise-covariance matrices, forward    solution files, and inverse operator decompositions.

.. _BABDGFFG:

Downsampling the MEG/EEG data
=============================

The minimum practical sampling frequency of the Vectorview
system is 600 Hz. Lower sampling frequencies are allowed but result
in elevated noise level in the data. It is advisable to lowpass
filter and downsample the large raw data files often emerging in
cognitive and patient studies to speed up subsequent processing.
This can be accomplished with the mne_process_raw and mne_browse_raw software
modules. For details, see :ref:`CACFAAAJ` and :ref:`CACBDDIE`.

.. note:: It is recommended that the original raw file    is called <*name*>_raw.fif and    the downsampled version <*name*>_ds_raw.fif ,    respectively.

.. _BABEAEDF:

Off-line averaging
==================

The recommended tools for off-line averaging are mne_browse_raw and mne_process_raw . mne_browse_raw is
an interactive program for averaging and noise-covariance matrix
computation. It also includes routines for filtering so that the
downsampling and filtering steps can be skipped. Therefore, with mne_browse_raw you
can produce the off-line average and noise-covariance matrix estimates
directly. The batch-mode version of mne_browse_raw is
called mne_process_raw . Detailed
information on mne_browse_raw and mne_process_raw can
be found in :ref:`ch_browse`.

.. _CHDBEHDC:

Aligning the coordinate frames
##############################

The calculation of the forward solution requires knowledge
of the relative location and orientation of the MEG/EEG and MRI
coordinate systems. The MEG/EEG head coordinate system is defined
in :ref:`BJEBIBAI`. The conversion tools included in the MNE
software take care of the idiosyncrasies of the coordinate frame
definitions in different MEG and EEG systems so that the fif files
always employ the same definition of the head coordinate system.

Ideally, the head coordinate frame has a fixed orientation
and origin with respect to the head anatomy. Therefore, a single
MRI-head coordinate transformation for each subject should be sufficient.
However, as explained in :ref:`BJEBIBAI`, the head coordinate
frame is defined by identifying the fiducial landmark locations,
making the origin and orientation of the head coordinate system
slightly user dependent. As a result, the most conservative choice
for the definition of the coordinate transformation computation
is to re-establish it for each experimental session, *i.e.*,
each time when new head digitization data are employed.

The interactive source analysis software mne_analyze provides
tools for coordinate frame alignment, see :ref:`ch_interactive_analysis`. :ref:`CHDIJBIG` also
contains tips for using mne_analyze for
this purpose.

Another useful tool for the coordinate system alignment is MRIlab ,
the Neuromag MEG-MRI integration tool. Section 3.3.1 of the MRIlab User's
Guide, Neuromag P/N NM20419A-A contains a detailed description of
this task. Employ the images in the set ``mri/T1-neuromag/sets/COR.fif`` for
the alignment. Check the alignment carefully using the digitization
data included in the measurement file as described in Section 5.3.1
of the above manual. Save the aligned description file in the same
directory as the original description file without the alignment
information but under a different name.

.. warning:: This step is extremely important. If    the alignment of the coordinate frames is inaccurate all subsequent    processing steps suffer from the error. Therefore, this step should    be performed by the person in charge of the study or by a trained    technician. Written or photographic documentation of the alignment    points employed during the MEG/EEG acquisition can also be helpful.

.. _BABCHEJD:

Computing the forward solution
##############################

After the MRI-MEG/EEG alignment has been set, the forward
solution, *i.e.*, the magnetic fields and electric
potentials at the measurement sensors and electrodes due to dipole
sources located on the cortex, can be calculated with help of the
convenience script mne_do_forward_solution. See :ref:`mne_do_forward_solution`
for command-line options.


.. _BABDEEEB:

Setting up the noise-covariance matrix
######################################

The MNE software employs an estimate of the noise-covariance
matrix to weight the channels correctly in the calculations. The
noise-covariance matrix provides information about field and potential
patterns representing uninteresting noise sources of either human
or environmental origin.

The noise covariance matrix can be calculated in several
ways:

- Employ the individual epochs during
  off-line averaging to calculate the full noise covariance matrix.
  This is the recommended approach for evoked responses.

- Employ empty room data (collected without the subject) to
  calculate the full noise covariance matrix. This is recommended
  for analyzing ongoing spontaneous activity.

- Employ a section of continuous raw data collected in the presence
  of the subject to calculate the full noise covariance matrix. This
  is the recommended approach for analyzing epileptic activity. The
  data used for this purpose should be free of technical artifacts
  and epileptic activity of interest. The length of the data segment
  employed should be at least 20 seconds. One can also use a long
  (`*> 200 s`) segment of data with epileptic spikes present provided
  that the spikes occur infrequently and that the segment is apparently
  stationary with respect to background brain activity.

The new raw data processing tools, mne_browse_raw or mne_process_raw include
computation of noise-covariance matrices both from raw data and
from individual epochs. For details, see :ref:`ch_browse`.

.. _CIHCFJEI:

Calculating the inverse operator decomposition
##############################################

The MNE software doesn't calculate the inverse operator
explicitly but rather computes an SVD of a matrix composed of the
noise-covariance matrix, the result of the forward calculation,
and the source covariance matrix. This approach has the benefit
that the regularization parameter ('SNR') can
be adjusted easily when the final source estimates or dSPMs are
computed. For mathematical details of this approach, please consult :ref:`CBBDJFBJ`.

This computation stage is facilitated by the convenience
script mne_do_inverse_operator . It
invokes the program mne_inverse_operator with
appropriate options, derived from the command line of mne_do_inverse_operator .

See :ref:`mne_do_inverse_operator` for command-line options.


Analyzing the data
##################

Once all the preprocessing steps described above have been
completed, the inverse operator computed can be applied to the MEG
and EEG data and the results can be viewed and stored in several
ways:

- The interactive analysis tool mne_analyze can
  be used to explore the data and to produce quantitative analysis
  results, screen snapshots, and QuickTime (TM) movie files.
  For comprehensive information on mne_analyze ,
  please consult :ref:`ch_interactive_analysis`.

- The command-line tool mne_make_movie can
  be invoked to produce QuickTime movies and snapshots. mne_make_movie can
  also output the data in the stc (movies) and w (snapshots) formats
  for subsequent processing. Furthermore, subject-to-subject morphing
  is included in mne_make_movie to
  facilitate cross-subject averaging and comparison of data among
  subjects. mne_make_movie is described
  in :ref:`CBBECEDE`.

- The command-line tool mne_make_movie can
  be employed to interrogate the source estimate waveforms from labels
  (ROIs).

- The mne_make_movie tool
  can be also used to create movies from stc files and to resample
  stc files in time.

- The mne_compute_raw_inverse tool
  can be used to produce fif files containing source estimates at
  selected ROIs. The input data file can be either a raw data or evoked
  response MEG/EEG file, see :ref:`CBBCGHAH`.

- Using the MNE Matlab toolbox, it is possible to perform many
  of the above operations in Matlab using your own Matlab code based
  on the MNE Matlab toolbox. For more information on the MNE Matlab
  toolbox, see :ref:`ch_matlab`.

- It is also possible to average the source estimates across
  subjects as described in :ref:`ch_morph`.
