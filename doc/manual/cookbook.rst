

.. _ch_cookbook:

============
The Cookbook
============

.. contents:: Contents
   :local:
   :depth: 2

Overview
########

This section describes a typical MEG/EEG workflow, eventually up to source
reconstruction. The workflow is summarized in :ref:`flow_diagram`.

.. _flow_diagram:

.. figure:: ../_static/flow_diagram.svg
    :alt: MNE Workflow Flowchart
    :align: center

    Workflow of the MNE software

    References below refer to Python functions and objects.

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
for the MEG/MRI coordinate in the Neuromag MRI directories, see :ref:`BABCHEJD`. The fif files created by mne_setup_mrit can
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

All of the above is accomplished with the convenience script mne_setup_source_space . This
script assumes that:

- The anatomical MRI processing has been
  completed as described in :ref:`CHDBBCEJ`.

- The environment variable SUBJECTS_DIR is set correctly.

The script accepts the following options:

**\---subject <*subject*>**

    Defines the name of the subject. If the environment variable SUBJECT
    is set correctly, this option is not required.

**\---morph <*name*>**

    Name of a subject in SUBJECTS_DIR. If this option is present, the source
    space will be first constructed for the subject defined by the --subject
    option or the SUBJECT environment variable and then morphed to this
    subject. This option is useful if you want to create a source spaces
    for several subjects and want to directly compare the data across
    subjects at the source space vertices without any morphing procedure
    afterwards. The drawback of this approach is that the spacing between
    source locations in the "morph" subject is not going
    to be as uniform as it would be without morphing.

**\---spacing <*spacing/mm*>**

    Specifies the grid spacing for the source space in mm. If not set,
    a default spacing of 7 mm is used. Either the default or a 5-mm
    spacing is recommended.

**\---ico <*number*>**

    Instead of using the traditional method for cortical surface decimation
    it is possible to create the source space using the topology of
    a recursively subdivided icosahedron (<*number*> > 0)
    or an octahedron (<*number*> < 0).
    This method uses the cortical surface inflated to a sphere as a
    tool to find the appropriate vertices for the source space. The
    benefit of the ``--ico`` option is that the source space
    will have triangulation information for the decimated vertices included, which
    future versions of MNE software may be able to utilize. The number
    of triangles increases by a factor of four in each subdivision,
    starting from 20 triangles in an icosahedron and 8 triangles in an
    octahedron. Since the number of vertices on a closed surface is :math:`n_{vert} = (n_{tri} + 4)/2`,
    the number of vertices in the *k* th subdivision of
    an icosahedron and an octahedron are :math:`10 \cdot 4^k + 2` and :math:`4^{k + 1} + 2`, respectively.
    The recommended values for <*number*> and
    the corresponding number of source space locations are listed in :ref:`BABGCDHA`.

**\---surface <*name*>**

    Name of the surface under the surf directory to be used. Defaults
    to 'white'. ``mne_setup_source_space`` looks
    for files ``rh.`` <*name*> and ``lh.`` <*name*> under
    the ``surf`` directory.

**\---overwrite**

    An existing source space file with the same name is overwritten only
    if this option is specified.

**\---cps**

    Compute the cortical patch statistics. This is need if current-density estimates
    are computed, see :ref:`CBBDBHDI`. If the patch information is
    available in the source space file the surface normal is considered to
    be the average normal calculated over the patch instead of the normal
    at each source space location. The calculation of this information
    takes a considerable amount of time because of the large number
    of Dijkstra searches involved.

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

mne_setup_forward_model accepts
the following options:

**\---subject <*subject*>**

    Defines the name of the subject. This can be also accomplished
    by setting the SUBJECT environment variable.

**\---surf**

    Use the FreeSurfer surface files instead of the default ASCII triangulation
    files. Please consult :ref:`BABDBBFC` for the standard file
    naming scheme.

**\---noswap**

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

**\---ico <*number*>**

    This option is relevant (and required) only with the ``--surf`` option and
    if the surface files have been produced by the watershed algorithm.
    The watershed triangulations are isomorphic with an icosahedron,
    which has been recursively subdivided six times to yield 20480 triangles.
    However, this number of triangles results in a long computation
    time even in a workstation with generous amounts of memory. Therefore,
    the triangulations have to be decimated. Specifying ``--ico 4`` yields 5120 triangles per surface while ``--ico 3`` results
    in 1280 triangles. The recommended choice is ``--ico 4`` .

**\---homog**

    Use a single compartment model (brain only) instead a three layer one
    (scalp, skull, and brain). Only the ``inner_skull.tri`` triangulation
    is required. This model is usually sufficient for MEG but invalid
    for EEG. If you are employing MEG data only, this option is recommended
    because of faster computation times. If this flag is specified,
    the options ``--brainc`` , ``--skullc`` , and ``--scalpc`` are irrelevant.

**\---brainc <*conductivity/ S/m*>**

    Defines the brain compartment conductivity. The default value is 0.3 S/m.

**\---skullc <*conductivity/ S/m*>**

    Defines the skull compartment conductivity. The default value is 0.006 S/m
    corresponding to a conductivity ratio 1/50 between the brain and
    skull compartments.

**\---scalpc <*conductivity/ S/m*>**

    Defines the brain compartment conductivity. The default value is 0.3 S/m.

**\---innershift <*value/mm*>**

    Shift the inner skull surface outwards along the vertex normal directions
    by this amount.

**\---outershift <*value/mm*>**

    Shift the outer skull surface outwards along the vertex normal directions
    by this amount.

**\---scalpshift <*value/mm*>**

    Shift the scalp surface outwards along the vertex normal directions by
    this amount.

**\---nosol**

    Omit the BEM model geometry dependent data preparation step. This
    can be done later by running mne_setup_forward_model without the ``--nosol`` option.

**\---model <*name*>**

    Name for the BEM model geometry file. The model will be created into
    the directory bem as <*name*>- ``bem.fif`` .	If
    this option is missing, standard model names will be used (see below).

As a result of running the mne_setup_foward_model script, the
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
available in :ref:`CHDBFDIC`. It is recommended that this
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
see :ref:`CHDJGGGC` and :ref:`CHDGAAJC`.

.. _BABBHCFG:

Designating bad channels
========================

Sometimes some MEG or EEG channels are not functioning properly
for various reasons. These channels should be excluded from the
analysis by marking them bad using the mne_mark_bad_channels utility,
see :ref:`CHDDHBEE`. Especially if a channel does not show
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
convenience script mne_do_forward_solution .
This utility accepts the following options:

**\---subject <*subject*>**

    Defines the name of the subject. This can be also accomplished
    by setting the SUBJECT environment variable.

**\---src <*name*>**

    Source space name to use. This option overrides the ``--spacing`` option. The
    source space is searched first from the current working directory
    and then from ``$SUBJECTS_DIR/`` <*subject*> /bem.
    The source space file must be specified exactly, including the ``fif`` extension.

**\---spacing <*spacing/mm*>  or ``ico-`` <*number  or ``oct-`` <*number*>**

    This is an alternate way to specify the name of the source space
    file. For example, if ``--spacing 6`` is given on the command
    line, the source space files searched for are./<*subject*> -6-src.fif
    and ``$SUBJECTS_DIR/$SUBJECT/`` bem/<*subject*> -6-src.fif.
    The first file found is used. Spacing defaults to 7 mm.

**\---bem <*name*>**

    Specifies the BEM to be used. The name of the file can be any of <*name*> , <*name*> -bem.fif, <*name*> -bem-sol.fif.
    The file is searched for from the current working directory and
    from ``bem`` . If this option is omitted, the most recent
    BEM file in the ``bem`` directory is used.

**\---mri <*name*>**

    The name of the MRI description file containing the MEG/MRI coordinate
    transformation. This file was saved as part of the alignment procedure
    outlined in :ref:`CHDBEHDC`. The file is searched for from
    the current working directory and from ``mri/T1-neuromag/sets`` .
    The search order for MEG/MRI coordinate transformations is discussed
    below.

**\---trans	 <*name*>**

    The name of a text file containing the 4 x 4 matrix for the coordinate transformation
    from head to mri coordinates, see below. If the option ``--trans`` is
    present, the ``--mri`` option is not required. The search
    order for MEG/MRI coordinate transformations is discussed below.

**\---meas <*name*>**

    This file is the measurement fif file or an off-line average file
    produced thereof. It is recommended that the average file is employed for
    evoked-response data and the original raw data file otherwise. This
    file provides the MEG sensor locations and orientations as well as
    EEG electrode locations as well as the coordinate transformation between
    the MEG device coordinates and MEG head-based coordinates.

**\---fwd <*name*>**

    This file will contain the forward solution as well as the coordinate transformations,
    sensor and electrode location information, and the source space
    data. A name of the form <*name*> ``-fwd.fif`` is
    recommended. If this option is omitted the forward solution file
    name is automatically created from the measurement file name and
    the source space name.

**\---destdir <*directory*>**

    Optionally specifies a directory where the forward solution will
    be stored.

**\---mindist <*dist/mm*>**

    Omit source space points closer than this value to the inner skull surface.
    Any source space points outside the inner skull surface are automatically
    omitted. The use of this option ensures that numerical inaccuracies
    for very superficial sources do not cause unexpected effects in
    the final current estimates. Suitable value for this parameter is
    of the order of the size of the triangles on the inner skull surface.
    If you employ the seglab software
    to create the triangulations, this value should be about equal to
    the wish for the side length of the triangles.

**\---megonly**

    Omit EEG forward calculations.

**\---eegonly**

    Omit MEG forward calculations.

**\---all**

    Compute the forward solution for all vertices on the source space.

**\---overwrite**

    Overwrite the possibly existing forward model file.

**\---help**

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

mne_do_inverse_operator assumes
the following options:

**\---fwd <*name of the forward solution file*>**

    This is the forward solution file produced in the computations step described
    in :ref:`BABCHEJD`.

**\---meg**

    Employ MEG data in the inverse calculation. If neither ``--meg`` nor ``--eeg`` is
    set only MEG channels are included.

**\---eeg**

    Employ EEG data in the inverse calculation. If neither ``--meg`` nor ``--eeg`` is
    set only MEG channels are included.

**\---fixed**

    Use fixed source orientations normal to the cortical mantle. By default,
    the source orientations are not constrained. If ``--fixed`` is specified,
    the ``--loose`` flag is ignored.

**\---loose <*amount*>**

    Use a 'loose' orientation constraint. This means
    that the source covariance matrix entries corresponding to the current
    component normal to the cortex are set equal to one and the transverse
    components are set to <*amount*> .
    Recommended value of amount is 0.1...0.6.

**\---depth**

    Employ depth weighting with the standard settings. For details,
    see :ref:`CBBDFJIE` and :ref:`CBBDDBGF`.

**\---bad <*name*>**

    Specifies a text file to designate bad channels, listed one channel name
    (like MEG 1933) on each line of the file. Be sure to include both
    noisy and flat (non-functioning) channels in the list. If bad channels
    were designated using mne_mark_bad_channels in
    the measurement file which was specified with the ``--meas`` option when
    the forward solution was computed, the bad channel information will
    be automatically included. Also, any bad channel information in
    the noise-covariance matrix file will be included.

**\---noisecov <*name*>**

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

**\---noiserank <*value*>**

    Specifies the rank of the noise covariance matrix explicitly rather than
    trying to reduce it automatically. This option is sheldom needed,

**\---megreg <*value*>**

    Regularize the MEG part of the noise-covariance matrix by this amount.
    Suitable values are in the range 0.05...0.2. For details, see :ref:`CBBHEGAB`.

**\---eegreg <*value*>**

    Like ``--megreg`` but applies to the EEG channels.

**\---diagnoise**

    Omit the off-diagonal terms of the noise covariance matrix. This option
    is irrelevant to most users.

**\---fmri <*name*>**

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

**\---fmrithresh <*value*>**

    This option is mandatory and has an effect only if a weighting function
    has been specified with the ``--fmri`` option. If the value
    is in the *a priori* files falls below this value
    at a particular source space point, the source covariance matrix
    values are multiplied by the value specified with the ``--fmrioff`` option
    (default 0.1). Otherwise it is left unchanged.

**\---fmrioff <*value*>**

    The value by which the source covariance elements are multiplied
    if the *a priori* weight falls below the threshold
    set with ``--fmrithresh`` , see above.

**\---srccov <*name*>**

    Use this diagonal source covariance matrix. By default the source covariance
    matrix is a multiple of the identity matrix. This option is irrelevant
    to most users.

**\---proj <*name*>**

    Include signal-space projection information from this file.

**\---inv <*name*>**

    Save the inverse operator decomposition here. By default, the script looks
    for a file whose name is derived from the forward solution file by
    replacing its ending ``-fwd.fif`` by <*options*> ``-inv.fif`` , where
    <*options*> includes options ``--meg``, ``--eeg``, and ``--fixed`` with the double
    dashes replaced by single ones.

**\---destdir <*directory*>**

    Optionally specifies a directory where the inverse operator will
    be stored.

.. note:: If bad channels are included in the calculation,    strange results may ensue. Therefore, it is recommended that the    data to be analyzed is carefully inspected with to assign the bad    channels correctly.

.. note:: For convenience, the MNE software includes bad-channel    designation files which can be used to ignore all magnetometer or    all gradiometer channels in Vectorview measurements. These files are    called ``vv_grad_only.bad`` and ``vv_mag_only.bad`` , respectively.    Both files are located in ``$MNE_ROOT/share/mne/templates`` .

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
