
.. _create_bem_model:

=======================
Creating the BEM meshes
=======================

.. contents:: Contents
   :local:
   :depth: 2


.. _BABBDHAG:

Using the watershed algorithm
#############################

The watershed algorithm [Segonne *et al.*,
2004] is part of the FreeSurfer software.
The name of the program is mri_watershed .
Its use in the MNE environment is facilitated by the script `mne_watershed_bem`.

After mne_watershed_bem has
completed, the following files appear in the subject's ``bem/watershed`` directory:

** <*subject*> _brain_surface**

    Contains the brain surface triangulation.

** <*subject*> _inner_skull_surface**

    Contains the inner skull triangulation.

** <*subject*> _outer_skull_surface**

    Contains the outer skull triangulation.

** <*subject*> _outer_skin_surface**

    Contains the scalp triangulation.

All of these surfaces are in the FreeSurfer format. In addition,
there will be a directory called ``bem/watershed/ws`` which
contains the brain MRI volume. Furthermore, mne_watershed_bem script
converts the scalp surface to fif format and saves the result to ``bem/``  <*subject*> ``-head.fif`` . The mne_analyze tool
described :ref:`ch_interactive_analysis` looks for this file the visualizations
involving the scalp surface.

.. _BABFCDJH:

Using FLASH images
##################

This method depends on the availablily of MRI data acquired
with a multi-echo FLASH sequence at two flip angles (5 and 30 degrees).
These data can be acquired separately from the MPRAGE data employed
in FreeSurfer cortical reconstructions but it is strongly recommended
that they are collected at the same time with the MPRAGEs or at
least with the same scanner. For easy co-registration, the images
should have FOV, matrix, slice thickness, gap, and slice orientation
as the MPRAGE data. For information on suitable pulse sequences,
see reference [B. Fischl *et al.* and J. Jovicich *et
al.*, 2006] in :ref:`CEGEGDEI`. At the Martinos
Center, use of the 1.5-T Avanto scanner (Bay 2) is recommended for
best results.

Creation of the BEM meshes using this method involves the
following steps:

- Organizing the MRI data. This is facilitated
  by the script mne_organize_dicom ,
  see :ref:`BABEBJHI`.

- Creating a synthetic 5-degree flip angle FLASH volume, register
  it with the MPRAGE data, and run the segmentation and meshing program.
  This step is accomplished by running the script mne_flash_bem , see :ref:`BABGICFE`.

- Inspecting the meshes with tkmedit, see :ref:`BABHJBED`.

.. note:: Different methods can be employed for the creation of the
          individual surfaces. For example, it may turn out that the
          watershed algorithm produces are better quality skin surface than
          the segmentation approach based on the FLASH images. If this is
          the case, ``outer_skin.surf`` can set to point to the corresponding
          watershed output file while the other surfaces can be picked from
          the FLASH segmentation data.

.. note:: The :ref:`mne_convert_surface` C utility can be used to convert
          text format triangulation files into the FreeSurfer surface format.

.. note:: The following sections assume that you have run the appropriate
          setup scripts to make both MNE and FreeSurfer software available.

.. _BABEBJHI:

Organizing MRI data into directories
====================================

Since all images comprising the multi-echo FLASH data are
contained in a single series, it is necessary to organize the images
according to the echoes before proceeding to the BEM surface reconstruction.
This is accomplished by the mne_organize_dicom script,
which creates a directory tree with symbolic links to the original
DICOM image files. To run mne_organize_dicom ,
proceed as follows:

- Copy all of your images or create symbolic
  links to them in a single directory. The images must be in DICOM
  format. We will refer to this directory as  <*source*> .

- Create another directory to hold the output of mne_organize_dicom . We
  will refer to this directory as  <*dest*> .

- Change the working directory to  <*dest*> .

- Say ``mne_organize_dicom``  <*source*> .
  Depending on the total number of images in  <*source*> this
  script may take quite a while to run. Progress is  indicated by
  listing the number of images processed at 50-image intervals.

As a result,  <*dest*> will
contain several directories named  <*three-digit number*> _ <*protocol_name*> corresponding
to the different series of images acquired. Spaces and parenthesis
in protocol names will be replaced by underscores. Under each of
these directories there are one or more directories named  <*three-digit*> number
corresponding to one or more subsets of images in this series (protocol).
The only subset division scheme implemented in mne_organize_dicom is
that according to different echoes, typically found in multi-echo
FLASH data. These second level directories will contain symbolic
links pointing to the original image data.

.. note:: mne_organize_dicom was    developed specifically for Siemens DICOM data. Its correct behavior    with DICOM files originating from other MRI scanners has not been    verified at this time.

.. note:: Since mne_organize_dicom processes    all images, not only the FLASH data, it may be a useful preprocessing    step before FreeSurfer reconstruction process as well.

.. _BABGICFE:

Creating the surface tessellations
==================================

The BEM surface segmentation and tessellation is automated
with the script :ref:`mne_flash_bem`.
It assumes that a FreeSurfer reconstruction for this subject is
already in place.

Before running mne_flash_bem do the following:

- Run mne_organize_dicom as
  described above.

- Change to the  <*dest*> directory
  where mne_organize_dicom created the
  image directory structure.

- Create symbolic links from the directories containing the
  5-degree and 30-degree flip angle FLASH series to ``flash05`` and ``flash30`` , respectively:

  - ``ln -s``  <*FLASH 5 series dir*> ``flash05``

  - ``ln -s``  <*FLASH 30 series dir*> ``flash30``

- Some partition formats (e.g. FAT32) do not support symbolic links. In this case, copy the file to the appropriate series:

  - ``cp`` <*FLASH 5 series dir*> ``flash05``

  - ``cp`` <*FLASH 30 series dir*> ``flash30``

- Set the ``SUBJECTS_DIR`` and ``SUBJECT`` environment
  variables

.. note:: If mne_flash_bem is    run with the ``--noflash30`` option, the flash30 directory is not needed, *i.e.*,    only the 5-degree flip angle flash data are employed.

It may take a while for mne_flash_bem to
complete. It uses the FreeSurfer directory structure under ``$SUBJECTS_DIR/$SUBJECT`` .
The script encapsulates the following processing steps:

- It creates an mgz file corresponding
  to each of the eight echoes in each of the FLASH directories in ``mri/flash`` .
  The files will be called ``mef``  <*flip-angle*> _ <*echo-number*> ``.mgz`` .

- If the ``--unwarp`` option is specified, run grad_unwarp and produce
  files ``mef``  <*flip-angle*> _ <*echo-number*> ``u.mgz`` .
  These files will be then used in the following steps.

- It creates parameter maps in ``mri/flash/parameter_maps`` using mri_ms_fitparms .

- It creates a synthetic 5-degree flip angle volume in ``mri/flash/parameter_maps/flash5.mgz`` using mri_synthesize .

- Using fsl_rigid_register ,
  it creates a registered 5-degree flip angle volume ``mri/flash/parameter_maps/flash5_reg.mgz`` by
  registering ``mri/flash/parameter_maps/flash5.mgz`` to
  the *T1* volume under ``mri`` .

- Using mri_convert , it converts
  the flash5_reg volume to COR
  format under ``mri/flash5`` . If necessary, the T1 and brain volumes
  are also converted into the COR format.

- It runs mri_make_bem_surfaces to
  create the BEM surface tessellations.

- It creates the directory ``bem/flash`` , moves the
  tri-format tringulations there and creates the corresponding FreeSurfer
  surface files in the same directory.

- The COR format volumes created by mne_flash_bem are
  removed.

If the ``--noflash30`` option is specified to mne_flash_bem ,
steps 3 and 4 in the above are replaced by averaging over the different
echo times in 5-degree flip angle data.

.. _BABHJBED:

Inspecting the meshes
=====================

It is advisable to check the validity of the BEM meshes before
using them. This can be done with help of tkmedit, see :ref:`CIHDBFEG`.

Using seglab
############

The brain segmentation provided by FreeSurfer in the directory ``mri/brain`` can
be employed to create the inner skull surface triangulation with
help of seglab, the Neuromag MRI segmentation tool. The description
below assumes that the user is familiar with the seglab tool. If
necessary, consult the seglab manual, Neuromag P/N NM20420A-A.

The data set mri/brain typically
contains tissues within or outside the skull, in particular around
the eyes. These must be removed manually before the inner skull
triangulation is created.The editing and triangulation can be accomplished
as outlined below

**1. Set up the MRIs for Neuromag software access**

    Run the mne_setup_mri too as described in :ref:`BABCCEHF`.
    As a result, the directories mri/T1-neuromag and mri/brain-neuromag
    are set up.

**2. Load the MRI data**

    Open the file mri/brain-neuromag/sets/COR.fif and adjust the scaling
    of the data.

**3. Preparatory steps**

    Set the minimum data value to 1 using the min3D operator.
    Make a backup of the data with the backup3D operator.

**4. Manual editing**

    The maskDraw3D operation is recommended
    for manual editing. To use it, first employ the grow3D operator
    with threshold interval 2...255 and the seed point inside
    the brain. Then do the editing in the slicer window as described
    in Section 5.4.2 of the seglab manual. Note that it is enough to
    remove the connectivity to the extracerebral tissues rather than
    erasing them completely.

**5. Grow again and mask**

    Once manual editing is complete, employ the grow3D operator again
    and do mask3D with the backup
    data to see whether the result is satisfactory. If not, undo mask3D and
    continue manual editing. Otherwise, undo mask3D and
    proceed to the next step.

**6. Dilation**

    It is advisable to make the inner skull surface slightly bigger
    than the brain envelope obtained in the previous step. Therefore,
    apply the dilate3D operation
    once or twice. Use the values 1 for nbours and 26 for nhood in the
    first dilation and 1 and 18 in the second one, respectively.

**7. Triangulation**

    Triangulate the resulting object with the triangulate3D operator. Use
    a sidelength of 5 to 6 mm. Check that the triangulation looks reasonable
    in the 3D viewing window.

**8. Save the triangulation**

    Save the triangulated surface as a mesh into bem/inner_skull.tri. Select
    unit of measure as millimeters and employ the MRI coordinate system.

Using BrainSuite
################

The BrainSuite software
running under the Windows operating system can also be used for
BEM mesh generation. This software, written by David W. Shattuck,
is distributed as a collaborative project between the Laboratory
of Neuro Imaging at the University of California Los Angeles (Director:
Dr. Arthur W. Toga) and the Biomedical Imaging Research Group at
the University of Southern California (Director: Dr. Richard M. Leahy).
For further information, see http://brainsuite.usc.edu/.

The conversion of BrainSuite tessellation
files to MNE software compatible formats is accomplished with the mne_convert_surface utility,
covered in :ref:`mne_convert_surface`.

The workflow needed to employ the BrainSuite tessellations
is:

**Step 1**

    Using the mri_convert utility
    available in FreeSurfer , convert
    an MRI volume to the img (Analyze) format. This volume should be the
    T1.mgz volume or a volume registered with T1.mgz in FreeSurfer :``mri_convert``  <*volume*> ``.mgz``  <*volume*> ``.img``

**Step 2**

    Transfer  <*volume*> ``.mgz`` to
    a location accessible to BrainSuite , running
    on Windows.

**Step 3**

    Using  <*volume*> ``.img`` as
    input, create the tessellations of scalp, outer skull, and inner
    skull surfaces in BrainSuite .

**Step 4**

    Transfer the dfs files containing the tessellations in the bem directory
    of your subject's FreeSurfer reconstruction.

**Step 5**

    Go to the bem directory where you placed the two dfs files. Using mne_convert_surface ,
    convert them to the FreeSurfer surface
    format, *e,g.*:
    ``mne_convert_surface `` ``--dfs inner_skull.dfs `` ``--mghmri ../mri/T1.mgz `` ``--surf inner_skull_dfs.surf``

**Step 6**

    Using tkmedit, check that the surfaces are correct, *e.g.*:
    ``tkmedit -f ../mri/T1.mgz `` ``-surface inner_skull_dfs.surf``

**Step7**

    Using the mne_reduce_surface function
    in Matlab, reduce the number of triangles on the surfaces to 10000
    - 20000. Call the output files ``outer_skin.surf`` , ``outer_skull.surf`` ,
    and ``inner_skull.surf`` .

**Step 8**

    Proceed to mne_setup_forward_model .
    Use the ``--surf`` and ``--noswap`` options.

.. note:: If left and right are flipped in BrainSuite,    use the ``--flip`` option in mne_convert_surface to    set the coordinate transformation correctly.

.. note:: The BrainSuite scalp    surface can be also used for visualization in mne_analyze ,    see :ref:`CHDCGHIF`.
