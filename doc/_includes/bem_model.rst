:orphan:

Creating the BEM meshes
=======================

.. contents:: Page contents
   :local:
   :depth: 2

.. NOTE: part of this file is included in doc/overview/implementation.rst.
   Changes here are reflected there. If you want to link to this content, link
   to :ref:`bem-model` to link to that section of the implementation.rst page.
   The next line is a target for :start-after: so we can omit the title from
   the include:
   bem-begin-content

.. _bem_watershed_algorithm:

Using the watershed algorithm
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The watershed algorithm [Segonne *et al.*,
2004] is part of the FreeSurfer software.
The name of the program is mri_watershed .
Its use in the MNE environment is facilitated by the script
:ref:`gen_mne_watershed_bem`.

After ``mne watershed_bem`` has completed, the following files appear in the
subject's :file:`bem/watershed` directory:

- :file:`{<subject>}_brain_surface` contains the brain surface triangulation.

- :file:`{<subject>}_inner_skull_surface` contains the inner skull
  triangulation.

- :file:`{<subject>}_outer_skull_surface` contains the outer skull
  triangulation.

- :file:`{<subject>}_outer_skin_surface` contains the scalp triangulation.

All of these surfaces are in the FreeSurfer format. In addition, there will be
a file called :file:`bem/watershed/ws.mgz` which contains the brain MRI
volume. Furthermore, ``mne watershed_bem`` script converts the scalp surface to
fif format and saves the result to :file:`bem/{<subject>}-head.fif`.


Using FLASH images
~~~~~~~~~~~~~~~~~~

This method depends on the availablily of MRI data acquired with a multi-echo
FLASH sequence at two flip angles (5 and 30 degrees). These data can be
acquired separately from the MPRAGE data employed in FreeSurfer cortical
reconstructions but it is strongly recommended that they are collected at the
same time with the MPRAGEs or at least with the same scanner. For easy
co-registration, the images should have FOV, matrix, slice thickness, gap, and
slice orientation as the MPRAGE data. For information on suitable pulse
sequences, see reference [B. Fischl *et al.* and J. Jovicich *et al.*, 2006] in
:ref:`CEGEGDEI`.

Creation of the BEM meshes using this method involves the following steps:

- Creating a synthetic 5-degree flip angle FLASH volume, register
  it with the MPRAGE data, and run the segmentation and meshing program.
  This step is accomplished by running the script :ref:`gen_mne_flash_bem`.

- Inspecting the meshes with tkmedit, see :ref:`inspecting-meshes`.

.. note:: Different methods can be employed for the creation of the
          individual surfaces. For example, it may turn out that the
          watershed algorithm produces are better quality skin surface than
          the segmentation approach based on the FLASH images. If this is
          the case, ``outer_skin.surf`` can set to point to the corresponding
          watershed output file while the other surfaces can be picked from
          the FLASH segmentation data.


Organizing MRI data into directories
------------------------------------

Since all images comprising the multi-echo FLASH data are contained in a single
series, it is necessary to organize the images according to the echoes before
proceeding to the BEM surface reconstruction. This can be accomplished by using
`dcm2niix <https://www.nitrc.org/plugins/mwiki/index.php/dcm2nii:MainPage>`__
or the MNE-C tool ``mne_organize_dicom`` if necessary, then use
:func:`mne.bem.convert_flash_mris`.

Creating the surface tessellations
----------------------------------

The BEM surface segmentation and tessellation is automated with the script
:ref:`gen_mne_flash_bem`. It assumes that a FreeSurfer reconstruction for this
subject is already in place.

Before running :ref:`gen_mne_flash_bem` do the following:

- Create symbolic links from the directories containing the 5-degree and
  30-degree flip angle FLASH series to ``flash05`` and ``flash30``,
  respectively:

  - :samp:`ln -s {<FLASH 5 series dir>} flash05`

  - :samp:`ln -s {<FLASH 30 series dir>} flash30`

- Some partition formats (e.g. FAT32) do not support symbolic links. In this
  case, copy the file to the appropriate series:

  - :samp:`cp {<FLASH 5 series dir>} flash05`

  - :samp:`cp {<FLASH 30 series dir>} flash30`

- Set the ``SUBJECTS_DIR`` and ``SUBJECT`` environment variables or pass
  the ``--subjects-dir`` and ``--subject`` options to ``mne flash_bem``

.. note:: If ``mne flash_bem`` is run with the ``--noflash30`` option, the
   :file:`flash30` directory is not needed, *i.e.*, only the 5-degree flip
   angle flash data are employed.

It may take a while for ``mne flash_bem`` to complete. It uses the FreeSurfer
directory structure under ``$SUBJECTS_DIR/$SUBJECT``. The script encapsulates
the following processing steps:

- It creates an mgz file corresponding to each of the eight echoes in each of
  the FLASH directories in ``mri/flash``. The files will be called
  :file:`mef {<flip-angle>}_{<echo-number>}.mgz`.

- If the ``unwarp=True`` option is specified, run grad_unwarp and produce
  files :file:`mef {<flip-angle>}_{<echo-number>}u.mgz`. These files will be
  then used in the following steps.

- It creates parameter maps in :file:`mri/flash/parameter_maps` using
  ``mri_ms_fitparms``.

- It creates a synthetic 5-degree flip angle volume in
  :file:`mri/flash/parameter_maps/flash5.mgz` using ``mri_synthesize``.

- Using ``fsl_rigid_register``, it creates a registered 5-degree flip angle
  volume ``mri/flash/parameter_maps/flash5_reg.mgz`` by registering
  :file:`mri/flash/parameter_maps/flash5.mgz` to the *T1* volume under ``mri``.

- Using ``mri_convert``, it converts the flash5_reg volume to COR format under
  ``mri/flash5``. If necessary, the T1 and brain volumes are also converted
  into the COR format.

- It runs ``mri_make_bem_surfaces`` to create the BEM surface tessellations.

- It creates the directory :file:`bem/flash`, moves the tri-format
  tringulations there and creates the corresponding FreeSurfer surface files
  in the same directory.

- The COR format volumes created by ``mne flash_bem`` are removed.

If the ``--noflash30`` option is specified to ``mne flash_bem``,
steps 3 and 4 in the above are replaced by averaging over the different
echo times in 5-degree flip angle data.

.. _inspecting-meshes:

Inspecting the meshes
---------------------

It is advisable to check the validity of the BEM meshes before
using them. This can be done with:

- the ``--view`` option of :ref:`gen_mne_flash_bem`
- calling :func:`mne.viz.plot_bem` directly
- Using FreeSurfer tools ``tkmedit`` or ``freeview``
