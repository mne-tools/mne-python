

.. _ch_forward:

====================
The forward solution
====================

.. contents:: Contents
   :local:
   :depth: 2


Overview
########

This Chapter covers the definitions of different coordinate
systems employed in MNE software and FreeSurfer, the details of
the computation of the forward solutions, and the associated low-level
utilities.

.. _CHDEDFIB:

MEG/EEG and MRI coordinate systems
##################################

The coordinate systems used in MNE software (and FreeSurfer)
and their relationships are depicted in :ref:`CHDFFJIJ`.
Except for the *Sensor coordinates*, all of the
coordinate systems are Cartesian and have the "RAS" (Right-Anterior-Superior)
orientation, *i.e.*, the :math:`x` axis
points to the right, the :math:`y` axis
to the front, and the :math:`z` axis up.

.. _CHDFFJIJ:

.. figure:: pics/CoordinateSystems.png
    :alt: MEG/EEG and MRI coordinate systems

    MEG/EEG and MRI coordinate systems

    The coordinate transforms present in the fif files in MNE and the FreeSurfer files as well as those set to fixed values are indicated with :math:`T_x`, where :math:`x` identifies the transformation.

The coordinate systems related
to MEG/EEG data are:

**Head coordinates**

    This is a coordinate system defined with help of the fiducial landmarks
    (nasion and the two auricular points). In fif files, EEG electrode
    locations are given in this coordinate system. In addition, the head
    digitization data acquired in the beginning of an MEG, MEG/EEG,
    or EEG acquisition are expressed in head coordinates. For details,
    see :ref:`CHDEDFIB`.

**Device coordinates**

    This is a coordinate system tied to the MEG device. The relationship
    of the Device and Head coordinates is determined during an MEG measurement
    by feeding current to three to five head-position
    indicator (HPI) coils and by determining their locations with respect
    to the MEG sensor array from the magnetic fields they generate.

**Sensor coordinates**

    Each MEG sensor has a local coordinate system defining the orientation
    and location of the sensor. With help of this coordinate system,
    the numerical integration data needed for the computation of the
    magnetic field can be expressed conveniently as discussed in :ref:`BJEIAEIE`. The channel information data in the fif files
    contain the information to specify the coordinate transformation
    between the coordinates of each sensor and the MEG device coordinates.

The coordinate systems related
to MRI data are:

**Surface RAS coordinates**

    The FreeSurfer surface data are expressed in this coordinate system. The
    origin of this coordinate system is at the center of the conformed
    FreeSurfer MRI volumes (usually 256 x 256 x 256 isotropic 1-mm3  voxels)
    and the axes are oriented along the axes of this volume. The BEM
    surface and the locations of the sources in the source space are
    usually expressed in this coordinate system in the fif files. In
    this manual, the *Surface RAS coordinates* are
    usually referred to as *MRI coordinates* unless
    there is need to specifically discuss the different MRI-related
    coordinate systems.

**RAS coordinates**

    This coordinate system has axes identical to the Surface RAS coordinates but the location of the origin
    is different and defined by the original MRI data, i.e. ,
    the origin is in a scanner-dependent location. There is hardly any
    need to refer to this coordinate system explicitly in the analysis
    with the MNE software. However, since the Talairach coordinates,
    discussed below, are defined with respect to *RAS coordinates* rather
    than the *Surface RAS coordinates*, the RAS coordinate
    system is implicitly involved in the transformation between Surface RAS coordinates and the two *Talairach* coordinate
    systems.

**MNI Talairach coordinates**

    The definition of this coordinate system is discussed, e.g. ,
    in  http://imaging.mrc-cbu.cam.ac.uk/imaging/MniTalairach. This
    transformation is determined during the FreeSurfer reconstruction
    process.

**FreeSurfer Talairach coordinates**

    The problem with the MNI Talairach coordinates is that the linear MNI
    Talairach transform does matched the brains completely to the Talairach
    brain. This is probably because the Talairach atlas brain is a rather
    odd shape, and as a result, it is difficult to match a standard brain
    to the atlas brain using an affine transform. As a result, the MNI
    brains are slightly larger (in particular higher, deeper and longer)
    than the Talairach brain. The differences are larger as you get
    further from the middle of the brain, towards the outside. The FreeSurfer
    Talairach coordinates mitigate this problem by additing a an additional
    transformation, defined separately for negatice and positive MNI
    Talairach :math:`z` coordinates. These two
    transformations, denoted by :math:`T_-` and :math:`T_+` in :ref:`CHDFFJIJ`, are fixed as discussed in http://imaging.mrc-cbu.cam.ac.uk/imaging/MniTalairach
    (*Approach 2*).

The different coordinate systems are related by coordinate
transformations depicted in :ref:`CHDFFJIJ`. The arrows and
coordinate transformation symbols (:math:`T_x`)
indicate the transformations actually present in the FreeSurfer
files. Generally,

.. math::    \begin{bmatrix}
		x_2 \\
		y_2 \\
		z_2 \\
		1
	        \end{bmatrix} = T_{12} \begin{bmatrix}
		x_1 \\
		y_1 \\
		z_1 \\
		1
	        \end{bmatrix} = \begin{bmatrix}
		R_{11} & R_{12} & R_{13} & x_0 \\
		R_{13} & R_{13} & R_{13} & y_0 \\
		R_{13} & R_{13} & R_{13} & z_0 \\
		0 & 0 & 0 & 1
	        \end{bmatrix} \begin{bmatrix}
		x_1 \\
		y_1 \\
		z_1 \\
		1
	        \end{bmatrix}\ ,

where :math:`x_k`,:math:`y_k`,and :math:`z_k` are the location
coordinates in two coordinate systems, :math:`T_{12}` is
the coordinate transformation from coordinate system "1" to "2",
:math:`x_0`, :math:`y_0`,and :math:`z_0` is the location of the origin
of coordinate system "1" in coordinate system "2",
and :math:`R_{jk}` are the elements of the rotation
matrix relating the two coordinate systems. The coordinate transformations
are present in different files produced by FreeSurfer and MNE as
summarized in :ref:`CHDJDEDJ`. The fixed transformations :math:`T_-` and :math:`T_+` are:

.. math::    T_{-} = \begin{bmatrix}
		0.99 & 0 & 0 & 0 \\
		0 & 0.9688 & 0.042 & 0 \\
		0 & -0.0485 & 0.839 & 0 \\
		0 & 0 & 0 & 1
	        \end{bmatrix}

and

.. math::    T_{+} = \begin{bmatrix}
		0.99 & 0 & 0 & 0 \\
		0 & 0.9688 & 0.046 & 0 \\
		0 & -0.0485 & 0.9189 & 0 \\
		0 & 0 & 0 & 1
	        \end{bmatrix}

.. note:: This section does not discuss the transformation    between the MRI voxel indices and the different MRI coordinates.    However, it is important to note that in FreeSurfer, MNE, as well    as in Neuromag software an integer voxel coordinate corresponds    to the location of the center of a voxel. Detailed information on    the FreeSurfer MRI systems can be found at  https://surfer.nmr.mgh.harvard.edu/fswiki/CoordinateSystems.


.. tabularcolumns:: |p{0.2\linewidth}|p{0.3\linewidth}|p{0.5\linewidth}|
.. _CHDJDEDJ:
.. table:: Coordinate transformations in FreeSurfer and MNE software packages. The symbols :math:`T_x` are defined in :ref:`CHDFFJIJ`. Note: mne_make_cor_set /mne_setup_mri prior to release 2.6 did not include transformations :math:`T_3`, :math:`T_4`, :math:`T_-`, and :math:`T_+` in the fif files produced.

    +------------------------------+-------------------------------+--------------------------------------+
    | Transformation               | FreeSurfer                    | MNE                                  |
    +------------------------------+-------------------------------+--------------------------------------+
    | :math:`T_1`                  | Not present                   | | Measurement data files             |
    |                              |                               | | Forward solution files (`*fwd.fif`)|
    |                              |                               | | Inverse operator files (`*inv.fif`)|
    +------------------------------+-------------------------------+--------------------------------------+
    | :math:`T_{s_1}\dots T_{s_n}` | Not present                   | Channel information in files         |
    |                              |                               | containing :math:`T_1`.              |
    +------------------------------+-------------------------------+--------------------------------------+
    | :math:`T_2`                  | Not present                   | | MRI description files Separate     |
    |                              |                               | | coordinate transformation files    |
    |                              |                               | | saved from mne_analyze             |
    |                              |                               | | Forward solution files             |
    |                              |                               | | Inverse operator files             |
    +------------------------------+-------------------------------+--------------------------------------+
    | :math:`T_3`                  | `mri/*mgz` files              | MRI description files saved with     |
    |                              |                               | mne_make_cor_set if the input is in  |
    |                              |                               | mgz or mgh format.                   |
    +------------------------------+-------------------------------+--------------------------------------+
    | :math:`T_4`                  | mri/transforms/talairach.xfm  | MRI description files saved with     |
    |                              |                               | mne_make_cor_set if the input is in  |
    |                              |                               | mgz or mgh format.                   |
    +------------------------------+-------------------------------+--------------------------------------+
    | :math:`T_-`                  | Hardcoded in software         | MRI description files saved with     |
    |                              |                               | mne_make_cor_set if the input is in  |
    |                              |                               | mgz or mgh format.                   |
    +------------------------------+-------------------------------+--------------------------------------+
    | :math:`T_+`                  | Hardcoded in software         | MRI description files saved with     |
    |                              |                               | mne_make_cor_set if the input is in  |
    |                              |                               | mgz or mgh format.                   |
    +------------------------------+-------------------------------+--------------------------------------+

.. _BJEBIBAI:

The head and device coordinate systems
######################################

.. figure:: pics/HeadCS.png
    :alt: Head coordinate system

    The head coordinate system

The MEG/EEG head coordinate system employed in the MNE software
is a right-handed Cartesian coordinate system. The direction of :math:`x` axis
is from left to right, that of :math:`y` axis
to the front, and the :math:`z` axis thus
points up.

The :math:`x` axis of the head coordinate
system passes through the two periauricular or preauricular points
digitized before acquiring the data with positive direction to the
right. The :math:`y` axis passes through
the nasion and is normal to the :math:`x` axis.
The :math:`z` axis points up according to
the right-hand rule and is normal to the :math:`xy` plane.

The origin of the MEG device coordinate system is device
dependent. Its origin is located approximately at the center of
a sphere which fits the occipital section of the MEG helmet best
with :math:`x` axis axis going from left to right
and :math:`y` axis pointing front. The :math:`z` axis
is, again, normal to the :math:`xy` plane
with positive direction up.

.. note:: The above definition is identical to that    of the Neuromag MEG/EEG (head) coordinate system. However, in 4-D    Neuroimaging and CTF MEG systems the head coordinate frame definition    is different. The origin of the coordinate system is at the midpoint    of the left and right auricular points. The :math:`x` axis    passes through the nasion and the origin with positive direction    to the front. The :math:`y` axis is perpendicular    to the :math:`x` axis on the and lies in    the plane defined by the three fiducial landmarks, positive direction    from right to left. The :math:`z` axis is    normal to the plane of the landmarks, pointing up. Note that in    this convention the auricular points are not necessarily located    on :math:`y` coordinate axis. The file conversion utilities (see :ref:`BEHIAADG`)    take care of these idiosyncrasies and convert all coordinate information    to the MNE software head coordinate frame.

.. _BEHCGJDD:

Creating a surface-based source space
#####################################

The fif format source space files containing the dipole locations
and orientations are created with the utility mne_make_source_space .
This utility is usually invoked by the convenience script mne_setup_source_space ,
see :ref:`CIHCHDAE`.

The command-line options are:

**\---version**

    Show the program version and compilation date.

**\---help**

    List the command-line options.

**\---subject <*name*>**

    Name of the subject in SUBJECTS_DIR. In the absence of this option,
    the SUBJECT environment variable will be consulted. If it is not
    defined, mne_setup_source_space exits
    with an error.

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

**\---surf <*name1*>: <*name2*>:...**

    FreeSurfer surface file names specifying the source surfaces, separated
    by colons.

**\---spacing <*spacing/mm*>**

    Specifies the approximate grid spacing of the source space in mm.

**\---ico <*number*>**

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

**\---all**

    Include all nodes to the output. The active dipole nodes are identified
    in the fif file by a separate tag. If tri files were used as input
    the output file will also contain information about the surface
    triangulation. This option is always recommended to include complete
    information.

**\---src <*name*>**

    Output file name. Use a name <*dir*>/<*name*>-src.fif

.. note:: If both ``--ico`` and ``--spacing`` options    are present the later one on the command line takes precedence.

.. note:: Due to the differences between the FreeSurfer    and MNE libraries, the number of source space points generated with    the ``--spacing`` option may be different between the current    version of MNE and versions 2.5 or earlier (using ``--spacing`` option    to mne_setup_source_space ) if    the FreeSurfer surfaces employ the (old) quadrangle format or if    there are topological defects on the surfaces. All new FreeSurfer    surfaces are specified as triangular tessellations and are e of    defects.

.. _BJEFEHJI:

Creating a volumetric or discrete source space
##############################################

In addition to source spaces confined to a surface, the MNE
software provides some support for three-dimensional source spaces
bounded by a surface as well as source spaces comprised of discrete,
arbitrarily located source points. The mne_volume_source_space utility
assists in generating such source spaces.

The command-line options are:

**\---version**

    Show the program version and compilation date.

**\---help**

    List the command-line options.

**\---surf <*name*>**

    Specifies a FreeSurfer surface file containing the surface which
    will be used as the boundary for the source space.

**\---bem <*name*>**

    Specifies a BEM file (ending in ``-bem.fif`` ). The inner
    skull surface will be used as the boundary for the source space.

**\---origin <*x/mm*> : <*y/mm*> : <*z/mm*>**

    If neither of the two surface options described above is present,
    the source space will be spherical with the origin at this location,
    given in MRI (RAS) coordinates.

**\---rad <*radius/mm*>**

    Specifies the radius of a spherical source space. Default value
    = 90 mm

**\---grid <*spacing/mm*>**

    Specifies the grid spacing in the source space.

**\---mindist <*distance/mm*>**

    Only points which are further than this distance from the bounding surface
    are included. Default value = 5 mm.

**\---exclude <*distance/mm*>**

    Exclude points that are closer than this distance to the center
    of mass of the bounding surface. By default, there will be no exclusion.

**\---mri <*name*>**

    Specifies a MRI volume (in mgz or mgh format).
    If this argument is present the output source space file will contain
    a (sparse) interpolation matrix which allows mne_volume_data2mri to
    create an MRI overlay file, see :ref:`BEHDEJEC`.

**\---pos <*name*>**

    Specifies a name of a text file containing the source locations
    and, optionally, orientations. Each line of the file should contain
    3 or 6 values. If the number of values is 3, they indicate the source
    location, in millimeters. The orientation of the sources will be
    set to the z-direction. If the number of values is 6, the source
    orientation will be parallel to the vector defined by the remaining
    3 numbers on each line. With ``--pos`` , all of the options
    defined above will be ignored. By default, the source position and
    orientation data are assumed to be given in MRI coordinates.

**\---head**

    If this option is present, the source locations and orientations
    in the file specified with the ``--pos`` option are assumed
    to be given in the MEG head coordinates.

**\---meters**

    Indicates that the source locations in the file defined with the ``--pos`` option
    are give in meters instead of millimeters.

**\---src <*name*>**

    Specifies the output file name. Use a name * <*dir*>/ <*name*>*-src.fif

**\---all**

    Include all vertices in the output file, not just those in use.
    This option is implied when the ``--mri`` option is present.
    Even with the ``--all`` option, only those vertices actually
    selected will be marked to be "in use" in the
    output source space file.

.. _BEHCACCJ:

Creating the BEM meshes
#######################

The mne_surf2bem utility
converts surface triangle meshes from ASCII and FreeSurfer binary
file formats to the fif format. The resulting fiff file also contains
conductivity information so that it can be employed in the BEM calculations.

.. note:: The utility mne_tri2fiff previously    used for this task has been replaced by mne_surf2bem .

.. note:: The convenience script mne_setup_forward_model described in :ref:`CIHDBFEG` calls mne_surf2bem with    the appropriate options.

.. note:: The vertices of all surfaces should be given    in the MRI coordinate system.

Command-line options
====================

This program has the following
command-line options:

**\---version**

    Show the program version and compilation date.

**\---help**

    List the command-line options.

**\---surf <*name*>**

    Specifies a FreeSurfer binary format surface file. Before specifying the
    next surface (``--surf`` or ``--tri`` options)
    details of the surface specification can be given with the options
    listed in :ref:`BEHCDICC`.

**\---tri <*name*>**

    Specifies a text format surface file. Before specifying the next
    surface (``--surf`` or ``--tri`` options) details
    of the surface specification can be given with the options listed
    in :ref:`BEHCDICC`. The format of these files is described
    in :ref:`BEHDEFCD`.

**\---check**

    Check that the surfaces are complete and that they do not intersect. This
    is a recommended option. For more information, see :ref:`BEHCBDDE`.

**\---checkmore**

    In addition to the checks implied by the ``--check`` option,
    check skull and skull thicknesses. For more information, see :ref:`BEHCBDDE`.

**\---fif <*name*>**

    The output fif file containing the BEM. These files normally reside in
    the bem subdirectory under the subject's mri data. A name
    ending with ``-bem.fif`` is recommended.

.. _BEHCDICC:

Surface options
===============

These options can be specified after each ``--surf`` or ``--tri`` option
to define details for the corresponding surface.

**\---swap**

    Swap the ordering or the triangle vertices. The standard convention in
    the MNE software is to have the vertices ordered so that the vector
    cross product of the vectors from vertex 1 to 2 and 1 to 3 gives the
    direction of the outward surface normal. Text format triangle files
    produced by the some software packages have an opposite order. For
    these files, the ``--swap`` . option is required. This option does
    not have any effect on the interpretation of the FreeSurfer surface
    files specified with the ``--surf`` option.

**\---sigma <*value*>**

    The conductivity of the compartment inside this surface in S/m.

**\---shift <*value/mm*>**

    Shift the vertices of this surface by this amount, given in mm,
    in the outward direction, *i.e.*, in the positive
    vertex normal direction.

**\---meters**

    The vertex coordinates of this surface are given in meters instead
    of millimeters. This option applies to text format files only. This
    definition does not affect the units of the shift option.

**\---id <*number*>**

    Identification number to assign to this surface. (1 = inner skull, 3
    = outer skull, 4 = scalp).

**\---ico <*number*>**

    Downsample the surface to the designated subdivision of an icosahedron.
    This option is relevant (and required) only if the triangulation
    is isomorphic with a recursively subdivided icosahedron. For example,
    the surfaces produced by with mri_watershed are
    isomorphic with the 5th subdivision of a an icosahedron thus containing 20480
    triangles. However, this number of triangles is too large for present
    computers. Therefore, the triangulations have to be decimated. Specifying ``--ico 4`` yields 5120 triangles per surface while ``--ico 3`` results
    in 1280 triangles. The recommended choice is ``--ico 4`` .

.. _BEHDEFCD:

Tessellation file format
========================

The format of the text format surface files is the following:

  | <*nvert*>
  | <*vertex 1*>
  | <*vertex 2*>
  | ...
  | <*vertex nvert*>
  | <*ntri*>
  | <*triangle 1*>
  | <*triangle 2*>
  | ...
  | <*triangle ntri*> ,

where <*nvert*> and <*ntri*> are
the number of vertices and number of triangles in the tessellation,
respectively.

The format of a vertex entry is
one of the following:

**x y z**

    The x, y, and z coordinates of the vertex location are given in
    mm.

**number x y z**

    A running number and the x, y, and z coordinates are given. The running
    number is not considered by mne_tri2fiff. The nodes must be thus
    listed in the correct consecutive order.

**x y z nx ny nz**

    The x, y, and z coordinates as well as the approximate vertex normal direction
    cosines are given.

**number x y z nx ny nz**

    A running number is given in addition to the vertex location and vertex
    normal.

Each triangle entry consists of the numbers of the vertices
belonging to a triangle. The vertex numbering starts from one. The
triangle list may also contain running numbers on each line describing
a triangle.

.. _BEHCBDDE:

Topology checks
===============

If the ``--check`` option is specified, the following
topology checks are performed:

- The completeness of each surface is
  confirmed by calculating the total solid angle subtended by all
  triangles from a point inside the triangulation. The result should
  be very close to :math:`4 \pi`. If the result
  is :math:`-4 \pi` instead, it is conceivable
  that the ordering of the triangle vertices is incorrect and the
  ``--swap`` option should be specified.

- The correct ordering of the surfaces is verified by checking
  that the surfaces are inside each other as expected. This is accomplished
  by checking that the sum solid angles subtended by triangles of
  a surface :math:`S_k` at all vertices of another
  surface :math:`S_p` which is supposed to be
  inside it equals :math:`4 \pi`. Naturally, this
  check is applied only if the model has more than one surface. Since
  the surface relations are transitive, it is enough to check that
  the outer skull surface is inside the skin surface and that the
  inner skull surface is inside the outer skull one.

- The extent of each of the triangulated volumes is checked.
  If the extent is smaller than 50mm, an error is reported. This
  may indicate that the vertex coordinates have been specified in
  meters instead of millimeters.

.. _CHDJFHEB:

Computing the BEM geometry data
###############################

The utility mne_prepare_bem_model computes
the geometry information for BEM. This utility is usually invoked
by the convenience script mne_setup_forward_model ,
see :ref:`CIHDBFEG`. The command-line options are:

**\---bem <*name*>**

    Specify the name of the file containing the triangulations of the BEM
    surfaces and the conductivities of the compartments. The standard
    ending for this file is ``-bem.fif`` and it is produced
    either with the utility mne_surf2bem (:ref:`BEHCACCJ`) or the convenience script mne_setup_forward_model ,
    see :ref:`CIHDBFEG`.

**\---sol <*name*>**

    Specify the name of the file containing the triangulation and conductivity
    information together with the BEM geometry matrix computed by mne_prepare_bem_model .
    The standard ending for this file is ``-bem-sol.fif`` .

**\---method <*approximation method*>**

    Select the BEM approach. If <*approximation method*> is ``constant`` ,
    the BEM basis functions are constant functions on each triangle
    and the collocation points are the midpoints of the triangles. With ``linear`` ,
    the BEM basis functions are linear functions on each triangle and
    the collocation points are the vertices of the triangulation. This
    is the preferred method to use. The accuracy will be the same or
    better than in the constant collocation approach with about half
    the number of unknowns in the BEM equations.

.. _BJEIAEIE:

Coil geometry information
#########################

This Section explains the presentation of MEG detection coil
geometry information the approximations used for different detection
coils in MNE software. Two pieces of information are needed to characterize
the detectors:

- The location and orientation a local
  coordinate system for each detector.

- A unique identifier, which has an one-to-one correspondence
  to the geometrical description of the coil.

The sensor coordinate system
============================

The sensor coordinate system is completely characterized
by the location of its origin and the direction cosines of three
orthogonal unit vectors pointing to the directions of the x, y,
and z axis. In fact, the unit vectors contain redundant information
because the orientation can be uniquely defined with three angles.
The measurement fif files list these data in MEG device coordinates.
Transformation to the MEG head coordinate frame can be easily accomplished
by applying the device-to-head coordinate transformation matrix
available in the data files provided that the head-position indicator
was used. Optionally, the MNE software forward calculation applies
another coordinate transformation to the head-coordinate data to
bring the coil locations and orientations to the MRI coordinate system.

If :math:`r_0` is a row vector for
the origin of the local sensor coordinate system and :math:`e_x`, :math:`e_y`, and :math:`e_z` are the row vectors for the
three orthogonal unit vectors, all given in device coordinates,
a location of a point :math:`r_C` in sensor coordinates
is transformed to device coordinates (:math:`r_D`)
by

.. math::    [r_D 1] = [r_C 1] T_{CD}\ ,

where

.. math::    T = \begin{bmatrix}
		e_x & 0 \\
		e_y & 0 \\
		e_z & 0 \\
		r_{0D} & 1
	        \end{bmatrix}\ .

Calculation of the magnetic field
=================================

The forward calculation in the MNE software computes the
signals detected by each MEG sensor for three orthogonal dipoles
at each source space location. This requires specification of the
conductor model, the location and orientation of the dipoles, and
the location and orientation of each MEG sensor as well as its coil
geometry.

The output of each SQUID sensor is a weighted sum of the
magnetic fluxes threading the loops comprising the detection coil.
Since the flux threading a coil loop is an integral of the magnetic
field component normal to the coil plane, the output of the k :sup:`th`
MEG channel, :math:`b_k` can be approximated by:

.. math::    b_k = \sum_{p = 1}^{N_k} {w_{kp} B(r_{kp}) \cdot n_{kp}}

where :math:`r_{kp}` are a set of :math:`N_k` integration
points covering the pickup coil loops of the sensor, :math:`B(r_{kp})` is
the magnetic field due to the current sources calculated at :math:`r_{kp}`, :math:`n_{kp}` are
the coil normal directions at these points, and :math:`w_{kp}` are
the weights associated to the integration points. This formula essentially
presents numerical integration of the magnetic field over the pickup
loops of sensor :math:`k`.

There are three accuracy levels for the numerical integration
expressed above. The *simple* accuracy means
the simplest description of the coil. This accuracy is not used
in the MNE forward calculations. The *normal* or *recommended* accuracy typically uses
two integration points for planar gradiometers, one in each half
of the pickup coil and four evenly distributed integration points
for magnetometers. This is the default accuracy used by MNE. If
the ``--accurate`` option is specified, the forward calculation typically employs
a total of eight integration points for planar gradiometers and
sixteen for magnetometers. Detailed information about the integration
points is given in the next section.

Implemented coil geometries
===========================

This section describes the coil geometries currently implemented
in Neuromag software. The coil types fall in two general categories:

- Axial gradiometers and planar gradiometers
  and

- Planar gradiometers.

For axial sensors, the *z* axis of the
local coordinate system is parallel to the field component detected, *i.e.*,
normal to the coil plane.For circular coils, the orientation of
the *x* and *y* axes on the
plane normal to the z axis is irrelevant. In the square coils employed
in the Vectorview (TM) system the *x* axis
is chosen to be parallel to one of the sides of the magnetometer
coil. For planar sensors, the *z* axis is likewise
normal to the coil plane and the x axis passes through the centerpoints
of the two coil loops so that the detector gives a positive signal
when the normal field component increases along the *x* axis.

:ref:`BGBBHGEC` lists the parameters of the *normal* coil
geometry descriptions :ref:`CHDBDFJE` lists the *accurate* descriptions. For simple accuracy,
please consult the coil definition file, see :ref:`BJECIGEB`.
The columns of the tables contain the following data:

- The number identifying the coil id.
  This number is used in the coil descriptions found in the FIF files.

- Description of the coil.

- Number of integration points used

- The locations of the integration points in sensor coordinates.

- Weights assigned to the field values at the integration points.
  Some formulas are listed instead of the numerical values to demonstrate
  the principle of the calculation. For example, in the normal coil
  descriptions of the planar gradiometers the weights are inverses
  of the baseline of the gradiometer to show that the output is in
  T/m.

.. note:: The coil geometry information is stored in the file $MNE_ROOT/share/mne/coil_def.dat, which is automatically created by the utility mne_list_coil_def , see :ref:`BJEHHJIJ`.

.. XXX : table of normal coil description is missing

.. tabularcolumns:: |p{0.1\linewidth}|p{0.3\linewidth}|p{0.1\linewidth}|p{0.25\linewidth}|p{0.2\linewidth}|
.. _BGBBHGEC:
.. table:: Normal coil descriptions. Note: If a plus-minus sign occurs in several coordinates, all possible combinations have to be included.

    +------+-------------------------+----+----------------------------------+----------------------+
    | Id   | Description             | n  | r/mm                             | w                    |
    +======+=========================+====+==================================+======================+
    | 2    | Neuromag-122            | 2  | (+/-8.1, 0, 0) mm                | +/-1 ⁄ 16.2mm        | 
    |      | planar gradiometer      |    |                                  |                      |
    +------+-------------------------+----+----------------------------------+----------------------+
    | 2000 | A point magnetometer    | 1  | (0, 0, 0)mm                      | 1                    |
    +------+-------------------------+----+----------------------------------+----------------------+
    | 3012 | Vectorview type 1       | 2  | (+/-8.4, 0, 0.3) mm              | +/-1 ⁄ 16.8mm        |
    |      | planar gradiometer      |    |                                  |                      |
    +------+-------------------------+----+----------------------------------+----------------------+
    | 3013 | Vectorview type 2       | 2  | (+/-8.4, 0, 0.3) mm              | +/-1 ⁄ 16.8mm        |
    |      | planar gradiometer      |    |                                  |                      |
    +------+-------------------------+----+----------------------------------+----------------------+
    | 3022 | Vectorview type 1       | 4  | (+/-6.45, +/-6.45, 0.3)mm        | 1/4                  |
    |      | magnetometer            |    |                                  |                      |
    +------+-------------------------+----+----------------------------------+----------------------+
    | 3023 | Vectorview type 2       | 4  | (+/-6.45, +/-6.45, 0.3)mm        | 1/4                  |
    |      | magnetometer            |    |                                  |                      |
    +------+-------------------------+----+----------------------------------+----------------------+
    | 3024 | Vectorview type 3       | 4  | (+/-5.25, +/-5.25, 0.3)mm        | 1/4                  |
    |      | magnetometer            |    |                                  |                      |
    +------+-------------------------+----+----------------------------------+----------------------+
    | 2000 | An ideal point          | 1  | (0.0, 0.0, 0.0)mm                | 1                    |
    |      | magnetometer            |    |                                  |                      |
    +------+-------------------------+----+----------------------------------+----------------------+
    | 4001 | Magnes WH               | 4  | (+/-5.75, +/-5.75, 0.0)mm        | 1/4                  |
    |      | magnetometer            |    |                                  |                      |
    +------+-------------------------+----+----------------------------------+----------------------+
    | 4002 | Magnes WH 3600          | 8  | (+/-4.5, +/-4.5, 0.0)mm          | 1/4                  |
    |      | axial gradiometer       |    | (+/-4.5, +/-4.5, 50.0)mm         | -1/4                 |
    +------+-------------------------+----+----------------------------------+----------------------+
    | 4003 | Magnes reference        | 4  | (+/-7.5, +/-7.5, 0.0)mm          | 1/4                  |
    |      | magnetometer            |    |                                  |                      |
    +------+-------------------------+----+----------------------------------+----------------------+
    | 4004 | Magnes reference        | 8  | (+/-20, +/-20, 0.0)mm            | 1/4                  |
    |      | gradiometer measuring   |    | (+/-20, +/-20, 135)mm            | -1/4                 |
    |      | diagonal gradients      |    |                                  |                      |
    +------+-------------------------+----+----------------------------------+----------------------+
    | 4005 | Magnes reference        | 8  | (87.5, +/-20, 0.0)mm             | 1/4                  |
    |      | gradiometer measuring   |    | (47.5, +/-20, 0.0)mm             | -1/4                 |
    |      | off-diagonal gradients  |    | (-87.5, +/-20, 0.0)mm            | 1/4                  |
    |      |                         |    | (-47.5, +/-20, 0.0)mm            | -1/4                 |
    +------+-------------------------+----+----------------------------------+----------------------+
    | 5001 | CTF 275 axial           | 8  | (+/-4.5, +/-4.5, 0.0)mm          | 1/4                  |
    |      | gradiometer             |    | (+/-4.5, +/-4.5, 50.0)mm         | -1/4                 |
    +------+-------------------------+----+----------------------------------+----------------------+
    | 5002 | CTF reference           | 4  | (+/-4, +/-4, 0.0)mm              | 1/4                  |
    |      | magnetometer            |    |                                  |                      |
    +------+-------------------------+----+----------------------------------+----------------------+
    | 5003 | CTF reference           | 8  | (+/-8.6, +/-8.6, 0.0)mm          | 1/4                  |
    |      | gradiometer measuring   |    | (+/-8.6, +/-8.6, 78.6)mm         | -1/4                 |
    |      | diagonal gradients      |    |                                  |                      |
    +------+-------------------------+----+----------------------------------+----------------------+

.. tabularcolumns:: |p{0.1\linewidth}|p{0.3\linewidth}|p{0.05\linewidth}|p{0.25\linewidth}|p{0.15\linewidth}|
.. _CHDBDFJE:
.. table:: Accurate coil descriptions

    +------+-------------------------+----+----------------------------------+----------------------+
    | Id   | Description             | n  | r/mm                             | w                    |
    +======+=========================+====+==================================+======================+
    | 2    | Neuromag-122 planar     | 8  | +/-(8.1, 0, 0) mm                | +/-1 ⁄ 16.2mm        |
    |      | gradiometer             |    |                                  |                      |
    +------+-------------------------+----+----------------------------------+----------------------+
    | 2000 | A point magnetometer    | 1  | (0, 0, 0) mm                     | 1                    |
    +------+-------------------------+----+----------------------------------+----------------------+
    | 3012 | Vectorview type 1       | 2  | (+/-8.4, 0, 0.3) mm              | +/-1 ⁄ 16.8mm        |
    |      | planar gradiometer      |    |                                  |                      |
    +------+-------------------------+----+----------------------------------+----------------------+
    | 3013 | Vectorview type 2       | 2  | (+/-8.4, 0, 0.3) mm              | +/-1 ⁄ 16.8mm        |
    |      | planar gradiometer      |    |                                  |                      |
    +------+-------------------------+----+----------------------------------+----------------------+
    | 3022 | Vectorview type 1       | 4  | (+/-6.45, +/-6.45, 0.3)mm        | 1/4                  |
    |      | magnetometer            |    |                                  |                      |
    +------+-------------------------+----+----------------------------------+----------------------+
    | 3023 | Vectorview type 2       | 4  | (+/-6.45, +/-6.45, 0.3)mm        | 1/4                  |
    |      | magnetometer            |    |                                  |                      |
    +------+-------------------------+----+----------------------------------+----------------------+
    | 3024 | Vectorview type 3       | 4  | (+/-5.25, +/-5.25, 0.3)mm        | 1/4                  |
    |      | magnetometer            |    |                                  |                      |
    +------+-------------------------+----+----------------------------------+----------------------+
    | 4001 | Magnes WH magnetometer  | 4  | (+/-5.75, +/-5.75, 0.0)mm        | 1/4                  |
    +------+-------------------------+----+----------------------------------+----------------------+
    | 4002 | Magnes WH 3600          | 4  | (+/-4.5, +/-4.5, 0.0)mm          | 1/4                  |
    |      | axial gradiometer       |    | (+/-4.5, +/-4.5, 0.0)mm          | -1/4                 |
    +------+-------------------------+----+----------------------------------+----------------------+
    | 4004 | Magnes reference        | 8  | (+/-20, +/-20, 0.0)mm            | 1/4                  |
    |      | gradiometer measuring   |    | (+/-20, +/-20, 135)mm            | -1/4                 |
    |      | diagonal gradients      |    |                                  |                      |
    +------+-------------------------+----+----------------------------------+----------------------+
    | 4005 | Magnes reference        | 8  | (87.5, +/-20, 0.0)mm             | 1/4                  |
    |      | gradiometer measuring   |    | (47.5, +/-20, 0.0)mm             | -1/4                 |
    |      | off-diagonal gradients  |    | (-87.5, +/-20, 0.0)mm            | 1/4                  |
    |      |                         |    | (-47.5, +/-20, 0.0)mm            | -1/4                 |
    +------+-------------------------+----+----------------------------------+----------------------+
    | 5001 | CTF 275 axial           | 8  | (+/-4.5, +/-4.5, 0.0)mm          | 1/4                  |
    |      | gradiometer             |    | (+/-4.5, +/-4.5, 50.0)mm         | -1/4                 |
    +------+-------------------------+----+----------------------------------+----------------------+
    | 5002 | CTF reference           | 4  | (+/-4, +/-4, 0.0)mm              | 1/4                  |
    |      | magnetometer            |    |                                  |                      |
    +------+-------------------------+----+----------------------------------+----------------------+
    | 5003 | CTF 275 reference       | 8  | (+/-8.6, +/-8.6, 0.0)mm          | 1/4                  |
    |      | gradiometer measuring   |    | (+/-8.6, +/-8.6, 78.6)mm         | -1/4                 |
    |      | diagonal gradients      |    |                                  |                      |
    +------+-------------------------+----+----------------------------------+----------------------+
    | 5004 | CTF 275 reference       | 8  | (47.8, +/-8.5, 0.0)mm            | 1/4                  |
    |      | gradiometer measuring   |    | (30.8, +/-8.5, 0.0)mm            | -1/4                 |
    |      | off-diagonal gradients  |    | (-47.8, +/-8.5, 0.0)mm           | 1/4                  |
    |      |                         |    | (-30.8, +/-8.5, 0.0)mm           | -1/4                 |
    +------+-------------------------+----+----------------------------------+----------------------+
    | 6001 | MIT KIT system axial    | 8  | (+/-3.875, +/-3.875, 0.0)mm      | 1/4                  |
    |      | gradiometer             |    | (+/-3.875, +/-3.875, 0.0)mm      | -1/4                 |
    +------+-------------------------+----+----------------------------------+----------------------+


.. _BJECIGEB:

The coil definition file
========================

The coil geometry information is stored in the text file
$MNE_ROOT/share/mne/coil_def.dat. In this file, any lines starting
with the pound sign (#) are comments. A coil definition starts with
a description line containing the following fields:

** <*class*>**

    This is a number indicating class of this coil. Possible values
    are listed in :ref:`BJEFABHA`.

** <*id*>**

    Coil id value. This value is listed in the first column of Tables :ref:`BGBBHGEC` and :ref:`CHDBDFJE`.

** <*accuracy*>**

    The coil representation accuracy. Possible values and their meanings
    are listed in :ref:`BJEHIBJC`.

** <*np*>**

    Number of integration points in this representation.

** <*size/m*>**

    The size of the coil. For circular coils this is the diameter of
    the coil and for square ones the side length of the square. This
    information is mainly included to facilitate drawing of the coil
    geometry. It should not be employed to infer a coil approximation
    for the forward calculations.

** <*baseline/m*>**

    The baseline of a this kind of a coil. This will be zero for magnetometer
    coils. This information is mainly included to facilitate drawing
    of the coil geometry. It should not be employed to infer a coil
    approximation for the forward calculations.

** <*description*>**

    Short description of this kind of a coil. If the description contains several
    words, it is enclosed in quotes.

.. _BJEFABHA:

.. table:: Coil class values

    =======  =======================================================
    Value    Meaning
    =======  =======================================================
    1        magnetometer
    2        first-order axial gradiometer
    3        planar gradiometer
    4        second-order axial gradiometer
    1000     an EEG electrode (used internally in software only).
    =======  =======================================================


.. tabularcolumns:: |p{0.1\linewidth}|p{0.5\linewidth}|
.. _BJEHIBJC:
.. table:: Coil representation accuracies.

    =======  =====================================================================
    Value    Meaning
    =======  =====================================================================
    1        The simplest representation available
    2        The standard or *normal* representation (see :ref:`BGBBHGEC`)
    3        The most *accurate* representation available (see :ref:`CHDBDFJE`)
    =======  =====================================================================

Each coil description line is followed by one or more integration
point lines, consisting of seven numbers:

** <*weight*>**

    Gives the weight for this integration point (last column in Tables :ref:`BGBBHGEC` and :ref:`CHDBDFJE`).

** <*x/m*> <*y/m*> <*z/m*>**

    Indicates the location of the integration point (fourth column in Tables :ref:`BGBBHGEC` and :ref:`CHDBDFJE`).

** <*nx*> <*ny*> <*nz*>**

    Components of a unit vector indicating the field component to be selected.
    Note that listing a separate unit vector for each integration points
    allows the implementation of curved coils and coils with the gradiometer
    loops tilted with respect to each other.

.. _BJEHHJIJ:

Creating the coil definition file
=================================

The standard coil definition file $MNE_ROOT/share/mne/coil_def.dat
is included with the MNE software package. The coil definition file
can be recreated with the utility mne_list_coil_def
as follows:

mne_list_coil_def --out $MNE_ROOT/share/mne/coil_def.dat

.. _CHDDIBAH:

Computing the forward solution
##############################

Purpose
=======

Instead of using the convenience script mne_do_forward_solution it
is also possible to invoke the forward solution computation program mne_forward_solution directly.
In this approach, the convenience of the automatic file naming conventions
present in mne_do_forward_solution are
lost. However, there are some special-purpose options available
in mne_forward_solution only.
Please refer to :ref:`BABCHEJD` for information on mne_do_forward_solution.

.. _BJEIGFAE:

Command line options
====================

mne_forward_solution accepts
the following command-line options:

**\---src <*name*>**

    Source space name to use. The name of the file must be specified exactly,
    including the directory. Typically, the source space files reside
    in $SUBJECTS_DIR/$SUBJECT/bem.

**\---bem <*name*>**

    Specifies the BEM to be used. These files end with bem.fif or bem-sol.fif and
    reside in $SUBJECTS_DIR/$SUBJECT/bem. The former file contains only
    the BEM surface information while the latter files contain the geometry
    information precomputed with mne_prepare_bem_model ,
    see :ref:`CHDJFHEB`. If precomputed geometry is not available,
    the linear collocation solution will be computed by mne_forward_solution .

**\---origin <*x/mm*> : <*x/mm*> : <*z/mm*>**

    Indicates that the sphere model should be used in the forward calculations.
    The origin is specified in MEG head coordinates unless the ``--mricoord`` option
    is present. The MEG sphere model solution computed using the analytical
    Sarvas formula. For EEG, an approximative solution described in

**\---eegmodels <*name*>**

    This option is significant only if the sphere model is used and
    EEG channels are present. The specified file contains specifications
    of the EEG sphere model layer structures as detailed in :ref:`CHDIAFIG`. If this option is absent the file ``$HOME/.mne/EEG_models`` will
    be consulted if it exists.

**\---eegmodel <*model name*>**

    Specifies the name of the sphere model to be used for EEG. If this option
    is missing, the model Default will
    be employed, see :ref:`CHDIAFIG`.

**\---eegrad <*radius/mm*>**

    Specifies the radius of the outermost surface (scalp) of the EEG sphere
    model, see :ref:`CHDIAFIG`. The default value is 90 mm.

**\---eegscalp**

    Scale the EEG electrode locations to the surface of the outermost sphere
    when using the sphere model.

**\---accurate**

    Use accurate MEG sensor coil descriptions. This is the recommended
    choice. More information

**\---fixed**

    Compute the solution for sources normal to the cortical mantle only. This
    option should be used only for surface-based and discrete source
    spaces.

**\---all**

    Compute the forward solution for all vertices on the source space.

**\---label <*name*>**

    Compute the solution only for points within the specified label. Multiple
    labels can be present. The label files should end with ``-lh.label`` or ``-rh.label`` for
    left and right hemisphere label files, respectively. If ``--all`` flag
    is present, all surface points falling within the labels are included.
    Otherwise, only decimated points with in the label are selected.

**\---mindist <*dist/mm*>**

    Omit source space points closer than this value to the inner skull surface.
    Any source space points outside the inner skull surface are automatically
    omitted. The use of this option ensures that numerical inaccuracies
    for very superficial sources do not cause unexpected effects in
    the final current estimates. Suitable value for this parameter is
    of the order of the size of the triangles on the inner skull surface.
    If you employ the seglab software to create the triangulations, this
    value should be about equal to the wish for the side length of the
    triangles.

**\---mindistout <*name*>**

    Specifies a file name to contain the coordinates of source space points
    omitted due to the ``--mindist`` option.

**\---mri <*name*>**

    The name of the MRI description file containing the MEG/MRI coordinate
    transformation. This file was saved as part of the alignment procedure
    outlined in :ref:`CHDBEHDC`. These files typically reside in ``$SUBJECTS_DIR/$SUBJECT/mri/T1-neuromag/sets`` .

**\---trans	 <*name*>**

    The name of a text file containing the 4 x 4 matrix for the coordinate transformation
    from head to mri coordinates. With ``--trans``, ``--mri`` option is not
    required.

**\---notrans**

    The MEG/MRI coordinate transformation is taken as the identity transformation, *i.e.*,
    the two coordinate systems are the same. This option is useful only
    in special circumstances. If more than one of the ``--mri`` , ``--trans`` ,
    and ``--notrans`` options are specified, the last one remains
    in effect.

**\---mricoord**

    Do all computations in the MRI coordinate system. The forward solution
    matrix is not affected by this option if the source orientations
    are fixed to be normal to the cortical mantle. If all three source components
    are included, the forward three source orientations parallel to
    the coordinate axes is computed. If ``--mricoord`` is present, these
    axes correspond to MRI coordinate system rather than the default
    MEG head coordinate system. This option is useful only in special
    circumstances.

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
    data. A name of the form <*name*>-fwd.fif is
    recommended.

**\---meg**

    Compute the MEG forward solution.

**\---eeg**

    Compute the EEG forward solution.

**\---grad**

    Include the derivatives of the fields with respect to the dipole
    position coordinates to the output, see :ref:`BJEFEJJG`.

Implementation of software gradient compensation
================================================

As described in :ref:`BEHDDFBI` the CTF and 4D Neuroimaging
data may have been subjected to noise cancellation employing the
data from the reference sensor array. Even though these sensor are
rather far away from the brain sources, mne_forward_solution takes
them into account in the computations. If the data file specified
with the ``--meas`` option has software gradient compensation
activated, mne_forward_solution computes
the field of at the reference sensors in addition to the main MEG
sensor array and computes a compensated forward solution using the
methods described in :ref:`BEHDDFBI`.

.. warning:: If a data file specified with the ``--meas`` option    and that used in the actual inverse computations with mne_analyze and mne_make_movie have    different software gradient compensation states., the forward solution    will be in mismatch with the data to be analyzed and the current    estimates will be slightly erroneous.

.. _CHDIAFIG:

The EEG sphere model definition file
====================================

For the computation of the electric potential distribution
on the surface of the head (EEG) it is necessary to define the conductivities
(:math:`\sigma`) and radiuses of the spherically
symmetric layers. Different sphere models can be specified with
the ``--eegmodels`` option.

The EEG sphere model definition files may contain comment
lines starting with a # and model
definition lines in the following format:

 <*name*>: <*radius1*>: <*conductivity1*>: <*radius2*>: <*conductivity2*>:...

When the file is loaded the layers are sorted so that the
radiuses will be in ascending order and the radius of the outermost
layer is scaled to 1.0. The scalp radius specified with the ``--eegrad`` option
is then consulted to scale the model to the correct dimensions.
Even if the model setup file is not present, a model called Default is
always provided. This model has the structure given in :ref:`BABEBGDA`


.. tabularcolumns:: |p{0.1\linewidth}|p{0.25\linewidth}|p{0.2\linewidth}|
.. _BABEBGDA:
.. table:: Structure of the default EEG model

    ========  =======================  =======================
    Layer     Relative outer radius    :math:`\sigma` (S/m)
    ========  =======================  =======================
    Head      1.0                      0.33
    Skull     0.97                     0.04
    CSF       0.92                     1.0
    Brain     0.90                     0.33
    ========  =======================  =======================

EEG forward solution in the sphere model
========================================

When the sphere model is employed, the computation of the
EEG solution can be substantially accelerated by using approximation
methods described by Mosher, Zhang, and Berg, see :ref:`CEGEGDEI` (Mosher *et
al.* and references therein). mne_forward_solution approximates
the solution with three dipoles in a homogeneous sphere whose locations
and amplitudes are determined by minimizing the cost function:

.. math::    S(r_1,\dotsc,r_m\ ,\ \mu_1,\dotsc,\mu_m) = \int_{scalp} {(V_{true} - V_{approx})}\,dS

where :math:`r_1,\dotsc,r_m` and :math:`\mu_1,\dotsc,\mu_m` are
the locations and amplitudes of the approximating dipoles and :math:`V_{true}` and :math:`V_{approx}` are
the potential distributions given by the true and approximative
formulas, respectively. It can be shown that this integral can be
expressed in closed form using an expansion of the potentials in
spherical harmonics. The formula is evaluated for the most superficial
dipoles, *i.e.*, those lying just inside the
inner skull surface.

.. _BJEFEJJG:

Field derivatives
=================

If the ``--grad`` option is specified, mne_forward_solution includes
the derivatives of the forward solution with respect to the dipole
location coordinates to the output file. Let

.. math::    G_k = [g_{xk} g_{yk} g_{zk}]

be the :math:`N_{chan} \times 3` matrix containing
the signals produced by three orthogonal dipoles at location :math:`r_k` making
up :math:`N_{chan} \times 3N_{source}` the gain matrix

.. math::    G = [G_1 \dotso G_{N_{source}}]\ .

With the ``--grad`` option, the output from mne_forward_solution also
contains the :math:`N_{chan} \times 9N_{source}` derivative matrix

.. math::    D = [D_1 \dotso D_{N_{source}}]\ ,

where

.. math::    D_k = [\frac{\delta g_{xk}}{\delta x_k} \frac{\delta g_{xk}}{\delta y_k} \frac{\delta g_{xk}}{\delta z_k} \frac{\delta g_{yk}}{\delta x_k} \frac{\delta g_{yk}}{\delta y_k} \frac{\delta g_{yk}}{\delta z_k} \frac{\delta g_{zk}}{\delta x_k} \frac{\delta g_{zk}}{\delta y_k} \frac{\delta g_{zk}}{\delta z_k}]\ ,

where :math:`x_k`, :math:`y_k`, and :math:`z_k` are the location
coordinates of the :math:`k^{th}` dipole. If
the dipole orientations are to the cortical normal with the ``--fixed``
option, the dimensions of :math:`G` and :math:`D` are :math:`N_{chan} \times N_{source}` and :math:`N_{chan} \times 3N_{source}`,
respectively. Both :math:`G` and :math:`D` can
be read with the mne_read_forward_solution Matlab
function, see Table 10.1.

.. _CHDBBFCA:

Averaging forward solutions
###########################

Purpose
=======

One possibility to make a grand average over several runs
of a experiment is to average the data across runs and average the
forward solutions accordingly. For this purpose, mne_average_forward_solutions computes a
weighted average of several forward solutions. The program averages both
MEG and EEG forward solutions. Usually the EEG forward solution is
identical across runs because the electrode locations do not change.

Command line options
====================

mne_average_forward_solutions accepts
the following command-line options:

**\---version**

    Show the program version and compilation date.

**\---help**

    List the command-line options.

**\---fwd <*name*> :[ <*weight*> ]**

    Specifies a forward solution to include. If no weight is specified,
    1.0 is assumed. In the averaging process the weights are divided
    by their sum. For example, if two forward solutions are averaged
    and their specified weights are 2 and 3, the average is formed with
    a weight of 2/5 for the first solution and 3/5 for the second one.

**\---out <*name*>**

    Specifies the output file which will contain the averaged forward solution.
