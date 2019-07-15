

.. _ch_forward:

====================
The forward solution
====================

.. contents:: Contents
   :local:
   :depth: 2


Overview
########

This page covers the definitions of different coordinate
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
to the front, and the :math:`z` axis up. In some MNE-Python objects (e.g., :class:`~mne.Forward`, :class:`~mne.SourceSpaces`, etc), information about the coordinate frame is encoded as a constant integer value. The meaning of those integers is determined `in the source code <https://github.com/mne-tools/mne-python/blob/master/mne/io/constants.py#L186-L197>`__.

.. _CHDFFJIJ:

.. figure:: ../pics/CoordinateSystems.png
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
    FreeSurfer MRI volume (usually 256 x 256 x 256 isotropic 1-mm3  voxels)
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
    process. These coordinates are in MNI305 space.

**FreeSurfer Talairach coordinates**

    The problem with the MNI Talairach coordinates is that the linear MNI
    Talairach transform does not match the brains completely to the Talairach
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
		R_{21} & R_{22} & R_{23} & y_0 \\
		R_{31} & R_{32} & R_{33} & z_0 \\
		0 & 0 & 0 & 1
	        \end{bmatrix} \begin{bmatrix}
		x_1 \\
		y_1 \\
		z_1 \\
		1
	        \end{bmatrix}\ ,

where :math:`x_k`, :math:`y_k`,and :math:`z_k` are the location
coordinates in two coordinate systems, :math:`T_{12}` is
the coordinate transformation from coordinate system "1" to "2",
:math:`x_0`, :math:`y_0`, and :math:`z_0` is the location of the origin
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
.. table:: Coordinate transformations in FreeSurfer and MNE software packages.

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

.. note:: The symbols :math:`T_x` are defined in :ref:`CHDFFJIJ`. mne_make_cor_set /mne_setup_mri prior to release 2.6 did not include transformations :math:`T_3`, :math:`T_4`, :math:`T_-`, and :math:`T_+` in the fif files produced.

.. _BJEBIBAI:

The head and device coordinate systems
######################################

.. figure:: ../pics/HeadCS.png
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

.. note:: The above definition is identical to that of the Neuromag MEG/EEG
          (head) coordinate system. However, in 4-D Neuroimaging and CTF MEG
          systems the head coordinate frame definition is different. The origin
          of the coordinate system is at the midpoint of the left and right
          auricular points. The :math:`x` axis passes through the nasion and the
          origin with positive direction to the front. The :math:`y` axis is
          perpendicular to the :math:`x` axis on the and lies in the plane
          defined by the three fiducial landmarks, positive direction from right
          to left. The :math:`z` axis is normal to the plane of the landmarks,
          pointing up. Note that in this convention the auricular points are not
          necessarily located on :math:`y` coordinate axis. The file conversion
          utilities take care of these idiosyncrasies and convert all coordinate
          information to the MNE software head coordinate frame.

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

.. note:: MNE ships with several coil geometry configurations.
          They can be found in ``mne/data``.
          See :ref:`sphx_glr_auto_examples_visualization_plot_meg_sensors.py`
          for a comparison between different coil geometries, and
          :ref:`implemented_coil_geometries` for detailed information regarding
          the files describing Neuromag coil geometries.

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
the origin of the local sensor coordinate system and :math:`e_x`, :math:`e_y`,
and :math:`e_z` are the row vectors for the
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
the magnetic field due to the current sources calculated at :math:`r_{kp}`,
:math:`n_{kp}` are the coil normal directions at these points, and
:math:`w_{kp}` are the weights associated to the integration points. This
formula essentially presents numerical integration of the magnetic field over
the pickup loops of sensor :math:`k`.


.. _CHDDIBAH:

Computing the forward solution
##############################

Examples on how to compute the forward solution using
:func:`mne.make_forward_solution` can be found
:ref:`plot_forward_compute_forward_solution` and :ref:`BABCHEJD`.

.. note:: Notice that systems such as CTF and 4D Neuroimaging
          data may have been subjected to noise cancellation employing the
          data from the reference sensor array. Even though these sensor are
          rather far away from the brain sources, this can be taken into account
          using :meth:`mne.io.Raw.apply_gradient_compensation`.
          See :ref:`plot_brainstorm_phantom_ctf`.

.. _CHDIAFIG:
.. _ch_forward_spherical_model:

EEG forward solution in the sphere model
========================================

For the computation of the electric potential distribution
on the surface of the head (EEG) it is necessary to define the conductivities
(:math:`\sigma`) and radiuses of the spherically
symmetric layers. Different sphere models can be specified with
through :func:`mne.make_sphere_model`.
Here follows the default structure given when calling ``sphere = mne.make_sphere_model()``

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

Although it is not BEM model per se the ``sphere`` structure
describes the head geometry so it can be passed as ``bem`` parameter in
functions such as :func:`mne.fit_dipole`, :func:`mne.viz.plot_alignment`
or :func:`mne.make_forward_solution`.

When the sphere model is employed to compute the forward model using
:func:`mne.make_forward_solution`, the computation of the
EEG solution can be substantially accelerated by using approximation
methods described by Mosher, Zhang, and Berg, see :ref:`CEGEGDEI` (Mosher *et
al.* and references therein). In such scenario, MNE approximates
the solution with three dipoles in a homogeneous sphere whose locations
and amplitudes are determined by minimizing the cost function:

.. math::    S(r_1,\dotsc,r_m\ ,\ \mu_1,\dotsc,\mu_m) = \int_{scalp} {(V_{true} - V_{approx})}\,dS

where :math:`r_1,\dotsc,r_m` and :math:`\mu_1,\dotsc,\mu_m` are
the locations and amplitudes of the approximating dipoles and
:math:`V_{true}` and :math:`V_{approx}` are
the potential distributions given by the true and approximative
formulas, respectively. It can be shown that this integral can be
expressed in closed form using an expansion of the potentials in
spherical harmonics. The formula is evaluated for the most superficial
dipoles, *i.e.*, those lying just inside the
inner skull surface.

.. note:: See :ref:`Brainstorm CTF phantom dataset tutorial <plt_brainstorm_phantom_ctf_eeg_sphere_geometry>`,
          :ref:`Brainstorm Elekta phantom dataset tutorial <plt_brainstorm_phantom_elekta_eeg_sphere_geometry>`,
          and :ref:`plot_source_alignment_without_mri`.

.. _CHDBBFCA:

Averaging forward solutions
###########################

Purpose
=======

One possibility to make a grand average over several runs
of a experiment is to average the data across runs and average the
forward solutions accordingly. For this purpose,
:func:`mne.average_forward_solutions` computes a weighted average of several
forward solutions. The function averages both MEG and EEG forward solutions.
Usually the EEG forward solution is identical across runs because the electrode
locations do not change.
