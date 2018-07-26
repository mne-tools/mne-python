# -*- coding: utf-8 -*-
r"""
===================================
Background information on morphing
===================================

Here we give some background information on morphing in general,
and how it is done in MNE-Python in particular. Recommended reading and
corresponding information can be found in Gramfort *et al.* 2013 [1]_ and
Avants *et al.* 2009 [2]_ as well as in this `dipy example`_. For a shorter and
version of this tutorial see :ref:`sphx_glr_auto_tutorials_plot_morph.py`.

.. contents::
    :local:

Problem statement
=================

Modern neuroimaging techniques such as source reconstruction or fMRI analyses,
make use of advanced mathematical models and hardware to map brain activity
patterns into a subject specific anatomical brain space.

This enables the study of spatio-temporal brain activity. Amongst many others,
the representation of spatio-temporal brain data is often mapped onto the
anatomical brain structure. Thereby activity patterns are overlaid with
anatomical locations that supposedly produced the activity. Volumetric
anatomical MR images are often used as such or are transformed into an
inflated surface representations.

In order to compute group level statistics, data representations across
subjects must be morphed to a common frame, such that anatomically and
functional similar structures are represented at the same spatial location.

.. _tut_morphing_basics:

Morphing basics
================

Morphing describes the procedure of transforming a data representation in n-
dimensional space into another data representation in the same space. In the
context of neuroimaging data this space will mostly be 3-dimensional. It is
necessary to bring individual subject brain spaces into a common frame. So, is
it possible to shift functional brain data such, that it matches the
corresponding anatomical location of a different brain.

Generally speaking yes, but it needs to be kept in mind, that in order to
achieve a highly accurate overlay, morphological differences have to be
accounted as well.

Not just do sizes of brains vary, but also how cortical and
subcortical structures are expressed in terms of size and their location
relative to other structures. Thus it is inevitable to use slightly more
complex operations than just shifting brains relative to each other.
Nevertheless, this can be a first step of a successful subject to subject
transformation.

We can understand mapping as a mathematical function :math:`f`. This function
can be seen as a description on how to move from one point in n-dimensional
space to another. Like with a real map, when attempting to reach a new real
world location, we try to derive a set of actions (in the example directional
decisions to reach the new location), to move from one point to another. This
set of actions can also be described as a function of the current position
that computes the new location.

In general morphing operations can be split into two different kinds of
transformation: linear and non-linear morphs or mappings:

.. _tut_morphing_linear:

Linear morphs
-------------

A mapping is linear if it satisfies the following two conditions:

.. math::    f(u + v) = f(u) + f(v)\ ,
.. math::    f(cu) = cf(u)\ ,

where :math:`u` and :math:`v` are from the same vector space and :math:`c` can
be any scalar value.

This means that any linear transform can be described as a mixture of additive
and multiplicative operations and hence is often represented in terms of a
transformation matrix :math:`M` multiplied with the data :math:`x`:

.. math::

    x^{(B)}=
    \begin{bmatrix}
    x\prime \\
    y\prime \\
    z\prime \\
    1 \end{bmatrix} =
    M^{(AB)}x^{(A)}=
    \begin{bmatrix}
    1 & 0 & 0 & 0 \\
    0 & 1 & 0 & 0 \\
    0 & 0 & 1 & 0 \\
    0 & 0 & 0 & 1
    \end{bmatrix}
    \begin{bmatrix}
    x \\
    y \\
    z \\
    1
    \end{bmatrix}\ ,

where :math:`x^{(A)}` is the data that is morphed (subject A) and
:math:`x^{(B)}` the morphed data (data of subject A transformed into the space
of subject B). :math:`M^{(AB)}` represents the transformation matrix
transforming :math:`A` into :math:`B`. In the above example :math:`M` is an
identity matrix :math:`I_4` that does not cause any change such that
:math:`x^{(A)} = x^{(B)}`.

The simplest example on how transformation matrices work, would be scaling. If
we replace the :math:`1` in the second column of :math:`M^{(AB)}` with
:math:`s_y`, :math:`y` will be multiplied with this value and
:math:`x^{(B)}` becomes:

.. math::

    x^{(B)}=
    \begin{bmatrix}
    x \\
    s_y\cdot y \\
    z \\
    1 \end{bmatrix} =
    \begin{bmatrix}
    1 & 0 & 0 & 0 \\
    0 & s_y & 0 & 0 \\
    0 & 0 & 1 & 0 \\
    0 & 0 & 0 & 1
    \end{bmatrix}
    \begin{bmatrix}
    x \\
    y \\
    z \\
    1
    \end{bmatrix}\ ,

Please see the `Wikipedia article about transformation matrices`_ for more
information.

For completeness just a short note on translation. It is a special case and
furthermore the reason, why an additional column was added to :math:`x` and
:math:`M` even though the morphing was done in 3-dimensional space.

It is mathematically impossible to translate data represented in in N
dimensions in :math:`R^N`. Thus, an additional dimension is added such
that :math:`R^{N}` becomes :math:`R^{N+1}`. Now we can exploit the fact, that
shearing in :math:`R^{N+1}` causes a translation in the remaining :math:`R^N`.

Keeping the added dimension constant at :math:`1` will shear our data in
:math:`R^{N+1}` causing a translation in :math:`R^{N}`.

Imagine a plane in 2D. It's impossible to find a "translation matrix" in
:math:`R^{2}`. However, if we add a column of ones and move to :math:`R^{3}`,
the plane becomes a cuboid heaving length :math:`1` in the added dimension.

If we now apply a shearing to this object, the upper plane of the cuboid will
move relative to the lower plane. If we now ignore the 3rd dimension again and
go back to :math:`R^{2}`, then a translation happened.

Non-linear morphs
-----------------

In turn, a non-linear mapping can include linear components, but as well
functions that are not limited by linear constraints
(see :ref:`tut_morphing_linear`).

However it needs to be understood that including non-linear operations will
alter the relationship of data points within a vector and thus cannot be
represented as a transformation matrix.

If we go back to the example of scaling, we can now not only multiply each
value :math:`y` with the a constant :math:`s_y`, but instead :math:`s_y` can be
variable for each point in :math:`x`. In fact each data point can be mapped
(more) independently.

In MNE-Python "brain space" data is represented as source estimate data,
obtained by one of the implemented source reconstruction methods. See
:ref:`sphx_glr_auto_tutorials_plot_mne_dspm_source_localization.py`

It is thus represented as :class:`mne.SourceEstimate`,
:class:`mne.VectorSourceEstimate`, :class:`mne.VolSourceEstimate` or a mixture
of those.

The data in the first two cases is represented as "surfaces". This means that
the data is represented as vertices on an inflated brain surface
representation in form of a 3-dimensional mesh.

In the last case the data is represented in a 4-dimensional space, were the
last dimension refers to the data's sample points.

Computing an inflated surface representation can be accomplished using
FreeSurfer. Thereby, spherical morphing of the surfaces can be employed to
bring data from different subjects into a common anatomical frame.

When dealing with volumetric data, a non-linear morph map - or better "morph
volume" - based on two different anatomical reference MRIs, can be computed.

.. _tut_surface_morphing:

Surface morphing
================

The spherical morphing of the surfaces accomplished by FreeSurfer can be
employed to bring data from different subjects into a common anatomical
frame. This chapter describes utilities which make use of the spherical
morphing procedure. mne_morph_labels morphs label files between subjects
allowing the definition of labels in a one brain and transforming them to
anatomically analogous labels in another. mne_average_estimates offers
the capability to compute averages of data computed with the MNE software
across subjects.

.. _tut_surface_morphing_maps:

Morph maps
----------

The MNE software accomplishes surface morphing with help of morphing
maps which can be either computed on demand or precomputed using
mne_make_morph_maps , see :ref:`tut_surface_morphing_precompute`. The morphing
is performed with help of the registered spherical surfaces (``lh.sphere.reg``
and ``rh.sphere.reg`` ) which must be produced in FreeSurfer .
A morphing map is a linear mapping from cortical surface values
in subject A (:math:`x^{(A)}`) to those in another
subject B (:math:`x^{(B)}`)

.. math::    x^{(B)} = M^{(AB)} x^{(A)}\ ,

where :math:`M^{(AB)}` is a sparse matrix
with at most three nonzero elements on each row. These elements
are determined as follows. First, using the aligned spherical surfaces,
for each vertex :math:`x_j^{(B)}`, find the triangle :math:`T_j^{(A)}` on the
spherical surface of subject A which contains the location :math:`x_j^{(B)}`.
Next, find the numbers of the vertices of this triangle and set
the corresponding elements on the *j* th row of :math:`M^{(AB)}` so that
:math:`x_j^{(B)}` will be a linear interpolation between the triangle vertex
values reflecting the location :math:`x_j^{(B)}` within the triangle
:math:`T_j^{(A)}`.

It follows from the above definition that in general

.. math::    M^{(AB)} \neq (M^{(BA)})^{-1}\ ,

*i.e.*,

.. math::    x_{(A)} \neq M^{(BA)} M^{(AB)} x^{(A)}\ ,

even if

.. math::    x^{(A)} \approx M^{(BA)} M^{(AB)} x^{(A)}\ ,

*i.e.*, the mapping is *almost* a
bijection.

.. _tut_surface_morphing_smoothing:

Smoothing
---------

The current estimates are normally defined only in a decimated
grid which is a sparse subset of the vertices in the triangular
tessellation of the cortical surface. Therefore, any sparse set
of values is distributed to neighboring vertices to make the visualized
results easily understandable. This procedure has been traditionally
called smoothing but a more appropriate name
might be smudging or blurring in
accordance with similar operations in image processing programs.

In MNE software terms, smoothing of the vertex data is an
iterative procedure, which produces a blurred image :math:`x^{(N)}` from
the original sparse image :math:`x^{(0)}` by applying
in each iteration step a sparse blurring matrix:

.. math::    x^{(p)} = S^{(p)} x^{(p - 1)}\ .

On each row :math:`j` of the matrix :math:`S^{(p)}` there
are :math:`N_j^{(p - 1)}` nonzero entries whose values
equal :math:`1/N_j^{(p - 1)}`. Here :math:`N_j^{(p - 1)}` is
the number of immediate neighbors of vertex :math:`j` which
had non-zero values at iteration step :math:`p - 1`.
Matrix :math:`S^{(p)}` thus assigns the average
of the non-zero neighbors as the new value for vertex :math:`j`.
One important feature of this procedure is that it tends to preserve
the amplitudes while blurring the surface image.

Once the indices non-zero vertices in :math:`x^{(0)}` and
the topology of the triangulation are fixed the matrices :math:`S^{(p)}` are
fixed and independent of the data. Therefore, it would be in principle
possible to construct a composite blurring matrix

.. math::    S^{(N)} = \prod_{p = 1}^N {S^{(p)}}\ .

However, it turns out to be computationally more effective
to do blurring with an iteration. The above formula for :math:`S^{(N)}` also
shows that the smudging (smoothing) operation is linear.

.. _tut_surface_morphing_precompute:

Precomputing
------------

The utility :ref:`mne_make_morph_maps` was created to assist mne_analyze and
:ref:`mne_make_movie` in morphing. Since the morphing maps described above take
a while to compute, it is beneficial to construct all necessary maps in advance
before using mne_make_movie.

The precomputed morphing maps are located in ``$SUBJECTS_DIR/morph-maps`` .
mne_make_morph_maps creates this directory automatically if it does not
exist. If this directory exists when mne_analyze or mne_make_movie is run
and morphing is requested, the software first looks for already
existing morphing maps there. Also, if mne_analyze or :ref:`mne_make_movie`
have to recompute any morphing maps, they will be saved to
``$SUBJECTS_DIR/morph-maps`` if this directory exists.

The names of the files in ``$SUBJECTS_DIR/morph-maps`` are
of the form:

 <*A*> - <*B*> -``morph.fif`` ,

where <*A*> and <*B*> are
names of subjects. These files contain the maps for both hemispheres,
and in both directions, *i.e.*, both :math:`M^{(AB)}` and :math:`M^{(BA)}`, as
defined above. Thus the files <*A*> - <*B*> -``morph.fif`` or <*B*> - <*A*> -
``morph.fif`` are functionally equivalent. The name of the file produced by
mne_analyze or mne_make_movie depends on the role of <*A*> and <*B*> in
the analysis.

If you choose to compute the morphing maps in batch in advance,
use :ref:`mne_make_morph_maps`.

.. _tut_volumetric_morphing:

Volumetric morphing
===================

The key difference between volumetric morphing and surfaces transformations, is
that the data is represented as a volume in a discrete space. A volume is a
3-dimensional data representation, but it is necessary to understand that the
difference to a mesh (what's commonly referred to as "3D model") is that the
mesh is an "empty" surface, while the volume is "filled". Whereas the mesh is
defined by the vertices of the outer hull, the volume is defined by that and
the points it is containing.

Note, that since volumetric data "contains" something, the number of points
will always be greater or equal when switching from surface to volume on the
same resolution.

For example, a sphere of 5 mm radius, would contain
:math:`4\pi r^2 \approx 314` vertices, when sampled at 1 mm resolution.

The number of volume points however would scale with :math:`\frac{4}{3}\pi r^3`
and thus at the same resolution would contain :math:`\approx 524` data points.

Hence the 3D-volume representation of a sphere will always contain
:math:`\frac{r}{3}` times more data points than the 3D mesh if represented in
unit space.

And since we have to deal with the surface *and* the content that needs to be
morphed, we can directly link this and say, it's not only the cortices we want
to overlap as good as possible, but sub-cortical structures as well. If the
surface surrounds a perfectly homogeneous space, then the result of the
volumetric morph would be exactly equal to the surface morph.

In MNE-Python the implementation of volumetric morphing is achieved by wrapping
the corresponding functionality from DiPy. See this `dipy example`_ for
reference.

The volumetric morphing is implemented as a two stage approach. First two
reference brains are aligned using an :ref:`tut_volumetric_morphing_affine`
and later a non-linear :ref:`tut_volumetric_morphing_sdr` in 3D.

.. _tut_volumetric_morphing_affine:

Affine registration
-------------------

See `dipy affine example`_ for reference.

Our goal is to pre-align both reference volumes as good as possible, to make it
easier for the later non-linear optimization to converge to an acceptable
minimum. The quality of the pre-alignment will be assessed using the
mutual information that is aimed to be maximized [3]_.

Mutual information can be defined as the amount of predictive value two
variables share. Thus how much information about a random variable :math:`B`
can be obtained through a random variable :math:`A`. It is formally defined as:

.. math::

    I(A;B)=
    \sum_{a\in A}\sum_{b\in B}p(a,b)log\left(\frac{p(a,b)}{p(a)p(b)}\right)\ .

:math:`A` and :math:`B` is our subject data, :math:`p(a,b)` the joint and
:math:`p(a)` and :math:`p(b)` the marginal probabilitiy. This means, that for
independent variables :math:`p(a,b) = p(a)p(b)`, thus
:math:`I(A;B)=\sum_{a\in A}\sum_{b\in B}p(a,b)log 1=0`.

In general it can be stated, that: :math:`p(a,b) \geq p(a)p(b)` according to
`Jensen's inequality`_. Hence the more dependent :math:`A` and :math:`B` are,
the higher :math:`I(A;B)` will be.

It can further be expressed in terms of entropy. More precise, as the
difference between the joined entropy of :math:`A` and :math:`B` and the
respective conditional entropies.

The higher the joint entropy, the lower the conditional and hence one variable
is more predictive for the respective other. Aiming for maximizing the mutual
information can thus be seen as reducing the conditional entropy and thus the
amount of information required from a second variable, to describe the system.

Or in other words, the higher the mutual information is, that is shared by
:math:`A` and :math:`B`, the more can be said about both, by only obtaining
data from one.

If we find a transformation such, that both volumes are overlapping as good as
possible, then the location of a particular area in one brain, would be highly
predictive for the same location on the second brain. In turn the mutual
information is high, whereas the conditional entropy is low.

The specific optimization algorithm used for mutual information driven affine
registration is described in Mattes *et al.* 2003 [4]_.

In essence, a gradient decent is used to minimize the negative mutual
information, while optimizing the set of parameters of the image discrepancy
function at the same time. Those parameters describe the actual morphing
operation.

.. _tut_volumetric_morphing_sdr:

Symmetric Diffeomorphic Registration
------------------------------------

See `dipy sdr example`_ for reference.

Symmetric Diffeomorphic Image Registration is described in Avants *et al.* 2009
[2]_.

A smooth map between two manifolds that is invertible is diffeomorphic, if it
is differentiable and so is it's inverse.

Hence it can be seen as a locally linear, smooth map, that describes how each
point on one object relates to the same point on a second object. Imagine a
scrambled and an intact sheet of paper. There is a clear mapping between each
point of the first, to each point of the second object. However this map is not
necessarily linear. See :ref:`tut_morphing_basics`.

In fact Symmetric Diffeomorphic mapping can be represented as bending the
correspondence space smoothly. Since it is differentiable we can find a linear
map at each point in space, while the map itself is non-linear.

The introduced "symmetry" refers to symmetry after implementation. That is that
:math:`A \rightarrow B` yields computationally the same result as
:math:`B \rightarrow A`.

As optimization criterion the cross-correlation was chosen, which is a
deflection of similarity between two data series. It describes the correlation
between :math:`A` and :math:`B` at sample :math:`t` for each sample
point :math:`t+\tau`. The result of computing the cross-correlation is a series
describing the correlation between :math:`A` and :math:`B` at all sample
points:

.. math::

    \rho_{AB}(\tau) =
    \frac{E[(A_t-\mu_A)(B_{t+\tau}-\mu_B)]}{\sigma_A \sigma_B}\ ,

where :math:`\mu` and :math:`\sigma` are the mean and standard deviation of
:math:`A` and :math:`B`. Optimizing parameter values to maximize for this value
will hence make the data series more similar.

.. _tut_sourcemorph:

SourceMorph
===========

:class:`mne.SourceMorph` is MNE-Python's source estimation morphing operator.
It can perform all necessary computations, to achieve the above transformations
on surface source estimate representations as well as for volumetric source
estimates. This includes :class:`mne.SourceEstimate` and
:class:`mne.VectorSourceEstimate` for surface representations and
:class:`mne.VolSourceEstimate` for volumetric representations.

The general idea is to use a single operator for both surface (
:class:`mne.SourceEstimate`) and volumetric (:class:`mne.VolSourceEstimate`)
data.

SourceMorph can take general keyword arguments to indicate on which subjects
the morph will be performed. Subjects are defined by setting
``subject_from='subjectA'`` and ``subject_to='subjectB'``.

Both subjects must be FreeSurfer directories located in ``SUBJECTS_DIR``, which
can be redefined by setting ``subjects_dir='/subjects/dir'``.

Furthermore :class:`mne.SourceSpaces` must be provided when dealing with
volumetric data by setting ``src=src``. See :class:`mne.SourceMorph` for more
information.

A SourceMorph object can be created, by initializing an instance and setting
desired key word arguments like so:

``my_morph = mne.SourceMorph(...)``

``my_morph`` will have all arguments set or their default values as attributes.
Furthermore it indicates of which kind it is ``my_morph.kind`` and what the
respective morphing parameters ``my_morph.params`` are.

``my_morph.params`` is a dictionary, that varies depending on the type of
morph. All morph-relevant information is stored here. See
:ref:`tut_sourcemorph_space` and :ref:`tut_sourcemorph_opt` for more
information about type specific parameters as well as
:ref:`tut_surface_morphing` and :ref:`tut_volumetric_morphing` for type
specific background information.

.. _tut_sourcemorph_space:

About spacing
-------------

SourceMorph can take multiple arguments depending on the underlying morph. See
:class:`mne.SourceMorph`.

Here we will point out a special notion on the parameter ``spacing`` when
attempting to morph surface data. In case of (Vector)
:class:`mne.SourceEstimate` 'spacing' can be an integer a list of 2 np.array or
None. The default is spacing=5. Spacing refers to what was known as grade in
previous versions of MNE-Python. It defines the resolution of the icosahedral
mesh (typically 5). If None, all vertices will be used (potentially filling the
surface). If a list, then values will be morphed to the set of vertices
specified in in spacing[0] and spacing[1].

In turn, when morphing :class:`mne.VolSourceEstimate` spacing can be an
integer, float, tuple of integer or float or None. The default is spacing=5.
Spacing refers to the voxel size that is used to compute the volumetric morph.

Since two volumes are compared "point wise" the number of slices in each
orthogonal direction has direct influence on the computation time and accuracy
of the morph. See :ref:`tut_volumetric_morphing_sdr` to understand why this is
the case.

Spacing thus can also be seen as the voxel size to which both reference volumes
will be resliced before computing the symmetric diffeomorphic map. An
integer or float value, will be interpreted as isotropic voxel size in mm.
Setting a tuple allows for anisotropic voxel sizes e.g. (1., 1., 1.2). If None
the full resolution of the MRIs will be used. Note, that this can cause long
computation times. Furthermore, 'spacing' is not the resolution of the output
volume, if converted into a NIfTI file (except if ``mri_resolution=False``).

.. _tut_sourcemorph_opt:

About optimization parameters
-----------------------------

As described in :ref:`tut_volumetric_morphing`, the optimization is iterative.
Not the full feature space will be searched, but instead a subset. The
behavior can be controlled by specifying the parameters ``niter_affine`` and
``niter_sdr``.

The iterative optimization is performed in multiple levels and a number of
iterations per level. A level is a stage of iterative refinement with a certain
level of precision. The higher (or later) the level the more refined the
iterative optimization will be, requiring more computation time.

The number of levels and the number of iterations per level are defined as a
tuple of integers, where the number of integers or the length of the tuple
defines the number of levels, whereas the integer values themselves represent
the number of iterations in that respective level. The default for
``niter_affine`` is (100, 100, 10) referring to a 3 stage optimization using
100, 100 and 10 iterations for the 1st, 2nd and 3rd level.

Note, that a 3 step approach is computed internally: A **translation**,
followed by a **rigid body transform** (adding rotation), followed by an
**affine transform** (adding scaling). Thereby the result of the first step
will be the initial morph of the second and so on. Thus the defined number of
iterations actually applies to 3 different computations.

Similar to the affine registration, ``niter_sdr`` refers to N levels of
optimization during Symmetric Diffeomorphic registration, using 5, 5 and 3
iterations for the 1st, 2nd and 3rd level by default.

.. _tut_sourcemorph_methods:

SourceMorph's methods explained
-------------------------------

Once an instance of :class:`mne.SourceMorph` was created, it exposes 3 methods:

:meth:`my_morph() <mne.SourceMorph.__call__>`:

    Calling an instance of SourceMorph using :class:`mne.SourceEstimate`,
    :class:`mne.VectorSourceEstimate` or :class:`mne.VolSourceEstimate` as an
    input argument, will apply the precomputed morph to the input data and
    return the morphed source estimate (``stc_morphed = my_morph(stc)``).

    If a surface morph was attempted and no :class:`mne.SourceSpaces` was
    provided during instantiation of SourceMorph, then the actual computation
    of the morph will take place using the input data as reference data
    (rather then precomputing it based on the source space data). Additionally
    the method can take the same keyword arguments as
    :meth:`my_morph.as_volume() <mne.SourceMorph.as_volume>`, given that
    `as_volume=True`. This means that the result will not be a source estimate,
    but instead a NIfTI image representing the source estimate data in the
    specified way. If `as_volume=False` all other volume related arguments will
    be ignored.

:meth:`my_morph.as_volume() <mne.SourceMorph.as_volume>`:

    This method
    only works with :class:`mne.VolSourceEstimate`. It returns a NIfTI image
    of the source estimate. *mri_resolution* can be defined to change the
    resolution of the output image.

    ``mri_resolution=True`` will output an image in the same resolution as
    the MRI information stored in :class:`src <mne.SourceSpaces>`. If
    ``mri_resolution=False`` the output image will have the same resolution
    as defined in *spacing* when instantiating the morph (see
    :ref:`tut_sourcemorph_opt`). Furthermore, *mri_resolution* can be defined
    as integer, float or tuple of integer or float to refer to the desired
    voxel size in mm. A single value will be interpreted as isotropic voxel
    size, whereas anisotropic dimensions can be defined using a tuple. Note,
    that a tuple must have a length of 3 referring to the 3 orthogonal
    spatial dimensions. The default is mri_resolution=False.

    The keyword argument *mri_space* asks, whether to use the same reference
    space as the reference MRI of the reference space of the source estimate.
    The default is ``mri_space=True``.

    Furthermore a keyword argument called *apply_morph* can be set,
    indicating whether to apply the precomputed morph. In combination with
    the keyword argument 'as_volume', this can be used to produce morphed and
    unmorphed NIfTIs. The default is apply_morph=False.

    If desired the output volume can be of type ``Nifti2Image``. In that case
    set ``format='nifti2'``.

:meth:`my_morph.save() <mne.SourceMorph.save>`:

    Saves the morph object to disk. The only input argument is the filename.
    Since the object is stored in HDF5 ('.h5') format, the filename will be
    extended by '-morph.h5' if no file extension was initially defined.

Shortcuts:

   ``stc_fsaverage = SourceMorph(src=src)(stc)``

   ``img = SourceMorph(src=src)(stc, as_volume=True)``

.. _tut_sourcemorph_read:

Read SourceMorph from disk
--------------------------

:class:`mne.SourceMorph` objects can be read from disk using
:func:`mne.read_source_morph`. Provide the file name (absolute or relative
path) including the ``.h5`` extension:

``my_morph = mne.read_source_morph('my-morph.h5')``

See :ref:`tut_sourcemorph_methods` for how to use ``my_morph.save()`` to store
a repsective SourceMorph object.

.. _tut_sourcemorph_alternative:

Alternative API
===============

Some operations can be performed using the respective source estimate itself.
This is mostly to support the API of previous versions of MNE-Python.

Un-morphed :class:`mne.VolSourceEstimate` can be converted into NIfTI images
using :meth:`mne.VolSourceEstimate.as_volume`.

Un-morphed :class:`mne.SourceEstimate` and
:class:`mne.VectorSourceEstimate` can be morphed using
:meth:`mne.SourceEstimate.morph`.

Note that in any of the above cases :class:`mne.SourceMorph` will be used under
the hood to perform the requested operation.

.. _tut_sourcemorph_hands_on:

Step-by-step hands on tutorial
==============================

Note that a compact version of the following tutorial can be found
:ref:`here <sphx_glr_auto_tutorials_plot_morph.py>`.

In this tutorial we will morph different kinds of source estimates between
individual subject spaces using :class:`mne.SourceMorph`.

We will use precomputed data and morph surface and volume source estimates to a
common space. The common space of choice will be FreeSurfer's "fsaverage".

Furthermore we will convert our volume source estimate into a NIfTI image using
:meth:`mne.SourceMorph.as_volume`.
"""

###############################################################################
# Setup
# -----
# We first import the required packages and define a list of file names for
# various data sets we are going to use to run this tutorial.
import os

import matplotlib.pylab as plt
import nibabel as nib
from mne import (read_evokeds, SourceMorph, read_source_estimate)
from mne.datasets import sample
from mne.minimum_norm import apply_inverse, read_inverse_operator
from nilearn.image import index_img
from nilearn.plotting import plot_glass_brain

# We use the MEG and MRI setup from the MNE-sample dataset
sample_dir_raw = sample.data_path()
sample_dir = sample_dir_raw + '/MEG/sample'
subjects_dir = sample_dir_raw + '/subjects'

fname_evoked = sample_dir + '/sample_audvis-ave.fif'

fname_surf = os.path.join(sample_dir, 'sample_audvis-meg')
fname_vol = os.path.join(sample_dir,
                         'sample_audvis-grad-vol-7-fwd-sensmap-vol.w')

fname_inv_surf = os.path.join(sample_dir,
                              'sample_audvis-meg-eeg-oct-6-meg-eeg-inv.fif')
fname_inv_vol = os.path.join(sample_dir,
                             'sample_audvis-meg-vol-7-meg-inv.fif')

fname_t1_fsaverage = subjects_dir + '/fsaverage/mri/brain.mgz'

###############################################################################
# Data preparation
# ----------------
#
# First we load the respective example data for surface and volume source
# estimates. In order to save computation time we crop our time series to a
# short period around the peak time, that we already know. For a real case
# scenario this might apply as well if a narrow time window of interest is
# known in advance.

stc_surf = read_source_estimate(fname_surf, subject='sample')

# The surface source space
src_surf = read_inverse_operator(fname_inv_surf)['src']

# The volume inverse operator
inv_src = read_inverse_operator(fname_inv_vol)

# The volume source space
src_vol = inv_src['src']

# Ensure subject is not None
src_vol[0]['subject_his_id'] = 'sample'

# For faster computation we redefine tmin and tmax
stc_surf.crop(0.09, 0.1)  # our prepared surface source estimate

# Read pre-computed evoked data
evoked = read_evokeds(fname_evoked, condition=0, baseline=(None, 0))

# Apply inverse operator
stc_vol = apply_inverse(evoked, inv_src, 1.0 / 3.0 ** 2, "dSPM")

# For faster computation we redefine tmin and tmax
stc_vol.crop(0.09, 0.1)  # our prepared volume source estimate

###############################################################################
# Setting up SourceMorph for SourceEstimate
# -----------------------------------------
#
# As explained in :ref:`tut_surface_morphing` and :ref:`tut_sourcemorph_space`
# we have several options to instantiate
# :class:`mne.SourceMorph`. We know, that if src is not provided, the morph
# will not be pre-computed but instead will be prepared for morphing when
# calling the instance. This works only with (Vector)
# :class:`mne.SourceEstimate`.
# Below you will find a common setup that will apply to most use cases.

morph_surf = SourceMorph(subject_from='sample',  # Default: None
                         subject_to='fsaverage',  # Default
                         subjects_dir=subjects_dir,  # Default: None
                         src=None,  # Default
                         spacing=5,  # Default
                         smooth=None,  # Default
                         xhemi=False)  # Default

###############################################################################
# Setting up SourceMorph for VolSourceEstimate
# --------------------------------------------
#
# From :ref:`tut_volumetric_morphing` and :ref:`tut_sourcemorph` we know, that
# src has to be provided when morphing a :class:`mne.VolSourceEstimate`.
# Furthermore we can define the parameters of the in general very costly
# computation.
#
# Below an example was chosen, using a non-default spacing of
# isotropic 3 mm. The default is 5 mm and you will experience a noticeable
# difference in computation time, when changing this parameter. Ideally
# subject_from can be inferred from src, subject_to is 'fsaverage' by default
# and subjects_dir is set in the environment. In that case
# :class:`mne.SourceMorph` can be initialized taking only src as parameter.
#
# For demonstrative purposes all available keyword arguments were set
# nevertheless. We use the default optimization parameters, that is 100, 100
# and 10 iterations for the 1st, 2nd and 3rd level of optimization for all 3
# steps when computing the affine transform. In turn the Symmetric
# Diffeomorphic transformation will use 5, 5 and 3 iterations for each level
# respectively (see :ref:`tut_sourcemorph_opt`).

morph_vol = SourceMorph(subject_from='sample',  # Default: None
                        subject_to='fsaverage',  # Default
                        subjects_dir=subjects_dir,  # Default: None
                        spacing=(3., 3., 3.),  # Default: 5
                        src=src_vol,  # Default: None
                        niter_affine=(100, 100, 10),  # Default
                        niter_sdr=(5, 5, 3))  # Default

###############################################################################
# Applying an instance of SourceMorph
# -----------------------------------
#
# Once we computed the morph for our respective dataset, we can morph the data
# by giving it as an argument to the :class:`mne.SourceMorph` instance. This
# operation applies pre-computed transforms to stc or computes the morph if
# instantiated without providing :class:`src <mne.SourceSpaces>`.
#
# Default keyword arguments are valid for both types of morph. However,
# changing the default only makes real sense when morphing
# :class:`mne.VolSourceEstimate` See :ref:`tut_sourcemorph_methods` and below
# for more information. The morph will be applied to all sample points equally
# (time domain).

stc_surf_m = morph_surf(stc_surf)  # SourceEstimate | VectorSourceEstimate

stc_vol_m = morph_vol(stc_vol)  # VolSourceEstimate

###############################################################################
# Transforming VolSourceEstimate into NIfTI
# -----------------------------------------
#
# In case of a :class:`mne.VolSourceEstimate`, we can further ask SourceMorph
# to output a volume of our data in the new space. We do this by calling the
# :meth:`mne.SourceMorph.as_volume`. Note, that un-morphed source estimates
# still can be converted into a NIfTI by using
# :meth:`mne.VolSourceEstimate.as_volume`.
#
# The shape of the output volume can be modified by providing the argument
# *mri_resolution*. This argument can be boolean, a tuple or an int. If
# ``mri_resolution=True``, the MRI resolution, that was stored in ``src`` will
# be used. Setting mri_resolution to False, will export the volume having voxel
# size corresponding to the spacing of the computed morph. Setting a tuple or
# single value, will cause the output volume to expose a voxel size of that
# values in mm.
#
# We can play around with those parameters and see the difference.

# Create full MRI resolution output volume
img_mri_res = morph_vol.as_volume(stc_vol_m, mri_resolution=True)

# Create morph resolution output volume
img_morph_res = morph_vol.as_volume(stc_vol_m, mri_resolution=False)

# Create output volume of manually defined voxel size directly from SourceMorph
img_any_res = morph_vol(stc_vol,  # use un-morphed source estimate and
                        as_volume=True,  # output NIfTI with
                        mri_resolution=7,  # isotropic voxel size of 2mm
                        mri_space=True,  # in MRI space
                        apply_morph=True,  # after applying the morph
                        format='nifti1')  # in NIfTI 1 format

###############################################################################
# Plot results
# ------------
#
# After plotting the results it is worth paying attention to the grid size of
# the different results. We implicitly defined a voxel size of isotropic 1 mm
# for the first volume, by defining ``mri_resolution=True``, which happens to
# be 1 mm.
#
# The second volume has a voxel size of isotropic 3 mm, because computed the
# transformation based on this value. It corresponds to *spacing*
#
# Lastly we defined the *mri_resolution* manually setting it to isotropic 7 mm.
# The significantly increased voxel size pops directly out.

# Load fsaverage anatomical image
t1_fsaverage = nib.load(fname_t1_fsaverage)

# Initialize figure
fig, [axes1, axes2, axes3] = plt.subplots(1, 3)
fig.subplots_adjust(top=0.8, left=0.1, right=0.9, hspace=0.5)
fig.patch.set_facecolor('white')

# Plot morphed volume source estimate for all different resolutions
for axes, img, res in zip([axes1, axes2, axes3],
                          [img_mri_res, img_morph_res, img_any_res],
                          ['Full MRI\nresolution',
                           'Morph\nresolution',
                           'isotropic\n7 mm']):
    # Setup nilearn plotting
    display = plot_glass_brain(t1_fsaverage,
                               display_mode='x',
                               cut_coords=0,
                               draw_cross=False,
                               axes=axes,
                               figure=fig,
                               annotate=False)

    # Transform into volume time series and use first one
    overlay = index_img(img, 0)

    display.add_overlay(overlay, alpha=0.75)
    display.annotate(size=8)
    axes.set_title(res, color='black', fontsize=12)
plt.show()

# save some memory
del stc_vol_m, morph_vol, morph_surf, img_mri_res, img_morph_res, img_any_res

###############################################################################
# Plot morphed surface source estimate

surfer_kwargs = dict(
    hemi='lh', subjects_dir=subjects_dir,
    clim=dict(kind='value', lims=[8, 12, 15]), views='lateral',
    initial_time=0.09, time_unit='s', smoothing_steps=5)
brain = stc_surf_m.plot(**surfer_kwargs)
brain.add_text(0.1, 0.9, 'Morphed to fsaverage', 'title', font_size=16)

###############################################################################
# Summary
# =======
#
# Morphing is the process of transforming a representation in one space to
# another. This is particularly important for neuroimaging data, since
# individual differences across subject's brains have to be accounted.
#
# In MNE-Python, morphing is achieved using
# :class:`mne.SourceMorph`. This class can morph surface and volumetric source
# estimates alike.
#
# Instantiate a new object by calling and use the new instance to morph the
# data:
#
# ``morph = mne.SourceMorph(src=src)``
#
# ``stc_fsaverage = morph(stc)``
#
# Furthermore the data can be converted into a NIfTI image:
#
# ``img_fsaverage = morph.as_volume(stc_fsaverage)``
#
# References
# ==========
#
# .. [1] Gramfort, A., Luessi, M., Larson, E., Engemann, D. A., Strohmeier, D.,
#        Brodbeck, C., ... & Hämäläinen, M. (2013). MEG and EEG data analysis
#        with MNE-Python. Frontiers in neuroscience, 7, 267.
# .. [2] Avants, B. B., Epstein, C. L., Grossman, M., & Gee, J. C. (2009).
#        Symmetric diffeomorphic image registration with cross-correlation:
#        evaluating automated labeling of elderly and neurodegenerative brain.
#        Medical image analysis, 12(1), 26-41.
# .. [3] Viola, P., & Wells III, W. M. (1997). Alignment by maximization of
#        mutual information. International journal of computer vision, 24(2),
#        137-154.
# .. [4] Mattes, D., Haynor, D. R., Vesselle, H., Lewellen, T. K., & Eubank, W.
#        (2003). PET-CT image registration in the chest using free-form
#        deformations. IEEE transactions on medical imaging, 22(1), 120-128.
# .. _dipy example: http://nipy.org/dipy/examples_built/syn_registration_3d.html  # noqa
# .. _dipy affine example: http://nipy.org/dipy/examples_built/affine_registration_3d.html  # noqa
# .. _dipy sdr example: http://nipy.org/dipy/examples_built/syn_registration_3d.html  # noqa
# .. _Wikipedia article about transformation matrices: https://en.wikipedia.org/wiki/Transformation_matrix  # noqa
# .. _Jensen's inequality: https://en.wikipedia.org/wiki/Jensen%27s_inequality  # noqa
