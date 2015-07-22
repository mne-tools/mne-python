

.. _ch_morph:

======================
Morphing and averaging
======================

.. contents:: Contents
   :local:
   :depth: 2

Overview
########

The spherical morphing of the surfaces accomplished by FreeSurfer can be
employed to bring data from different subjects into a common anatomical
frame. This chapter describes utilities which make use of the spherical morphing
procedure. mne_morph_labels morphs
label files between subjects allowing the definition of labels in
a one brain and transforming them to anatomically analogous labels
in another. mne_average_estimates offers
the capability to compute averages of data computed with the MNE software
across subjects.

.. _CHDJDHII:

The morphing maps
#################

The MNE software accomplishes morphing with help of morphing
maps which can be either computed on demand or precomputed using mne_make_morph_maps ,
see :ref:`CHDBBHDH`. The morphing is performed with help
of the registered spherical surfaces (``lh.sphere.reg`` and ``rh.sphere.reg`` )
which must be produced in FreeSurfer .
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
the corresponding elements on the *j* th row of :math:`M^{(AB)}` so that :math:`x_j^{(B)}` will
be a linear interpolation between the triangle vertex values reflecting
the location :math:`x_j^{(B)}` within the triangle :math:`T_j^{(A)}`.

It follows from the above definition that in general

.. math::    M^{(AB)} \neq (M^{(BA)})^{-1}\ ,

*i.e.*,

.. math::    x_{(A)} \neq M^{(BA)} M^{(AB)} x^{(A)}\ ,

even if

.. math::    x^{(A)} \approx M^{(BA)} M^{(AB)} x^{(A)}\ ,

*i.e.*, the mapping is *almost* a
bijection.

.. _CHDEBAHH:

About smoothing
###############

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

.. _CHDBBHDH:

Precomputing the morphing maps
##############################

The utility mne_make_morph_maps was
created to assist mne_analyze and mne_make_movie in
morphing. Since the morphing maps described above take a while to
compute, it is beneficial to construct all necessary maps in advance
before using mne_make_movie .
The precomputed morphing maps are located in ``$SUBJECTS_DIR/morph-maps`` . mne_make_morph_maps creates
this directory automatically if it does not exist. If this directory
exists when mne_analyze or mne_make_movie is run
and morphing is requested, the software first looks for already
existing morphing maps there. Also, if mne_analyze or mne_make_movie have
to recompute any morphing maps, they will be saved to ``$SUBJECTS_DIR/morph-maps`` if
this directory exists.

The names of the files in ``$SUBJECTS_DIR/morph-maps`` are
of the form:

 <*A*> - <*B*> -``morph.fif`` ,

where <*A*> and <*B*> are
names of subjects. These files contain the maps for both hemispheres,
and in both directions, *i.e.*, both :math:`M^{(AB)}` and :math:`M^{(BA)}`, as
defined above. Thus the files <*A*> - <*B*> -``morph.fif`` or <*B*> - <*A*> -``morph.fif`` are
functionally equivalent. The name of the file produced by mne_analyze or mne_make_movie depends
on the role of <*A*> and <*B*> in
the analysis.

If you choose to compute the morphing maps in batch in advance,
use :ref:`mne_make_morph_maps`.
