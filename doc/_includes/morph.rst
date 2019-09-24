:orphan:

Morphing and averaging source estimates
=======================================

The spherical morphing of BEM surfaces accomplished by FreeSurfer can be
employed to bring data from different subjects into a common anatomical frame.
This page describes utilities which make use of the spherical :term:`morphing`
procedure. :func:`mne.morph_labels` morphs label files between subjects
allowing the definition of labels in a one brain and transforming them to
anatomically analogous labels in another. :meth:`mne.SourceMorph.apply` offers
the capability to transform all subject data to the same space and,
e.g., compute averages of data across subjects.

.. contents:: Page contents
   :local:
   :depth: 2


.. NOTE: part of this file is included in doc/overview/implementation.rst.
   Changes here are reflected there. If you want to link to this content, link
   to :ref:`ch_morph` to link to that section of the implementation.rst page.
   The next line is a target for :start-after: so we can omit the title from
   the include:
   morph-begin-content


Why morphing?
~~~~~~~~~~~~~

.. sidebar:: Morphing examples in MNE-Python

   Examples of morphing in MNE-Python include :ref:`this tutorial
   <tut-mne-fixed-free>` on surface source estimation or these examples on
   :ref:`surface <ex-morph-surface>` and :ref:`volumetric <ex-morph-volume>`
   source estimation.

Modern neuroimaging techniques, such as source reconstruction or fMRI analyses,
make use of advanced mathematical models and hardware to map brain activity
patterns into a subject-specific anatomical brain space. This enables the study
of spatio-temporal brain activity. The representation of spatio-temporal brain
data is often mapped onto the anatomical brain structure to relate functional
and anatomical maps. Thereby activity patterns are overlaid with anatomical
locations that supposedly produced the activity. Anatomical MR images are often
used as such or are transformed into an inflated surface representations to
serve as  "canvas" for the visualization.

In order to compute group-level statistics, data representations across
subjects must be morphed to a common frame, such that anatomically and
functional similar structures are represented at the same spatial location for
*all subjects equally*. Since brains vary, :term:`morphing` comes into play to
tell us how the data produced by subject A would be represented on the brain of
subject B (and vice-versa).


The morphing maps
~~~~~~~~~~~~~~~~~

The MNE software accomplishes morphing with help of morphing maps.
The morphing is performed with help of the registered
spherical surfaces (``lh.sphere.reg`` and ``rh.sphere.reg`` ) which must be
produced in FreeSurfer. A morphing map is a linear mapping from cortical
surface values in subject A (:math:`x^{(A)}`) to those in another subject B
(:math:`x^{(B)}`)

.. math::    x^{(B)} = M^{(AB)} x^{(A)}\ ,

where :math:`M^{(AB)}` is a sparse matrix with at most three nonzero elements
on each row. These elements are determined as follows. First, using the aligned
spherical surfaces, for each vertex :math:`x_j^{(B)}`, find the triangle
:math:`T_j^{(A)}` on the spherical surface of subject A which contains the
location :math:`x_j^{(B)}`. Next, find the numbers of the vertices of this
triangle and set the corresponding elements on the *j* th row of
:math:`M^{(AB)}` so that :math:`x_j^{(B)}` will be a linear interpolation
between the triangle vertex values reflecting the location :math:`x_j^{(B)}`
within the triangle :math:`T_j^{(A)}`.

It follows from the above definition that in general

.. math::    M^{(AB)} \neq (M^{(BA)})^{-1}\ ,

*i.e.*,

.. math::    x_{(A)} \neq M^{(BA)} M^{(AB)} x^{(A)}\ ,

even if

.. math::    x^{(A)} \approx M^{(BA)} M^{(AB)} x^{(A)}\ ,

*i.e.*, the mapping is *almost* a bijection.


About smoothing
~~~~~~~~~~~~~~~

The current estimates are normally defined only in a decimated grid which is a
sparse subset of the vertices in the triangular tessellation of the cortical
surface. Therefore, any sparse set of values is distributed to neighboring
vertices to make the visualized results easily understandable. This procedure
has been traditionally called smoothing but a more appropriate name might be
smudging or blurring in accordance with similar operations in image processing
programs.

In MNE software terms, smoothing of the vertex data is an iterative procedure,
which produces a blurred image :math:`x^{(N)}` from the original sparse image
:math:`x^{(0)}` by applying in each iteration step a sparse blurring matrix:

.. math::    x^{(p)} = S^{(p)} x^{(p - 1)}\ .

On each row :math:`j` of the matrix :math:`S^{(p)}` there are :math:`N_j^{(p -
1)}` nonzero entries whose values equal :math:`1/N_j^{(p - 1)}`. Here
:math:`N_j^{(p - 1)}` is the number of immediate neighbors of vertex :math:`j`
which had non-zero values at iteration step :math:`p - 1`. Matrix
:math:`S^{(p)}` thus assigns the average of the non-zero neighbors as the new
value for vertex :math:`j`. One important feature of this procedure is that it
tends to preserve the amplitudes while blurring the surface image.

Once the indices non-zero vertices in :math:`x^{(0)}` and the topology of the
triangulation are fixed the matrices :math:`S^{(p)}` are fixed and independent
of the data. Therefore, it would be in principle possible to construct a
composite blurring matrix

.. math::    S^{(N)} = \prod_{p = 1}^N {S^{(p)}}\ .

However, it turns out to be computationally more effective to do blurring with
an iteration. The above formula for :math:`S^{(N)}` also shows that the
smudging (smoothing) operation is linear.
