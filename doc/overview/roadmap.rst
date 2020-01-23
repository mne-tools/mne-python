.. _roadmap:

Roadmap
=======

This page describes some of the major medium- to long-term goals for
MNE-Python. These are goals that require substantial effort and/or
API design considerations.

.. contents:: Page contents
   :local:
   :depth: 1

Clustering statistics API
^^^^^^^^^^^^^^^^^^^^^^^^^

The current clustering statistics has limited functionality. It should be
re-worked to create a new ``cluster_based_statistic`` or similar function.
In particular, the new API should:

1. Support mixed within- and between-subjects designs, different statistical
   functions, etc. This should be done via a ``design`` argument that mirrors
   :func:`patsy.dmatrices` or similar community standard (e.g., this is what
   is used by :class:`statsmodels.regression.linear_model.OLS`).
2. Have clear tutorials showing how different contrasts can be done (toy data).
3. Have clear tutorials showing some common analyses on real data (time-freq,
   sensor space, source space, etc.)
4. Not introduce any significant speed penalty (e.g., < 10% slower) compared
   to the existing, more specialized/limited functions.

3D visualization
^^^^^^^^^^^^^^^^

XXX remote viz, PyVista, etc.

We should also consider how we could integrate multiple functions as done
in ``mne_analyze``, e.g., simultaneous source estimate viewing, field map
viewing, head surface display, etc. These are all currently available in
separate functions, but we should be able to combine them in a single plot
as well.

2D visualization
^^^^^^^^^^^^^^^^

XXX code cruft, consistency, etc.

Tutorial / example overhaul
^^^^^^^^^^^^^^^^^^^^^^^^^^^

XXX only some are refactored / improved / consistent

XXX look at GSoC lists, etc. to list others

Coregistration / 3D viewer
^^^^^^^^^^^^^^^^^^^^^^^^^^

:ref:`gen_mne_coreg` is an excellent tool for coregistration, but is limited
by being tied to Mayavi, Traits, and TraitsUI. We should first refactor in
several separable steps:

1. Responsive code to use traitlets
2. GUI elements to use PyQt5 (rather than TraitsUI/pyface)
3. 3D plotting to use our 3D viz functions rather than Mayavi

XXX other ideas to follow...
