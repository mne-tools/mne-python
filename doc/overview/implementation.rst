.. _implementation:

Algorithms and other implementation details
===========================================

This page describes some of the technical details of MNE-Python implementation.

.. contents:: Page contents
   :local:
   :depth: 1


.. _units:

Internal representation (units)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. include:: ../_includes/units.rst
   :start-after: units-begin-content


.. _precision:

Floating-point precision
^^^^^^^^^^^^^^^^^^^^^^^^

.. include:: ../_includes/precision.rst
   :start-after: precision-begin-content


.. _channel-types:

Supported channel types
^^^^^^^^^^^^^^^^^^^^^^^

.. include:: ../_includes/channel_types.rst
   :start-after: channel-types-begin-content


.. _data-formats:

Supported data formats
^^^^^^^^^^^^^^^^^^^^^^

.. include:: ../_includes/data_formats.rst
   :start-after: data-formats-begin-content


.. _dig-formats:

Supported formats for digitized 3D locations
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. include:: ../_includes/dig_formats.rst
   :start-after: dig-formats-begin-content


.. _memory:

Memory-efficient I/O
^^^^^^^^^^^^^^^^^^^^

.. include:: ../_includes/memory.rst
   :start-after: memory-begin-content


.. _channel-interpolation:

Bad channel repair via interpolation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. include:: ../_includes/channel_interpolation.rst
   :start-after: channel-interpolation-begin-content
   :end-before: channel-interpolation-end-content


.. _maxwell:

Maxwell filtering
^^^^^^^^^^^^^^^^^

MNE-Python's implementation of Maxwell filtering is described in the
:ref:`tut-artifact-sss` tutorial.


.. _ssp-method:

Signal-Space Projection (SSP)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. include:: ../_includes/ssp.rst
   :start-after: ssp-begin-content


.. _bem-model:

The Boundary Element Model (BEM)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. include:: ../_includes/bem_model.rst
   :start-after: bem-begin-content


.. _ch_forward:

The forward solution
^^^^^^^^^^^^^^^^^^^^

.. include:: ../_includes/forward.rst
   :start-after: forward-begin-content
   :end-before: forward-end-content

.. _minimum_norm_estimates:

The minimum-norm current estimates
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. include:: ../_includes/inverse.rst
   :start-after: inverse-begin-content
   :end-before: inverse-end-content


.. _ch_morph:

Morphing and averaging source estimates
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. include:: ../_includes/morph.rst
   :start-after: morph-begin-content


References
^^^^^^^^^^
.. footbibliography::
