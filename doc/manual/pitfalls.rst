
.. contents:: Contents
   :local:
   :depth: 2

.. _pitfalls:

Pitfalls
########

Evoked Arithmetic
=================

Two evoked objects can be contrasted using::

	>>> evoked = evoked_cond1 - evoked_cond2

Note, however that the number of trials used to obtain the averages for
``evoked_cond1`` and ``evoked_cond2`` are taken into account when computing
``evoked``. That is, what you get is a weighted average, not a simple
element-by-element subtraction. To do a uniform (not weighted) average, use
the function :func:`mne.combine_evoked`.

Float64 vs float32
==================

MNE-Python performs all computation in memory using the double-precision
64-bit floating point format. This means that the data is typecasted into
`float64` format as soon as it is read into memory. The reason for this is
that operations such as filtering, preprocessing etc. are more accurate when
using the double-precision format. However, for backward compatibility, it
writes the `fif` files in a 32-bit format by default. This is advantageous
when saving data to disk as it consumes less space.

However, if the users save intermediate results to disk, they should be aware
that this may lead to loss in precision. The reason is that writing to disk is
32-bit by default and then typecasting to 64-bit does not recover the lost
precision. In case you would like to retain the 64-bit accuracy, there are two
possibilities:

* Chain the operations in memory and not save intermediate results
* Save intermediate results but change the ``dtype`` used for saving. However,
  this may render the files unreadable in other software packages
