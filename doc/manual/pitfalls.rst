
.. contents:: Contents
   :local:
   :depth: 2

.. _pitfalls:

Pitfalls
########

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
