:orphan:

Floating-point precision
========================

MNE-Python performs all computation in memory using the double-precision 64-bit
floating point format. This means that the data is typecast into float64 format
as soon as it is read into memory. The reason for this is that operations such
as filtering and preprocessing are more accurate when using the 64-bit format.
However, for backward compatibility, MNE-Python writes :file:`.fif` files in a
32-bit format by default. This reduces file size when saving data to disk, but
beware that *saving intermediate results to disk and re-loading them from disk
later may lead to loss in precision*. If you would like to ensure 64-bit
precision, there are two possibilities:

- Chain the operations in memory and avoid saving intermediate results.

- Save intermediate results but change the :class:`~numpy.dtype` used for
  saving, by using the ``fmt`` parameter of :meth:`mne.io.Raw.save` (or
  :meth:`mne.Epochs.save`, etc). However, note that this may render the
  :file:`.fif` files unreadable in software packages other than MNE-Python.
