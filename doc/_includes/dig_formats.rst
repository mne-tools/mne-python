:orphan:

Supported formats for digitized 3D locations
============================================

.. NOTE: If you want to link to this content, link to :ref:`dig-formats`
   for the implementation page. The next line is
   a target for :start-after: so we can omit the title above:
   dig-formats-begin-content

MNE-Python can load 3D point locations obtained by digitization systems.
Such files allow to obtain a :class:`montage <mne.channels.DigMontage>`
that can then be added to :class:`~mne.io.Raw` objects with the
:meth:`~mne.io.Raw.set_montage`. See the documentation for each reader
function for more info on reading specific file types.

.. NOTE: To include only the table, here's a different target for :start-after:
   dig-formats-begin-table

.. cssclass:: table-bordered
.. rst-class:: midvalign

=============  =========  ==============================================
Vendor         Extension  MNE-Python function
=============  =========  ==============================================
Neuromag       .fif       :func:`mne.channels.read_dig_fif`

Polhemus       .hsp       :func:`mne.channels.read_dig_polhemus_isotrak`
ISOTRAK        .elp
               .eeg

EGI            .xml       :func:`mne.channels.read_dig_egi`

MNE            .hpts       :func:`mne.channels.read_dig_hpts`

Brain          .bvct      :func:`mne.channels.read_dig_captrack`
Products
=============  =========  ==============================================

It is also possible to make :class:`montage <mne.channels.DigMontage>`
from arrays with :func:`montage <mne.channels.make_dig_montage>`.
To load the Polhemus FastSCAN files you can use
:func:`montage <mne.channels.read_polhemus_fastscan>`.
