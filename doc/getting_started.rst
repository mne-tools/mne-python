:orphan:

.. include:: links.inc

.. _installation:

Installation
============

To get started with MNE, visit the installation instructions for
the :ref:`MNE<install_python_and_mne_python>`. You can optionally also
install :ref:`MNE-C <install_mne_c>`:

.. container:: row

  .. container:: panel panel-default halfpad

    .. container:: panel-heading nosize

      MNE python module

    .. container:: panel-body nosize

      .. toctree::
        :maxdepth: 2

        install_mne_python

  .. container:: panel panel-default halfpad

    .. container:: panel-heading nosize

      MNE-C

    .. container:: panel-body nosize

      .. toctree::
        :maxdepth: 2

        install_mne_c

.. container:: row

  .. container:: panel panel-default

    .. container:: panel-heading nosize

      Historical notes

    .. container:: panel-body nosize

      MNE started as tool written in C by Matti Hämäläinen while at MGH in
      Boston.

      - :ref:`MNE-C <c_reference>` is Matti's C code. Historically, MNE was
        a software package for computing cortically constrained Minimum Norm
        Estimates from MEG and EEG data.

      - The MNE python module was built in the Python programming language to
        reimplement all MNE-C’s functionality, offer transparent scripting,
        and extend MNE-C’s functionality considerably (see left). Thus it is
        the primary focus of this documentation.

      - :ref:`ch_matlab` is available mostly to allow reading and writing
        FIF files.

      - :ref:`mne_cpp`  aims to provide modular and open-source tools for
        real-time acquisition, visualization, and analysis. It provides
        a :ref:`separate website <mne_cpp>` for documentation and releases.

      The MNE tools are based on the FIF file format from Neuromag.
      However, MNE can read native CTF, BTI/4D, KIT and various
      EEG formats (see :ref:`IO functions <ch_convert>`).

      If you have been using MNE-C, there is no need to convert your fif
      files to a new system or database -- MNE works nicely with
      the historical fif files.
