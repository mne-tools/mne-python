:orphan:

.. _install-python:

Installing Python
=================

MNE-Python requires Python and several Python packages. MNE-Python
version |version| requires Python version |min_python_version| or higher.

We recommend using a ``conda``-based Python installation, such as
`Anaconda`_, `Miniconda`_, or `Miniforge`_. For new users we recommend
our pre-built :ref:`installers`, which use ``conda`` environments under the hood.

.. warning::
   Anaconda Inc., the company that develops the Anaconda and Miniconda Python
   distributions,
   `changed their terms of service <https://www.anaconda.com/blog/update-on-anacondas-terms-of-service-for-academia-and-research>`__
   in March of 2024. If you're unsure about whether your usage situation requires a paid
   license, we recommend using Miniforge or our pre-built installer instead. These
   options, by default, install packages only from the community-maintained `conda-forge`_
   distribution channel, and avoid the distribution channels covered by Anaconda's terms
   of service.

.. _other-py-distros:

Other Python distributions
^^^^^^^^^^^^^^^^^^^^^^^^^^

While conda-based CPython distributions provide many conveniences, other types of
installation (``pip`` / ``poetry``, ``venv`` / system-level) and/or other Python
distributions (PyPy) *should* also work with MNE-Python. Generally speaking, if you can
install SciPy, getting MNE-Python to work should be unproblematic. Note however that we
do not offer installation support for anything other than conda-based installations.
