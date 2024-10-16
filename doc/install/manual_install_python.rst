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
   Anaconda, the company that develops the Anaconda and Miniconda Python
   distributions, is now strictly enforcing its end-user license agreement.
   This means that if you're found using the software in a commercial setting,
   you may be required to purchase a license. If you're unsure about your
   usage, we highly recommend using Miniforge or our installers instead. These
   options, by default, avoid Anaconda packages and rely on community-created
   packages from the `conda-forge`_ project.

.. _other-py-distros:

Other Python distributions
^^^^^^^^^^^^^^^^^^^^^^^^^^

While conda-based CPython distributions provide many conveniences, other types of
installation (``pip`` / ``poetry``, ``venv`` / system-level) and/or other Python
distributions (PyPy) *should* also work with MNE-Python. Generally speaking, if you can
install SciPy, getting MNE-Python to work should be unproblematic. Note however that we
do not offer installation support for anything other than conda-based installations.
