.. include:: ../links.inc

.. _installers:

MNE-Python installers
=====================

MNE-Python installers are the easiest way to install MNE-Python and
all dependencies. They also provide many additional
Python packages and tools, including the `Spyder`_ development environment.
Got any questions? Let us know on the `MNE Forum`_!

.. raw:: html
    :file: installers_platform_selector.html

First steps
^^^^^^^^^^^

The installer adds menu entries on Linux and Windows, and several application
bundles to the ``Applications`` folder on macOS.

.. rst-class:: list-unstyled
.. rst-class:: mx-5
.. rst-class:: mt-4
.. rst-class:: mb-5

- |code| |ensp| Use **MNE Spyder** to start writing your own analysis scripts right away, or to run one of our examples from this website.

  .. rst-class:: mt-3
- |desktop| |ensp| With **MNE System Info**, list the versions of all installed MNE-Python-related packages.

  .. rst-class:: mt-3
- |terminal| |ensp| The **MNE Prompt** drops you into a command line interface with a properly activated MNE-Python environment.


.. note::
   ⏳ Depending on your system, it may take a little while for these
   applications to start, especially on the very first run – which may take
   particularly long on Apple Silicon-based computers. Subsequent runs should
   usually be much faster.


VS Code Setup
^^^^^^^^^^^^^

If you want to use MNE-Python with `Visual Studio Code`_, you need to tell the
VS Code Python extension where to find the respective Python executable. To do
so, simply start the **MNE Prompt**. It will display several lines of
information, including a line that will read something like:

.. code-block::

   Using Python: /some/directory/mne-python_1.0.0_0/bin/python

This path is what you need to enter in VS Code when selecting the Python
interpreter.

.. note::
   This information is currently not displayed on the Windows platform.


.. |ensp| unicode:: U+2002 .. EN SPACE
