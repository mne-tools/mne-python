.. _installers:

MNE-Python installers
=====================

MNE-Python installers are the easiest way to install MNE-Python and
all dependencies. They also provide many additional
Python packages and tools, including the `Spyder`_ development environment.
Got any questions? Let us know on the `MNE Forum`_!

.. tab-set::
    :class: platform-selector-tabset

    .. tab-item:: Linux
        :class-content: text-center
        :name: linux-installers

        .. button-link:: https://github.com/mne-tools/mne-installers/releases/download/v1.6.1/MNE-Python-1.6.1_0-Linux.sh
            :ref-type: ref
            :color: primary
            :shadow:
            :class: font-weight-bold mt-3

            |cloud-arrow-down| |ensp| Download for Linux

        **Supported platforms:** Ubuntu 18.04 (Bionic Beaver) and newer

        Run the installer in a terminal via:

        .. code-block:: console

            $ sh ./MNE-Python-1.6.1_0-Linux.sh


    .. tab-item:: macOS (Intel)
        :class-content: text-center
        :name: macos-intel-installers

        .. button-link:: https://github.com/mne-tools/mne-installers/releases/download/v1.6.1/MNE-Python-1.6.1_0-macOS_Intel.pkg
            :ref-type: ref
            :color: primary
            :shadow:
            :class: font-weight-bold mt-3

            |cloud-arrow-down| |ensp| Download for macOS (Intel)

        **Supported platforms:**
        macOS 10.15 (Catalina) and newer


    .. tab-item:: macOS (Apple Silicon)
        :class-content: text-center
        :name: macos-apple-installers

        .. button-link:: https://github.com/mne-tools/mne-installers/releases/download/v1.6.1/MNE-Python-1.6.1_0-macOS_M1.pkg
            :ref-type: ref
            :color: primary
            :shadow:
            :class: font-weight-bold mt-3

            |cloud-arrow-down| |ensp| Download for macOS (Apple Silicon)


        **Supported platforms:**
        macOS 10.15 (Catalina) and newer

    .. tab-item:: Windows
        :class-content: text-center
        :name: windows-installers

        .. button-link:: https://github.com/mne-tools/mne-installers/releases/download/v1.6.1/MNE-Python-1.6.1_0-Windows.exe
            :ref-type: ref
            :color: primary
            :shadow:
            :class: font-weight-bold mt-3

            |cloud-arrow-down| |ensp| Download for Windows

        **Supported platforms:** Windows 10 and newer

.. raw:: html

   <script async="async" src="../_static/js/update_installer_version.js"></script>
   <script async="async" src="../_static/js/set_installer_tab.js"></script>

First steps
^^^^^^^^^^^

The installer adds menu entries on Linux and Windows, and several application
bundles to the ``Applications`` folder on macOS.

.. rst-class:: list-unstyled
.. rst-class:: mx-5
.. rst-class:: mt-4
.. rst-class:: mb-5

- |code| |ensp| Use **Spyder** to start writing your own analysis scripts right away, or to run one of our examples from this website.

  .. rst-class:: mt-3
- |desktop| |ensp| With **System Info**, list the versions of all installed MNE-Python-related packages.

  .. rst-class:: mt-3
- |terminal| |ensp| The **Prompt** drops you into a command line interface with a properly activated MNE-Python environment.


.. note::
   |hourglass-half| |ensp| Depending on your system, it may take a little while for these
   applications to start, especially on the very first run – which may take
   particularly long on Apple Silicon-based computers. Subsequent runs should
   usually be much faster.


VS Code Setup
^^^^^^^^^^^^^

If you want to use MNE-Python with `Visual Studio Code`_, you need to tell the
VS Code Python extension where to find the respective Python executable. To do
so, simply start the **Prompt**. It will display several lines of
information, including a line that will read something like:

.. code-block::

   Using Python: /some/directory/mne-python_1.6.1_0/bin/python

This path is what you need to enter in VS Code when selecting the Python
interpreter.

.. note::
   This information is currently not displayed on the Windows platform.
