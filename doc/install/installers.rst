.. _installers:

MNE-Python installers
=====================

MNE-Python installers are the easiest way to install MNE-Python and
all dependencies. They also provide many additional
Python packages and tools. Got any questions? Let us know on the `MNE Forum`_!

Platform-specific installers
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. tab-set::
    :class: install-selector-tabset

    .. tab-item:: Linux
        :class-content: text-center
        :name: install-linux

        .. button-link:: https://github.com/mne-tools/mne-installers/releases/download/v1.9.0/MNE-Python-1.9.0_0-Linux.sh
            :ref-type: ref
            :color: primary
            :shadow:
            :class: font-weight-bold mt-3 install-download-button

            |cloud-arrow-down| |ensp| Download for Linux

        **Supported platforms:** Ubuntu 18.04 (Bionic Beaver) and newer

        Run the installer in a terminal via:

        .. code-block:: console

            $ sh ./MNE-Python-1.9.0_0-Linux.sh


    .. tab-item:: macOS (Intel)
        :class-content: text-center
        :name: install-macos-intel

        .. button-link:: https://github.com/mne-tools/mne-installers/releases/download/v1.9.0/MNE-Python-1.9.0_0-macOS_Intel.pkg
            :ref-type: ref
            :color: primary
            :shadow:
            :class: font-weight-bold mt-3 install-download-button

            |cloud-arrow-down| |ensp| Download for macOS (Intel)

        **Supported platforms:**
        macOS 10.15 (Catalina) and newer


    .. tab-item:: macOS (Apple Silicon)
        :class-content: text-center
        :name: install-macos-apple

        .. button-link:: https://github.com/mne-tools/mne-installers/releases/download/v1.9.0/MNE-Python-1.9.0_0-macOS_M1.pkg
            :ref-type: ref
            :color: primary
            :shadow:
            :class: font-weight-bold mt-3 install-download-button

            |cloud-arrow-down| |ensp| Download for macOS (Apple Silicon)


        **Supported platforms:**
        macOS 10.15 (Catalina) and newer

    .. tab-item:: Windows
        :class-content: text-center
        :name: install-windows

        .. button-link:: https://github.com/mne-tools/mne-installers/releases/download/v1.9.0/MNE-Python-1.9.0_0-Windows.exe
            :ref-type: ref
            :color: primary
            :shadow:
            :class: font-weight-bold mt-3 install-download-button

            |cloud-arrow-down| |ensp| Download for Windows

        **Supported platforms:** Windows 10 and newer

.. card::
    :class-body: text-center
    :class-card: install-download-alert hidden

    .. We have to use a button-link here because button-ref doesn't properly nested parse the inline code

    .. button-link:: ides.html
        :ref-type: ref
        :color: success
        :shadow:
        :class: font-weight-bold mt-3
        :click-parent:

        |rocket| Go to IDE Setup

    Once installation completes, **set up your IDE**!

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

- |code| |ensp| Set up **Visual Studio Code** or another IDE (:ref:`instructions here <ide_setup>`) to start writing your own analysis scripts right away, or to run one of our examples from this website.

  .. rst-class:: mt-3
- |desktop| |ensp| With **System Info**, list the versions of all installed MNE-Python-related packages.

  .. rst-class:: mt-3
- |terminal| |ensp| The **Prompt** drops you into a command line interface with a properly activated MNE-Python environment.


.. note::
   |hourglass-half| |ensp| Depending on your system, it may take a little while for these
   applications to start, especially on the very first run – which may take
   particularly long on Apple Silicon-based computers. Subsequent runs should
   usually be much faster.

Uninstallation
^^^^^^^^^^^^^^

To remove the MNE-Python distribution provided by our installers above:

1. Remove relevant lines from your shell initialization scripts if you
   added them at installation time. To do this, you can run from the MNE Prompt:

   .. code-block:: bash

       $ conda init --reverse

   Or you can manually edit shell initialization scripts, e.g., ``~/.bashrc`` or
   ``~/.bash_profile``.

2. Follow the instructions below to remove the MNE-Python conda installation for your platform:

   .. tab-set::
       :class: uninstall-selector-tabset

       .. tab-item:: Linux
           :name: uninstall-linux

           In a BASH terminal you can do:

           .. code-block:: bash

               $ which python
               /home/username/mne-python/1.9.0_0/bin/python
               $ rm -Rf /home/$USER/mne-python
               $ rm /home/$USER/.local/share/applications/mne-python-*.desktop

       .. tab-item:: macOS
           :name: uninstall-macos

           You can simply `drag the MNE-Python folder to the trash in the Finder <https://support.apple.com/en-us/102610>`__.

           Alternatively, you can do something like:

           .. code-block:: bash

               $ which python
               /Users/username/Applications/MNE-Python/1.9.0_0/.mne-python/bin/python
               $ rm -Rf /Users/$USER/Applications/MNE-Python  # if user-specific
               $ rm -Rf /Applications/MNE-Python              # if system-wide

       .. tab-item:: Windows
           :name: uninstall-windows

           To uninstall MNE-Python, you can remove the application using the `Windows Control Panel <https://support.microsoft.com/en-us/windows/uninstall-or-remove-apps-and-programs-in-windows-4b55f974-2cc6-2d2b-d092-5905080eaf98>`__.
