.. include:: ../links.inc

MNE-Python installers
=====================

🛠 About the installers
~~~~~~~~~~~~~~~~~~~~~~~

Since the release of MNE-Python 1.0, we provide standalone installers for
MNE-Python. These contain all required dependencies and many additional
helpful Python packages and tools.

.. hint::
    **MNE-Python installers are the easiest way to install MNE-Python, and
    recommended for all users.**

.. note::
   ⏰ Please understand that due to our limited resources, we currently cannot
   test on more than the specified platforms. If your favorite platform is
   missing, you can try to use the installers anyway; however, you'll be on
   entirely unexplored territory. If you would like us to expand platform
   support, please get in touch via the forum.

.. warning::

   The installers are very new, and this is the first time we're making them
   available to a larger group of users. Chances are that you **will**
   experience some kind of glitches. If you do, please send a report to the
   forum so we can fix the problem.

🧰 What will be installed
~~~~~~~~~~~~~~~~~~~~~~~~~

Aside from MNE-Python and all dependencies required for 2D and 3D
visualization, MVPA ("decoding"), and reading and writing various data
formats, the following software and Python packages will be installed:

* ``Spyder``, an easy-to-use development environment
* ``MNE-BIDS`` for BIDS data access
* ``MNE-Connectivity`` for connectivity analysis
* ``MNE-FASTER``, a pre-processing pipeline
* ``MNE-NIRS`` for working with fNIRS data
* ``MNE-Realtime`` for realtime data analysis
* ``MNE-Features`` for feature extraction
* ``MNE-RSA`` for representational similarity analysis
* ``MNE-Microstates`` for microstates analysis
* ``MNE-ARI`` for all-resolutions Inference
* ``autoreject`` for automated artifact rejection
* ``Jupyter Notebook`` and ``JupyterLab`` for interactive data analysis in the
  webbrowser
* ``Pingouin`` for statistics, including repeated-measures ANOVA
* ``pycircstat`` for circular statistics
* ``FSLeyes`` for visualization of MRI data
* ``dcm2niix`` for conversion of DICOM to NIfTI data
* ``pactools``, ``tensorpac``, ``emd``, ``neurodsp``, ``bycycle``, and
  ``fooof`` for time-frequency analysis
* ``NeuroKit2`` for analysis of various biological signals (ECG, EMG, …)
* ``openneuro-py`` for downloading datasets from OpenNeuro


|apple| Installer for macOS
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. raw:: html

    <div class="text-center pt-3">
        <a
            class="btn btn-primary font-weight-bold shadow-sm"
            role="button"
            href="https://github.com/hoechenberger/mne-installers/releases/download/v1.0.0-pre8/MNE-Python-1.0.0_0-macOS.pkg"
        >
            <i class="fa fa-cloud-download-alt"></i>&ensp;Download for macOS
        </a>
    </div>

Supported platforms
^^^^^^^^^^^^^^^^^^^

* macOS 10.15 (Catalina)
* macOS 12 (Monterey) on Intel- and Apple Silicon-based computers

Running the installer
^^^^^^^^^^^^^^^^^^^^^

Simply double-click on the downloaded installer package to start the installer.
Follow the interactive wizard, accepting all default values.

.. note::
   Installation may take several minutes to complete, even if the installer
   indicates it will only take "less than a minute". Please be patient. 🧘‍♀️

.. note::
    If your computer has an **Apple Silicon** chip (i.e., M1), you may be
    prompted to install Rosetta first. The reason for this is that currently,
    we don't ship "native" packages for Apple Silicon. Instead, the provided
    software is designed for Intel chips, and hence requires the Rosetta
    emulation to work on Apple Silicon. Simply confirm the installation
    request; macOS will install Rosetta and then automatically resume with
    MNE-Python installation. A native Apple Silicon build that doesn't require
    Rosetta will be provided in the future.

|linux| Installer for Linux
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. raw:: html

    <div class="text-center pt-3">
        <a
            class="btn btn-primary font-weight-bold shadow-sm"
            role="button"
            href="https://github.com/hoechenberger/mne-installers/releases/download/v1.0.0-pre8/MNE-Python-1.0.0_0-Linux.sh"
        >
            <i class="fa fa-cloud-download-alt"></i>&ensp;Download for Linux
        </a>
    </div>

Supported platforms
^^^^^^^^^^^^^^^^^^^

* Ubuntu Linux 18.04 LTS (Bionic Beaver)
* Ubuntu Linux 20.04 LTS (Focal Fossa)

Verifying package integrity
^^^^^^^^^^^^^^^^^^^^^^^^^^^

To ensure your downloaded file has not been corrupted
or tampered with, please calculate the SHA256 hash
of your download and compare it to the expected hash
we provide below.

To do that, open a terminal, navigate to the directory
containing the downloaded installer, and run

.. code-block:: shell

    shasum -a 256 MNE-Python-1.0.0_0-Linux.sh

If the output doesn't match the following hash exactly, do **not**
run the installer, but delete the file and download it
again:

``990e221f40deabfdffdb196adf2b78166a65eef2a31c8f0125872ae25172d812``

Running the installer
^^^^^^^^^^^^^^^^^^^^^

Open a terminal, navigate to the directory
containing the downloaded installer, and run

.. code-block:: shell

    sh ./MNE-Python-1.0.0_0-Linux.sh

Follow the interactive installation procedure, accepting the default values.
MNE-Python will be installed into an ``mne-python`` sub-directory inside your
home directory.


|windows| Installer for Windows
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. raw:: html

    <div class="text-center pt-3">
        <a
            class="btn btn-primary font-weight-bold shadow-sm"
            role="button"
            href="https://github.com/hoechenberger/mne-installers/releases/download/v1.0.0-pre8/MNE-Python-1.0.0_0-Windows.exe"
        >
            <i class="fa fa-cloud-download-alt"></i>&ensp;Download for Windows
        </a>
    </div>

Supported platforms
^^^^^^^^^^^^^^^^^^^

* Windows 10

Verifying package integrity
^^^^^^^^^^^^^^^^^^^^^^^^^^^

We currently cannot digitally sign the Windows installers, and hence you
should manually verify their integrity to ensure the downloaded file has not
been corrupted or tampered with.

To do that, please calculate the SHA256 hash of your download and compare it
to the expected hash we provide below.

The software `Hash Tool`_, which is freely available from the Microsoft
Store, can be used to easily calculate the hash. Select the ``SHA256`` hash
type, and then drag and drop the downloaded installer file into the application
window. After a few seconds, the hash should be displayed.

If the output doesn't match the following hash exactly, do **not**
run the installer, but delete the file and download it
again:

``8F7CD15F889452ED5896CC57AFE09D1BAF5D4A1C1C800E4098AE953BC9EA9C5B``

Running the installer
^^^^^^^^^^^^^^^^^^^^^

Double-click the installer executable. Because the installer is not signed,
Windows may display a warning dialog, stating that
``Windows protected your PC``. Click on ``More info`` and then on
``Run anyway`` to start the installer.  Follow the installation wizard,
accepting the default values.

.. warning::
   The warning dialog displayed by Windows is there for a reason. Please
   **only** dismiss it if you've successfully verified the hash of the
   installer, as described in the section above. Otherwise, you may be at risk
   of infecting your computer with malware!
