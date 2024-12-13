.. _ide_setup:

IDE integration (VSCode, Spyder, etc.)
======================================

Most users find it convenient to write and run their code in an `Integrated
Development Environment`_ (IDE). Some popular choices for scientific
Python development are:

- `Visual Studio Code`_ (often shortened to "VS Code" or "vscode") is a
  development-focused text editor that supports many programming languages in
  addition to Python, includes an integrated terminal console, and has a rich
  extension ecosystem. Installing
  `Microsoft's Python Extension
  <https://marketplace.visualstudio.com/items?itemName=ms-python.python>`__ is
  enough to get most Python users up and running. VS Code is free and
  open-source.

- `Spyder`_ is a free and open-source IDE developed by and for scientists who
  use Python. It can be installed via a
  `standalone Spyder installer <https://docs.spyder-ide.org/current/installation.html#downloading-and-installing>`__.
  To avoid dependency conflicts with Spyder, you should install ``mne`` in a
  separate environment, as explained in previous sections or using our dedicated
  installer. Then, instruct
  Spyder to use the MNE-Python interpreter by opening
  Spyder and `navigating to <https://docs.spyder-ide.org/current/faq.html#using-existing-environment>`__
  :samp:`Tools > Preferences > Python Interpreter > Use the following interpreter`.

- `PyCharm`_ is an IDE specifically for Python development that provides an
  all-in-one solution (no extension packages needed). PyCharm comes in a
  free and open-source Community edition as well as a paid Professional edition.

For these IDEs, you'll need to provide the path to the Python interpreter you want it
to use. If you're using the MNE-Python installers, on Linux and macOS opening the
**Prompt** will display several lines of information, including a line that will read
something like:

.. code-block:: output

   Using Python: /some/directory/mne-python_1.7.1_0/bin/python

Altertatively (or on Windows), you can find that path by opening the Python interpreter
you want to use (e.g., the one from the MNE-Python installer, or a ``conda`` environment
that you have activated) and running::

   >>> import sys
   >>> print(sys.executable) # doctest:+SKIP

This should print something like
``C:\Program Files\MNE-Python\1.7.0_0\bin\python.exe`` (Windows) or
``/Users/user/Applications/MNE-Python/1.7.0_0/.mne-python/bin/python`` (macOS).

For Spyder, if the console cannot start because ``spyder-kernels`` is missing,
install the required version in the conda environment. For example, with the
environment you want to use activated, run ``conda install spyder-kernels``.
