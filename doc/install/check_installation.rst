.. _testing-installation:


Testing your installation
=========================

To make sure MNE-Python was installed correctly, type the following command in
a terminal::

    python -c "import mne; mne.sys_info()"

.. hint::
   If you installed MNE-Python using one of our installers, enter the above
   command in the **Prompt**.

This should display some system information along with the versions of
MNE-Python and its dependencies. Typical output looks like this::

    Platform                Windows-10-10.0.20348-SP0
    Python                  3.10.12 | packaged by conda-forge | (main, Jun 23 2023, 22:34:57) [MSC v.1936 64 bit (AMD64)]
    Executable              C:\Miniconda3\envs\mne\python.exe
    CPU                     Intel64 Family 6 Model 85 Stepping 7, GenuineIntel (2 cores)
    Memory                  7.0 GB

    Core
    ├☑ mne                  1.6.0.dev67+gb12384562
    ├☑ numpy                1.25.2 (OpenBLAS 0.3.23.dev with 1 thread)
    ├☑ scipy                1.11.2
    ├☑ matplotlib           3.7.2 (backend=QtAgg)
    ├☑ pooch                1.7.0
    └☑ jinja2               3.1.2

    Numerical (optional)
    ├☑ sklearn              1.3.0
    ├☑ nibabel              5.1.0
    ├☑ nilearn              0.10.1
    ├☑ dipy                 1.7.0
    ├☑ openmeeg             2.5.6
    ├☑ pandas               2.1.0
    └☐ unavailable          numba, cupy

    Visualization (optional)
    ├☑ pyvista              0.41.1 (OpenGL 3.3 (Core Profile) Mesa 10.2.4 (git-d92815a) via Gallium 0.4 on llvmpipe (LLVM 3.4, 256 bits))
    ├☑ pyvistaqt            0.0.0
    ├☑ ipyvtklink           0.2.2
    ├☑ vtk                  9.2.6
    ├☑ qtpy                 2.4.0 (PyQt5=5.15.8)
    ├☑ ipympl               0.9.3
    ├☑ pyqtgraph            0.13.3
    └☑ mne-qt-browser       0.5.2

    Ecosystem (optional)
    └☐ unavailable          mne-bids, mne-nirs, mne-features, mne-connectivity, mne-icalabel, mne-bids-pipeline


.. dropdown:: If you get an error...
    :color: danger
    :icon: alert-fill

    .. rubric:: If you see an error like:

    ::

        Traceback (most recent call last):
          File "<string>", line 1, in <module>
        ModuleNotFoundError: No module named 'mne'

    This suggests that your environment containing MNE-Python is not active.
    If you followed the setup for 3D plotting/source analysis (i.e., you
    installed to a new ``mne`` environment instead of the ``base`` environment)
    try running ``conda activate mne`` first, and try again. If this works,
    you might want to set your terminal to automatically activate the
    ``mne`` environment each time you open a terminal::

        echo conda activate mne >> ~/.bashrc    # for bash shells
        echo conda activate mne >> ~/.zprofile  # for zsh shells

If something else went wrong during installation and you can't figure it out,
check out the :ref:`advanced_setup` instructions to see if your problem is
discussed there. If not, the `MNE Forum`_ is a good resources for
troubleshooting installation problems.

.. highlight:: python

.. LINKS

.. _`pyvista`: https://docs.pyvista.org/
.. _`X server`: https://en.wikipedia.org/wiki/X_Window_System
.. _`xvfb`: https://en.wikipedia.org/wiki/Xvfb
