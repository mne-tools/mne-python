.. include:: ../links.inc

.. _install-python:

Installing Python
=================

MNE-Python runs within Python, and depends on several other Python packages.
Version |version| requires Python version |min_python_version| or higher. We
recommend the `Anaconda`_ distribution of Python, which comes with more than
250 scientific packages pre-bundled and includes the ``conda`` command line
tool for installing new packages and managing different package sets
("environments") for different projects.

To get started, follow the `installation instructions for Anaconda`_.

.. warning::
   If you have the ``PYTHONPATH`` or ``PYTHONHOME`` environment variables set,
   you may run into difficulty using Anaconda. See the
   `Anaconda troubleshooting guide`_ for more information. Note that it is
   easy to switch between ``conda``-managed Python installations and the system
   Python installation using the ``conda activate`` and ``conda deactivate``
   commands, so you may find that after adopting Anaconda it is possible
   (indeed, preferable) to leave ``PYTHONPATH`` and ``PYTHONHOME`` permanently
   unset.

When you are done, if you type the following commands in a command shell,
you should see outputs similar to the following (assuming you installed
conda to ``/home/user/anaconda3``)::

    $ conda --version && python --version
    conda 4.9.2
    Python 3.7.7 :: Anaconda, Inc.
    $ which python
    /home/user/anaconda3/bin/python
    $ which pip
    /home/user/anaconda3/bin/pip

.. collapse:: |hand-paper| If you get an error or these look incorrect...
    :class: danger

    .. rubric:: If you are on a |windows| Windows command prompt:

    Most of our instructions start with ``$``, which indicates
    that the commands are designed to be run from a ``bash`` command shell.

    Windows command prompts do not expose the same command-line tools as
    ``bash`` shells, so commands like ``which`` will not work. You can test
    your installation in Windows ``cmd.exe`` shells with ``where`` instead::

        > where python
        C:\Users\user\anaconda3\python.exe
        > where pip
        C:\Users\user\anaconda3\Scripts\pip.exe

    .. rubric:: If you see something like:

    ::

        conda: command not found

    It means that your ``PATH`` variable (what the system uses to find
    programs) is not set properly. In a correct installation, doing::

        $ echo $PATH
        ...:/home/user/anaconda3/bin:...

    Will show the Anaconda binary path (above) somewhere in the output
    (probably at or near the beginning), but the ``command not found`` error
    suggests that it is missing.

    On Linux or macOS, the installer should have put something
    like the following in your ``~/.bashrc`` or ``~/.bash_profile`` (or your
    ``.zprofile`` if you're using macOS Catalina or later, where the default
    shell is ``zsh``):

    .. code-block:: bash

        # >>> conda initialize >>>
        # !! Contents within this block are managed by 'conda init' !!
        __conda_setup= ...
        ...
        # <<< conda initialize <<<

    If this is missing, it is possible that you are not on the same shell that
    was used during the installation. You can verify which shell you are on by
    using the command::

        $ echo $SHELL

    If you do not find this line in the configuration file for the shell you
    are using (bash, zsh, tcsh, etc.), try running::

        conda init

    in your command shell. If your shell is not ``cmd.exe`` (Windows) or
    ``bash`` (Linux, macOS) you will need to pass the name of the shell to the
    ``conda init`` command. See ``conda init --help`` for more info and
    supported shells.

    You can also consult the Anaconda documentation and search for
    Anaconda install tips (`Stack Overflow`_ results are often helpful)
    to fix these or other problems when ``conda`` does not work.
