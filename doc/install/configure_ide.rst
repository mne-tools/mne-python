.. _configure-ide:

IDE Configuration Guide
=======================

.. include:: ../links.inc
.. highlight:: console

This page describes the most common IDE settings that are recommended for MNE-Python contributors.

Basic Settings
--------------
With the right configuration, using an Integrated Development Environment (IDE) can make contributing to MNE-Python much
easier. Basic settings include syntax highlighting, enabling linting, setting line margins, introspection, and docstring
type. Steps for each configuration differ with their respective IDE and versioning so the settings will only be reviewed
at a high level.


Syntax Highlighting
^^^^^^^^^^^^^^^^^^^
Syntax highlighting is a feature that determines the color and style of source code displayed in the IDE. Most users
find it helpful and is usually enabled by default. If not, search "python syntax highlight [name of IDE]" to learn how
to do it.

Linting
^^^^^^^
Linting is the automated checking of your source code for programmatic and stylistic errors. Most of the time, basic
linting is already enabled but it is a good idea to double-check. Search "python [name of IDE] linting" to learn how to
configure it.

Rulers
^^^^^^
We have code style rules that forbid lines longer than 79 characters, so having a ruler at 79 characters helps coders
adhere to that rule. Search "python line margins [name of IDE]" to learn how to configure it.

Code Completion
^^^^^^^^^^^^^^^
Code completion is a context-aware feature that speeds up the process of coding applications by reducing typos and other
common mistakes. Basic code completion helps with completing names of classes, methods, and more. Some IDE's
specifically catered towards Python have a default language server, but we recommend Jedi and Pylance if it is not
already configured. Simply search "code completion [name of IDE]" to learn how to configure it.

Line endings and Whitespaces
^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Fore whitespaces, make sure that whitespaces around classes/functions/methods and in docstrings
are taken care of. Removing trailing whitespaces that can be problematic when compiling or building helps reduce the
numbers of issues when debugging. Search "python trailing whitespaces [name of IDE]" to learn how to do it.

For lines, make sure you are using 4 spaces (not tabs) for indentation, line feed (LF) and not carriage return line feed
(CLRF), and remove trailing newlines at end of file (EOF). Search "python/git line formatting [name of IDE]" for more
information.

.. note:: In Git, CRLF and LF are used to indicate the end of a line of text in a file. This is important because
    different operating systems use different line endings. If the wrong line endings are used, the code may not be
    properly formatted and may not be readable by other developers. Therefore, it is important to understand the
    differences between CRLF and LF and how to use them in Git.

Docstring
^^^^^^^^^
A docstring is a string literal specified in source code that is used, like a comment, to document a specific segment of
code. PyDoc is the standard documentation module for Python and supported by Sphinx but we use NumPy because it supports
a combination of reStructured, which is used in most of our documentation, and GoogleDocstrings and supported is by
Sphinx. Search "python [name of IDE] docstring" to learn how to configure it.

Configuring Specific IDEs
-------------------------

.. note:: The IDE configurations recommended below are not catered towards specified versions of the IDE, but rather a
    general guide of where these settings may be found regardless of versioning.

Spyder
^^^^^^

For Spyder, most of the configuration changes mentioned above can be found within:

* :menuselection:`Tools --> Preferences --> {Completion and Linting/Editor}`

PyCharm
^^^^^^^

For PyCharm, most of the configuration changes mentioned above can be found within:

* :menuselection:`File --> Settings --> {Editor/Tools}`

Additional extensions can be downloaded from the marketplace, i.e. Closure (Linter).

VsCode
^^^^^^

For VsCode, most of the configuration changes mentioned above can be found within:

* :menuselection:`File --> Preferences --> {Settings/Keyboard Shortcuts}`

Additional extensions can be downloaded, i.e. Pylance (Python Language server if not already default).
