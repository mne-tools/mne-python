.. _configure-ide:

IDE Configuration Guide
=======================

.. include:: ../links.inc
.. highlight:: console

This guide steps through various IDE settings that are helpful when contributing to MNE-Python.

Basic Settings
--------------

Some of the basic settings include enabling linting, setting line margins, introspection, and docstring type:

* Linting is the automated checking of your source code for programmatic and stylistic errors. Most of the time, basic linting is already enabled but it is a good idea to double check.

* Line margins - we have code style rules that forbid lines longer than 79 characters, so having a ruler at 79 characters helps coders adhere to that rule.

* Introspection helps by removing trailing whitespaces that can be problematic when compiling or building.

* Docstring is a string literal specified in source code that is used, like a comment, to document a specific segment of code. PyDoc is the standard documentation module for Python and supported by Sphinx but we use NumPy because it supports a combination of reStructured, which is used in most of our documentation, and GoogleDocstrings and supported is by Sphinx.

IDE Settings
------------

.. note:: The IDE configurations recommended below are catered towards specified versions of the IDE. If the recommended features are not found, check the versioning of your IDE.

Spyder (v.5.4.2)
^^^^^^^^^^^^^^^^

* :menuselection:`Tools --> Preferences --> Completion and Linting --> Code Style and Formatting --> Enable Code Style Linting`
* :menuselection:`Tools --> Preferences --> Completion and Linting --> Introspection --> Advanced -->Add "mne" to list of modules to preload for code completion`
* :menuselection:`Tools --> Preferences --> Completion and Linting --> Code Style and Formatting --> Line Length --> Show Vertical Lines at == 79`
* :menuselection:`Tools --> Preferences --> Completion and Linting --> Enable docstring style linting and choose the numpy convention`
* :menuselection:`Tools -->Preferences --> Editor --> Source Code --> Automatically remove trailing spaces when saving files.`
* :menuselection:`Tools -->Preferences --> Editor --> Source Code --> Insert a newline at the end if one does not exist when saving a file.`
* :menuselection:`Tools -->Preferences --> Editor --> Source Code --> Trim all newlines after the final one when saving a file.`
* :menuselection:`Tools -->Preferences --> Editor --> Advanced Settings --> Convert end-of-line character to the following on save -> LF(Unix)`

PyCharm (v.2023.1 Community)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* :menuselection:`View --> Tool Windows --> Python Packages --> Search (Enter Package) --> Select Required Version and press Install`
* :menuselection:`File --> Settings --> Editor --> Code Style --> Python --> Wrapping and Braces --> Hard Wrap At --> Change to 79`
* :menuselection:`File --> Settings --> Tools --> Python Integrated Tools --> Docstring Format = NumPy`
* :menuselection:`File --> Settings --> Editors --> General --> Scroll to On Save --> Remove trailing spaces: ``Modified Lines```

VsCode
^^^^^^

* :menuselection:`Command Palette (Ctrl + Shift + P) --> Lint --> Python: Enable/Disable Linting (click on Enable)`
* :menuselection:`Use ``python.autoComplete.preloadModules": ["numpy", "pandas", "matplotlib", "mne"]```
* :menuselection:`File --> Preferences --> Settings --> Select Tab Option (User or Workspace) --> Search for rulers --> setting.json --> add line "editor.rulers": [79]`
* :menuselection:`Preferences --> Keyboard Shortcuts --> Change autoDocstring.docstringFormat to NumPy`
* :menuselection:`File --> Preferences --> Settings --> User Settings Tab --> Click the open document icon --> add files.trimTrailingWhitespace: true to User Settings Document --> Save`