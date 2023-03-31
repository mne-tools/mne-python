.. _configure-ide:

IDE Configuration Guide
=======================

.. include:: ../links.inc
.. highlight:: console

Thanks for taking the time to contribute! This guide steps through various IDE settings that are helpful when
contributing to MNE-Python.

Some of the basic settings include enabling linting, setting line margins, introspection, and docstring type:

* Linting - is the automated checking of your source code for programmatic and stylistic errors. Most of the time basic linting is already enabled but let's double check.

* Line margins - we have code style rules that forbid lines longer than 79 characters, so having a ruler at 79 characters helps coders adhere to that rule.

* Introspection - this helps removes trailing whitespaces that could be problematic when compiling or building

* Docstring - we use NumPy because it supports a combination of reStructured and GoogleDocstrings and supported by Sphinx


Spyder
------

* :menuselection:`Preferences --> Completion and Linting --> Linting --> Enable Basic Linting`
* :menuselection:`Preferences --> Completion and Linting --> Introspection --> Advanced -->Add "mne" to list of modules to preload for code completion`
* :menuselection:`Preferences --> Editor --> Show Vertical Lines at == 79`
* :menuselection:`Preferences --> Editor --> Advanced Settings --> Docstring Type == Numpy`
* :menuselection:`Preferences --> Editor --> Code Introspection/Analysis and “Automatically remove trailing spaces when saving files”`

PyCharm
-------

* :menuselection:`Settings/Preferences (Ctrl+Alt+S) --> Languages and Frameworks --> JavaScript --> Code Quality Tools --> Closure Linter > Enable`
* :menuselection:`View --> Tool Windows --> Python Packages --> Search (Enter Package) --> Select Required Version and press Install`
* :menuselection:`File --> Settings --> Editor --> Code Style --> General: Right Margin (columns) Change to 79`
* :menuselection:`File --> Settings --> Tools --> Python Integrated Tools --> Docstring = NumPy`
* :menuselection:`Settings --> Editors --> General --> scroll to On Save --> change Remove trailing spaces on: to ``Modified Lines```

VsCode
------

* :menuselection:`Command Palette (Ctrl + Shift + P) --> Lint --> Python: Enable/Disable Linting (click on Enable)`
* :menuselection:`Use ``python.autoComplete.preloadModules": ["numpy", "pandas", "matplotlib", "mne"]```
* :menuselection:`File --> Preferences --> Settings --> Select Tab Option (User or Workspace) --> Search for rulers --> setting.json --> add line "editor.rulers": [79]`
* :menuselection:`Preferences --> Keyboard Shortcuts --> Change autoDocstring.docstringFormat to NumPy`
* :menuselection:`File --> Preferences --> Settings --> User Settings Tab --> Click the open document icon --> add files.trimTrailingWhitespace: true to User Settings Document --> Save`