.. _contributing:

IDE Configuration Guide
==================

.. include:: ../links.inc
.. highlight:: console

Thanks for taking the time to contribute! This guide steps through various IDE settings that are helpful when contributing to MNE-Python. 


| **Spyder**
* Preferences -> Completion and Linting -> Linting -> Enable Basic Linting 
* Preferences -> Completion and Linting -> Introspection -> Advanced -> Add "mne" to list of modules to preload for code completion
* Preferences -> Editor -> Show Vertical Lines at == 79
* Preferences -> Editor -> Advanced Settings -> Docstring Type == Numpy
* Preferences -> Editor -> Code Introspection/Analysis and “Automatically remove trailing spaces when saving files”
| **PyCharm**
* Settings/Preferences (Ctrl+Alt+S) -> Languages and Frameworks -> JavaScript -> Code Quality Tools -> Closure Linter > Enable
* View -> Tool Windows -> Python Packages -> Search (Enter Package) -> Select Required Version and press Install
* File -> Settings -> Editor -> Code Style -> General: Right Margin (columns) Change to 79
* File -> Settings -> Tools -> Python Integrated Tools -> Docstring = NumPy
* Settings -> Editors -> General -> scroll to On Save -> change Remove trailing spaces on: to ``Modified Lines`` 
| **VsCode**
* Command Palette (Ctrl + Shift + P) -> Lint -> Python: Enable/Disable Linting (click on Enable)
* Use ``python.autoComplete.preloadModules": ["numpy", "pandas", "matplotlib", "mne"]``,
* File -> Preferences -> Settings -> Select Tab Option (User or Workspace) -> Search for rulers -> setting.json -> add line "editor.rulers": [79]
* Preferences -> Keyboard Shortcuts -> Change autoDocstring.docstringFormat to NumPy
* File -> Preferences -> Settings -> User Settings Tab -> Click the open document icon -> add ``files.trimTrailingWhitespace: true`` to User Settings Document -> Save

 
