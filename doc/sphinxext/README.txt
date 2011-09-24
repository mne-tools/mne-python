===================
 Sphinx Extensions
===================

We've copied these sphinx extensions over from nipy-core.  Any edits
should be done upstream in nipy-core, not here in nipype!

These are a few sphinx extensions we are using to build the nipy
documentation.  In this file we list where they each come from, since we intend
to always push back upstream any modifications or improvements we make to them.

It's worth noting that some of these are being carried (as copies) by more
than one project.  Hopefully once they mature a little more, they will be
incorproated back into sphinx itself, so that all projects can use a common
base.

* From numpy:
  * docscrape.py
  * docscrape_sphinx.py
  * numpydoc.py

* From matplotlib:
  * inheritance_diagram.py
  * ipython_console_highlighting.py
  * only_directives.py
