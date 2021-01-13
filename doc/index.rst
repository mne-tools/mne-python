.. title:: MNE

.. The page title must be in rST for it to show in next/prev page buttons.
   Therefore we add a special style rule to only this page that hides h1 tags

.. raw:: html

    <style type="text/css">h1 {display:none;}</style>

MNE-Python Homepage
===================

.. LOGO

.. image:: _static/mne_logo.svg
   :alt: MNE-Python
   :class: logo
   :align: center

Open-source Python package for exploring, visualizing, and
analyzing human neurophysiological data: MEG, EEG, sEEG, ECoG, NIRS, and more.


.. CAROUSEL

.. include:: carousel.inc


.. SIDEBAR (WHAT'S NEW)

.. raw:: html

  <div class="card mt-4">
    <h5 class="card-header">

Version |version|

.. raw:: html

    </h5>
    <div class="card-body">

.. rst-class:: list-group list-group-flush version-box

- |fw-newspaper| :ref:`Changelog <whats_new>`
- |fw-book| :ref:`Documentation <documentation_overview>`
- |fw-question-circle| :ref:`Get help <help>`
- |fw-quote-left| :ref:`Cite <cite>`
- |fw-code-branch| :ref:`Contribute <contributing>`

.. raw:: html

    </div>
  </div>

.. SIDEBAR (FUNDERS)

.. raw:: html

  <div class="card mt-4">
    <h5 class="card-header">Direct financial support</h5>
    <div class="card-body">

.. rst-class:: list-group list-group-flush funders

- |nih| **National Institutes of Health:**
  **R01**-EB009048, EB009048, EB006385, HD40712, NS44319, NS37462, NS104585,
  **P41**-EB015896, RR14075-06
- |nsf| **US National Science Foundation:** 0958669, 1042134
- |erc| **European Research Council:** **YStG**-263584, 676943
- |doe| **US Department of Energy:** **MIND** DE-FG02-99ER62764
- |anr| **Agence Nationale de la Recherche:**
  `14-NEUC-0002-01 <https://anr.fr/Project-ANR-14-NEUC-0002>`__,
  **IDEX** Paris-Saclay `11-IDEX-0003-02 <https://anr.fr/ProjetIA-11-IDEX-0003>`__

- |cds| **Paris-Saclay Center for Data Science:**
  `PARIS-SACLAY <http://www.datascience-paris-saclay.fr>`__
- |goo| **Google:** Summer of code (×6)
- |ama| **Amazon:** AWS Research Grants
- |czi| **Chan Zuckerberg Initiative:**
  `Essential Open Source Software for Science <https://chanzuckerberg.com/eoss/proposals/improving-usability-of-core-neuroscience-analysis-tools-with-mne-python>`__

.. raw:: html

    </div>
  </div>

.. |nih| image:: _static/funding/nih.png
.. |nsf| image:: _static/funding/nsf.png
.. |erc| image:: _static/funding/erc.svg
.. |doe| image:: _static/funding/doe.svg
.. |anr| image:: _static/funding/anr.svg
.. |cds| image:: _static/funding/cds.png
.. |goo| image:: _static/funding/google.svg
.. |ama| image:: _static/funding/amazon.svg
.. |czi| image:: _static/funding/czi.svg

.. toctree::
   :hidden:

   Install<install/index>
   Documentation<overview/index>
   Tutorials<auto_tutorials/index>
   Examples<auto_examples/index>
   glossary
   API<python_reference>
   Help<overview/get_help>
