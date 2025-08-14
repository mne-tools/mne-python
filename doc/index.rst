:html_theme.sidebar_secondary.remove:

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
   :class: logo, mainlogo, only-light
   :align: center

.. image:: _static/mne_logo_dark.svg
   :alt: MNE-Python
   :class: logo, mainlogo, only-dark
   :align: center

.. rst-class:: h4 text-center font-weight-light my-4

   Open-source Python package for exploring, visualizing, and analyzing
   human neurophysiological data: MEG, EEG, sEEG, ECoG, NIRS, and more.

.. frontpage gallery is added by a conditional in _templates/layout.html

.. toctree::
   :hidden:

   Install <install/index>
   Documentation <documentation/index>
   API Reference <api/python_reference>
   Get Help <help/index>
   Development <development/index>
