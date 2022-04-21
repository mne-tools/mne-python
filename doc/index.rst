.. title:: MNE

.. The page title must be in rST for it to show in next/prev page buttons.
   Therefore we add a special style rule to only this page that hides h1 tags

.. raw:: html

    <style type="text/css">h1 {display:none;}</style>

MNE-Python Homepage
===================

.. LOGO

.. raw:: html

    <script type="text/javascript">
    var observer = new MutationObserver(function(mutations) {
    const dark = document.documentElement.dataset.theme == 'dark';
    document.getElementsByClassName('mainlogo')[0].src = dark ? '_static/mne_logo_dark.svg' : "_static/mne_logo.svg";
    })
    observer.observe(document.documentElement, {attributes: true, attributeFilter: ['data-theme']});
    </script>
    <link rel="preload" href="_static/mne_logo_dark.svg" as="image">

.. image:: _static/mne_logo.svg
   :alt: MNE-Python
   :class: logo, mainlogo
   :align: center

.. rst-class:: h4 text-center font-weight-light my-4

   Open-source Python package for exploring, visualizing, and analyzing
   human neurophysiological data: MEG, EEG, sEEG, ECoG, NIRS, and more.

.. frontpage gallery is added by a conditional in _templates/layout.html

.. toctree::
   :hidden:

   Install <install/index>
   Documentation <overview/index>
   API Reference <python_reference>
   Get help <overview/get_help>
   Development <overview/development>
