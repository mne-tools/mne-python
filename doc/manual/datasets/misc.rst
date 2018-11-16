Miscellaneous Datasets
======================
These datasets are used for specific purposes in the documentation and in
general are not useful for separate analyses.

ECoG Dataset
^^^^^^^^^^^^
:func:`mne.datasets.misc.data_path`. Data exists at ``/ecog/sample_ecog.mat``.

This dataset contains a sample Electrocorticography (ECoG) dataset. It includes
a single grid of electrodes placed over the temporal lobe during an auditory
listening task. This dataset is primarily used to demonstrate visualization
functions in MNE and does not contain useful metadata for analysis.

.. topic:: Examples

    * :ref:`How to convert 3D electrode positions to a 2D image.
      <sphx_glr_auto_examples_visualization_plot_3d_to_2d.py>`: Demonstrates
      how to project a 3D electrode location onto a 2D image, a common procedure
      in electrocorticography.
