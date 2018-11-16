MEGSIM
======
:func:`mne.datasets.megsim.load_data`

This dataset contains experimental and simulated MEG data. To load data from this dataset, do::

    from mne.io import Raw
    from mne.datasets.megsim import load_data
    raw_fnames = load_data(condition='visual', data_format='raw', data_type='experimental', verbose=True)
    raw = Raw(raw_fnames[0])

Detailed description of the dataset can be found in the related publication [1]_.

.. topic:: Examples

    * :ref:`sphx_glr_auto_examples_datasets_plot_megsim_data.py`
