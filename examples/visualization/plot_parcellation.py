# -*- coding: utf-8 -*-
"""
============================
Plot a cortical parcellation
============================

In this example, we download the HCP-MMP1.0 parcellation [1]_ and show it
on fsaverage.

.. note:: The HCP-MMP dataset has license terms restricting its use.
          Of particular relevance:

              "I will acknowledge the use of WU-Minn HCP data and data
              derived from WU-Minn HCP data when publicly presenting any
              results or algorithms that benefitted from their use."

References
----------
.. [1] Glasser MF et al. (2016) A multi-modal parcellation of human
       cerebral cortex. Nature 536:171-178.
"""
# Author: Eric Larson <larson.eric.d@gmail.com>
#
# License: BSD (3-clause)

from surfer import Brain

import mne

subjects_dir = mne.datasets.sample.data_path() + '/subjects'
mne.datasets.fetch_hcp_mmp_parcellation(subjects_dir=subjects_dir,
                                        verbose=True)
labels = mne.read_labels_from_annot(
    'fsaverage', 'HCPMMP1', 'lh', subjects_dir=subjects_dir)

brain = Brain('fsaverage', 'lh', 'inflated', subjects_dir=subjects_dir,
              cortex='low_contrast', background='white', size=(800, 600))
brain.add_annotation('HCPMMP1')
aud_label = [label for label in labels if label.name == 'L_A1_ROI-lh'][0]
brain.add_label(aud_label, borders=False)
