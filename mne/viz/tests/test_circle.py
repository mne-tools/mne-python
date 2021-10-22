# Authors: Alexandre Gramfort <alexandre.gramfort@inria.fr>
#          Denis Engemann <denis.engemann@gmail.com>
#          Martin Luessi <mluessi@nmr.mgh.harvard.edu>
#
# License: Simplified BSD


import numpy as np
import pytest
import matplotlib
import matplotlib.pyplot as plt

from mne.viz import (plot_connectivity_circle, circular_layout,
                     plot_channel_labels_circle)


def test_plot_connectivity_circle():
    """Test plotting connectivity circle."""
    node_order = ['frontalpole-lh', 'parsorbitalis-lh',
                  'lateralorbitofrontal-lh', 'rostralmiddlefrontal-lh',
                  'medialorbitofrontal-lh', 'parstriangularis-lh',
                  'rostralanteriorcingulate-lh', 'temporalpole-lh',
                  'parsopercularis-lh', 'caudalanteriorcingulate-lh',
                  'entorhinal-lh', 'superiorfrontal-lh', 'insula-lh',
                  'caudalmiddlefrontal-lh', 'superiortemporal-lh',
                  'parahippocampal-lh', 'middletemporal-lh',
                  'inferiortemporal-lh', 'precentral-lh',
                  'transversetemporal-lh', 'posteriorcingulate-lh',
                  'fusiform-lh', 'postcentral-lh', 'bankssts-lh',
                  'supramarginal-lh', 'isthmuscingulate-lh', 'paracentral-lh',
                  'lingual-lh', 'precuneus-lh', 'inferiorparietal-lh',
                  'superiorparietal-lh', 'pericalcarine-lh',
                  'lateraloccipital-lh', 'cuneus-lh', 'cuneus-rh',
                  'lateraloccipital-rh', 'pericalcarine-rh',
                  'superiorparietal-rh', 'inferiorparietal-rh', 'precuneus-rh',
                  'lingual-rh', 'paracentral-rh', 'isthmuscingulate-rh',
                  'supramarginal-rh', 'bankssts-rh', 'postcentral-rh',
                  'fusiform-rh', 'posteriorcingulate-rh',
                  'transversetemporal-rh', 'precentral-rh',
                  'inferiortemporal-rh', 'middletemporal-rh',
                  'parahippocampal-rh', 'superiortemporal-rh',
                  'caudalmiddlefrontal-rh', 'insula-rh', 'superiorfrontal-rh',
                  'entorhinal-rh', 'caudalanteriorcingulate-rh',
                  'parsopercularis-rh', 'temporalpole-rh',
                  'rostralanteriorcingulate-rh', 'parstriangularis-rh',
                  'medialorbitofrontal-rh', 'rostralmiddlefrontal-rh',
                  'lateralorbitofrontal-rh', 'parsorbitalis-rh',
                  'frontalpole-rh']
    label_names = ['bankssts-lh', 'bankssts-rh', 'caudalanteriorcingulate-lh',
                   'caudalanteriorcingulate-rh', 'caudalmiddlefrontal-lh',
                   'caudalmiddlefrontal-rh', 'cuneus-lh', 'cuneus-rh',
                   'entorhinal-lh', 'entorhinal-rh', 'frontalpole-lh',
                   'frontalpole-rh', 'fusiform-lh', 'fusiform-rh',
                   'inferiorparietal-lh', 'inferiorparietal-rh',
                   'inferiortemporal-lh', 'inferiortemporal-rh', 'insula-lh',
                   'insula-rh', 'isthmuscingulate-lh', 'isthmuscingulate-rh',
                   'lateraloccipital-lh', 'lateraloccipital-rh',
                   'lateralorbitofrontal-lh', 'lateralorbitofrontal-rh',
                   'lingual-lh', 'lingual-rh', 'medialorbitofrontal-lh',
                   'medialorbitofrontal-rh', 'middletemporal-lh',
                   'middletemporal-rh', 'paracentral-lh', 'paracentral-rh',
                   'parahippocampal-lh', 'parahippocampal-rh',
                   'parsopercularis-lh', 'parsopercularis-rh',
                   'parsorbitalis-lh', 'parsorbitalis-rh',
                   'parstriangularis-lh', 'parstriangularis-rh',
                   'pericalcarine-lh', 'pericalcarine-rh', 'postcentral-lh',
                   'postcentral-rh', 'posteriorcingulate-lh',
                   'posteriorcingulate-rh', 'precentral-lh', 'precentral-rh',
                   'precuneus-lh', 'precuneus-rh',
                   'rostralanteriorcingulate-lh',
                   'rostralanteriorcingulate-rh', 'rostralmiddlefrontal-lh',
                   'rostralmiddlefrontal-rh', 'superiorfrontal-lh',
                   'superiorfrontal-rh', 'superiorparietal-lh',
                   'superiorparietal-rh', 'superiortemporal-lh',
                   'superiortemporal-rh', 'supramarginal-lh',
                   'supramarginal-rh', 'temporalpole-lh', 'temporalpole-rh',
                   'transversetemporal-lh', 'transversetemporal-rh']

    group_boundaries = [0, len(label_names) / 2]
    node_angles = circular_layout(label_names, node_order, start_pos=90,
                                  group_boundaries=group_boundaries)
    con = np.random.RandomState(0).randn(68, 68)
    plot_connectivity_circle(con, label_names, n_lines=300,
                             node_angles=node_angles, title='test',
                             )

    pytest.raises(ValueError, circular_layout, label_names, node_order,
                  group_boundaries=[-1])
    pytest.raises(ValueError, circular_layout, label_names, node_order,
                  group_boundaries=[20, 0])
    plt.close('all')


@pytest.mark.filterwarnings('ignore:invalid value encountered in greater_equal'
                            ':RuntimeWarning')
def test_plot_channel_labels_circle():
    """Test plotting channel labels in a circle."""
    fig, axes = plot_channel_labels_circle(
        dict(brain=['big', 'great', 'smart']),
        colors=dict(big='r', great='y', smart='b'))
    texts = [child.get_text() for child in axes.get_children()
             if isinstance(child, matplotlib.text.Text)]
    for text in ('brain', 'big', 'great', 'smart'):
        assert text in texts
    # check inputs
    with pytest.raises(ValueError, match='No color provided'):
        plot_channel_labels_circle(
            dict(brain=['big', 'great', 'smart']),
            colors=dict(big='r', great='y'))
