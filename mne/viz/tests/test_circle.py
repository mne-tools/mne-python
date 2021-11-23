# Authors: Alexandre Gramfort <alexandre.gramfort@inria.fr>
#          Denis Engemann <denis.engemann@gmail.com>
#          Martin Luessi <mluessi@nmr.mgh.harvard.edu>
#
# License: Simplified BSD


import pytest
import matplotlib

from mne.viz import plot_channel_labels_circle


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
