import os.path as op

from mne import fiff
from mne.layouts import read_layout
from mne.viz import plot_topo


fname = op.join(op.dirname(__file__), '..', 'fiff', 'tests', 'data',
                'test-ave.fif')

# Reading
evoked = fiff.read_evoked(fname, setno=0, baseline=(None, 0))


def test_plot_topo():
    """Test plotting of ERP topography
    """

    layout = read_layout('Vectorview-all')

    # Show topography
    plot_topo(evoked, layout)