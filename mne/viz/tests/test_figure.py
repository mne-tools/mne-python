# Authors: Daniel McCloy <dan@mccloy.info>
#
# License: Simplified BSD

import pytest
from mne.viz._figure import _get_browser


def test_browse_figure_constructor():
    """Test error handling in MNEBrowseFigure constructor."""
    with pytest.raises(TypeError, match='an instance of Raw, Epochs, or ICA'):
        _get_browser(show=False, block=False, inst='foo')
