# Authors: Daniel McCloy <dan@mccloy.info>
#
# License: Simplified BSD

import pytest
from mne.viz._figure import _browse_figure, _psd_figure


def test_browse_figure_constructor():
    """Test error handling in MNEBrowseFigure constructor."""
    with pytest.raises(TypeError, match='an instance of Raw, Epochs, or ICA'):
        _browse_figure('foo')


def test_psd_figure_constructor():
    """Test error handling in MNELineFigure constructor."""
    with pytest.raises(TypeError, match='an instance of Raw or Epochs, got'):
        _psd_figure('foo', *((None,) * 19))
