"""Run tests for the BaseRaw class."""
# Author: Stefan Appelhoff <stefan.appelhoff@mailbox.org>
#
# License: BSD (3-clause)
import pytest

from mne import create_info
from mne.io import BaseRaw


def test_orig_units():
    """Test the error handling for original units."""
    # Should work fine
    info = create_info(ch_names=['Cz'], sfreq=100, ch_types='eeg')
    BaseRaw(info, last_samps=[1], orig_units={'Cz': 'uV'})

    # Should complain that channel Cz does not have a corresponding original
    # unit.
    with pytest.raises(ValueError, match='has no associated original unit.'):
        info = create_info(ch_names=['Cz'], sfreq=100, ch_types='eeg')
        BaseRaw(info, last_samps=[1], orig_units={'not_Cz': 'uV'})

    # Test that a non-dict orig_units argument raises a ValueError
    with pytest.raises(ValueError, match='orig_units must be of type dict'):
        info = create_info(ch_names=['Cz'], sfreq=100, ch_types='eeg')
        BaseRaw(info, last_samps=[1], orig_units=True)
