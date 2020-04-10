from mne.utils import sizeof_fmt


def test_sizeof_fmt():
    """Test sizeof_fmt."""
    assert sizeof_fmt(0) == '0 bytes'
    assert sizeof_fmt(1) == '1 byte'
    assert sizeof_fmt(1000) == '1000 bytes'
