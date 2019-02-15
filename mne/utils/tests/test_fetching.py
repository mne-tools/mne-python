import os
import os.path as op

import pytest

from mne.utils import _fetch_file, requires_good_network


@requires_good_network
@pytest.mark.parametrize('url', ('https://www.github.com',))
def test_fetch_file(url, tmpdir):
    """Test URL retrieval."""
    tempdir = str(tmpdir)
    archive_name = op.join(tempdir, "download_test")
    _fetch_file(url, archive_name, timeout=30., verbose=False,
                resume=False)
    pytest.raises(Exception, _fetch_file, 'NOT_AN_ADDRESS',
                  op.join(tempdir, 'test'), verbose=False)
    resume_name = op.join(tempdir, "download_resume")
    # touch file
    with open(resume_name + '.part', 'w'):
        os.utime(resume_name + '.part', None)
    _fetch_file(url, resume_name, resume=True, timeout=30.,
                verbose=False)
    pytest.raises(ValueError, _fetch_file, url, archive_name,
                  hash_='a', verbose=False)
    pytest.raises(RuntimeError, _fetch_file, url, archive_name,
                  hash_='a' * 32, verbose=False)
