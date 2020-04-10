import os
import os.path as op

import pytest

from mne.utils import (_fetch_file, requires_good_network, catch_logging,
                       sizeof_fmt)


@pytest.mark.timeout(60)
@requires_good_network
@pytest.mark.parametrize('url', (
    'https://raw.githubusercontent.com/mne-tools/mne-python/master/README.rst',
))
def test_fetch_file(url, tmpdir):
    """Test URL retrieval."""
    tempdir = str(tmpdir)
    archive_name = op.join(tempdir, "download_test")
    with catch_logging() as log:
        _fetch_file(url, archive_name, timeout=30., verbose=True)
    log = log.getvalue()
    assert ', resuming at' not in log
    with open(archive_name, 'rb') as fid:
        data = fid.read()
    stop = len(data) // 2
    assert 0 < stop < len(data)
    with open(archive_name + '.part', 'wb') as fid:
        fid.write(data[:stop])
    with catch_logging() as log:
        _fetch_file(url, archive_name, timeout=30., verbose=True)
    log = log.getvalue()
    assert ', resuming at %s' % sizeof_fmt(stop) in log
    with pytest.raises(Exception, match='Cannot use'):
        _fetch_file('NOT_AN_ADDRESS', op.join(tempdir, 'test'), verbose=False)
    resume_name = op.join(tempdir, "download_resume")
    # touch file
    with open(resume_name + '.part', 'w'):
        os.utime(resume_name + '.part', None)
    _fetch_file(url, resume_name, resume=True, timeout=30.,
                verbose=False)
    with pytest.raises(ValueError, match='Bad hash value'):
        _fetch_file(url, archive_name, hash_='a', verbose=False)
    with pytest.raises(RuntimeError, match='Hash mismatch'):
        _fetch_file(url, archive_name, hash_='a' * 32, verbose=False)
