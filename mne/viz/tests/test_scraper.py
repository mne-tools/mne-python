# Authors: Eric Larson <larson.eric.d@gmail.com>
#
# License: Simplified BSD

import os.path as op
import pytest
import mne
from mne.utils import requires_version


@pytest.mark.pgtest
@requires_version('sphinx_gallery')
def test_qt_scraper(raw, pg_backend, tmp_path):
    """Test sphinx-gallery scraping of the browser."""
    # make sure there is only one to scrape from old tests
    fig = raw.plot(group_by='selection')
    (tmp_path / '_images').mkdir()
    image_paths = [
        str(tmp_path / '_images' / 'temp_{ii}.png') for ii in range(2)]
    gallery_conf = dict(builder_name='html', src_dir=str(tmp_path))
    block_vars = dict(
        example_globals=dict(fig=fig),
        image_path_iterator=iter(image_paths))
    assert not any(op.isfile(image_path) for image_path in image_paths)
    assert not getattr(fig, '_scraped', False)
    mne.viz._scraper._MNEQtBrowserScraper()(None, block_vars, gallery_conf)
    assert all(op.isfile(image_path) for image_path in image_paths)
    assert fig._scraped
