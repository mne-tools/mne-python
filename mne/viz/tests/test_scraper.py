# Authors: Eric Larson <larson.eric.d@gmail.com>
#
# License: Simplified BSD

import mne
import os.path as op


def test_pg_scraper(raw, pg_backend, tmp_path):
    """Test sphinx-gallery scraping of the browser."""
    # make sure there is only one to scrape from old tests
    fig = raw.plot()
    (tmp_path / '_images').mkdir()
    image_path = str(tmp_path / '_images' / 'temp.png')
    gallery_conf = dict(builder_name='html', src_dir=str(tmp_path))
    block_vars = dict(
        example_globals=dict(fig=fig),
        image_path_iterator=iter([image_path]))
    assert not op.isfile(image_path)
    assert not getattr(fig, '_scraped', False)
    mne.viz._scraper._PyQtGraphScraper()(None, block_vars, gallery_conf)
    assert op.isfile(image_path)
    assert fig._scraped
