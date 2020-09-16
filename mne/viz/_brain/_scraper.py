import os.path as op

from ._brain import _Brain


class _BrainScraper(object):
    """Scrape Brain objects."""

    def __repr__(self):
        return '<BrainScraper>'

    def __call__(self, block, block_vars, gallery_conf):
        rst = ''
        for brain in block_vars['example_globals'].values():
            # Only need to process if it's a brain with a time_viewer
            # with traces on and shown in the same window, otherwise
            # PyVista and matplotlib scrapers can just do the work
            if (not isinstance(brain, _Brain)) or brain._closed:
                continue
            from matplotlib.image import imsave
            from sphinx_gallery.scrapers import figure_rst
            img_fname = next(block_vars['image_path_iterator'])
            img = brain.screenshot(time_viewer=True)
            assert img.size > 0
            imsave(img_fname, img)
            assert op.isfile(img_fname)
            rst += figure_rst(
                [img_fname], gallery_conf['src_dir'], brain._title)
            brain.close()
        return rst
