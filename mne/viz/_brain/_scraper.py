import os.path as op

import numpy as np

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
            img = brain.screenshot()
            assert img.size > 0
            if getattr(brain, 'time_viewer', None) is not None and \
                    brain.time_viewer.show_traces and \
                    not brain.time_viewer.separate_canvas:
                canvas = brain.time_viewer.mpl_canvas.fig.canvas
                canvas.draw_idle()
                # In theory, one of these should work:
                #
                # trace_img = np.frombuffer(
                #     canvas.tostring_rgb(), dtype=np.uint8)
                # trace_img.shape = canvas.get_width_height()[::-1] + (3,)
                #
                # or
                #
                # trace_img = np.frombuffer(
                #     canvas.tostring_rgb(), dtype=np.uint8)
                # size = time_viewer.mpl_canvas.getSize()
                # trace_img.shape = (size.height(), size.width(), 3)
                #
                # But in practice, sometimes the sizes does not match the
                # renderer tostring_rgb() size. So let's directly use what
                # matplotlib does in lib/matplotlib/backends/backend_agg.py
                # before calling tobytes():
                trace_img = np.asarray(
                    canvas.renderer._renderer).take([0, 1, 2], axis=2)
                # need to slice into trace_img because generally it's a bit
                # smaller
                delta = trace_img.shape[1] - img.shape[1]
                if delta > 0:
                    start = delta // 2
                    trace_img = trace_img[:, start:start + img.shape[1]]
                img = np.concatenate([img, trace_img], axis=0)
            imsave(img_fname, img)
            assert op.isfile(img_fname)
            rst += figure_rst(
                [img_fname], gallery_conf['src_dir'], brain._title)
            brain.close()
        return rst
