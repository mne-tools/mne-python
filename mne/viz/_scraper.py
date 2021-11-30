# Authors: Eric Larson <larson.eric.d@gmail.com>
#
# License: Simplified BSD


from ..utils import _pl


class _PyQtGraphScraper:

    def __repr__(self):
        return '<PyQtGraphScraper>'

    def __call__(self, block, block_vars, gallery_conf):
        from mne_qt_browser import _pg_figure
        from sphinx_gallery.scrapers import figure_rst
        from PyQt5.QtWidgets import QApplication
        if gallery_conf['builder_name'] != 'html':
            return ''
        img_fnames = list()
        for gui in list(_pg_figure._browser_instances):
            if getattr(gui, '_scraped', False):
                return
            gui._scraped = True  # monkey-patch but it's easy enough
            img_fnames.append(next(block_vars['image_path_iterator']))
            if getattr(gui, 'load_thread', None) is not None:
                if gui.load_thread.isRunning():
                    gui.load_thread.wait(30000)
            inst = QApplication.instance()
            # processEvents to make sure our progressBar is updated
            for _ in range(1):  # iterate in case we need more at some point
                inst.processEvents()
            pixmap = gui.grab()
            pixmap.save(img_fnames[-1])
            gui.close()
        if not len(img_fnames):
            return ''
        return figure_rst(
            img_fnames, gallery_conf['src_dir'],
            f'Raw plot{_pl(len(img_fnames))}')
