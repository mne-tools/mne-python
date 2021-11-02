# Authors: Eric Larson <larson.eric.d@gmail.com>
#
# License: Simplified BSD

import time


class _PyQtGraphScraper:

    def __repr__(self):
        return '<PyQtGraphScraper>'

    def __call__(self, block, block_vars, gallery_conf):
        from mne_qt_browser._pg_figure import PyQtGraphBrowser
        from sphinx_gallery.scrapers import figure_rst
        from PyQt5.QtWidgets import QApplication
        for gui in block_vars['example_globals'].values():
            if (isinstance(gui, PyQtGraphBrowser) and
                    not getattr(gui, '_scraped', False) and
                    gallery_conf['builder_name'] == 'html'):
                gui._scraped = True  # monkey-patch but it's easy enough
                img_fname = next(block_vars['image_path_iterator'])
                inst = QApplication.instance()
                assert inst is not None
                for _ in range(30):  # max 30 sec
                    inst.processEvents()
                    inst.processEvents()
                    if gui.mne.data_precomputed:
                        inst.processEvents()
                        break
                    time.sleep(1.)
                pixmap = gui.grab()
                pixmap.save(img_fname)
                gui.close()
                inst.processEvents()
                inst.processEvents()
                return figure_rst(
                    [img_fname], gallery_conf['src_dir'], 'iEEG GUI')
        return ''
