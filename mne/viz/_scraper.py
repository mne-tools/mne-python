# Authors: Eric Larson <larson.eric.d@gmail.com>
#
# License: Simplified BSD

from ..utils import _pl


class _PyQtGraphScraper:

    def __repr__(self):
        return '<PyQtGraphScraper>'

    def __call__(self, block, block_vars, gallery_conf):
        import mne_qt_browser
        from sphinx_gallery.scrapers import figure_rst
        from PyQt5.QtWidgets import QApplication
        if gallery_conf['builder_name'] != 'html':
            return ''
        img_fnames = list()
        inst = None
        n_plot = 0
        for gui in list(mne_qt_browser._browser_instances):
            try:
                scraped = getattr(gui, '_scraped', False)
            except Exception:  # super __init__ not called, perhaps stale?
                scraped = True
            if scraped:
                continue
            gui._scraped = True  # monkey-patch but it's easy enough
            n_plot += 1
            img_fnames.append(next(block_vars['image_path_iterator']))
            if getattr(gui, 'load_thread', None) is not None:
                if gui.load_thread.isRunning():
                    gui.load_thread.wait(30000)
            if inst is None:
                inst = QApplication.instance()
            # processEvents to make sure our progressBar is updated
            for _ in range(2):
                inst.processEvents()
            pixmap = gui.grab()
            pixmap.save(img_fnames[-1])
            # child figures
            for fig in gui.mne.child_figs:
                # For now we only support Selection
                if not hasattr(fig, 'channel_fig'):
                    continue
                fig = fig.channel_fig
                img_fnames.append(next(block_vars['image_path_iterator']))
                fig.savefig(img_fnames[-1])
            gui.close()
            del gui, pixmap
        if not len(img_fnames):
            return ''
        for _ in range(2):
            inst.processEvents()
        return figure_rst(
            img_fnames, gallery_conf['src_dir'],
            f'Raw plot{_pl(n_plot)}')
