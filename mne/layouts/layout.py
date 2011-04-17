import os.path as op

import numpy as np


class Layout(object):
    """Sensor layouts"""

    def __init__(self, kind='Vectorview-all', path=None):
        """
        Parameters
        ----------
        kind : 'Vectorview-all' | 'CTF-275' | 'Vectorview-grad' | 'Vectorview-mag'
            Type of layout (can also be custom for EEG)
        path : string
            Path to folder where to find the layout file.
        """
        if path is None:
            path = op.dirname(__file__)
        lout_fname = op.join(path, kind + '.lout')

        f = open(lout_fname)
        f.readline()  # skip first line

        names = []
        pos = []

        for line in f:
            splits = line.split()
            if len(splits) == 7:
                _, x, y, dx, dy, chkind, nb = splits
                name = chkind + ' ' + nb
            else:
                _, x, y, dx, dy, name = splits
            pos.append(np.array([x, y, dx, dy], dtype=np.float))
            names.append(name)

        pos = np.array(pos)
        pos[:, 0] -= np.min(pos[:, 0])
        pos[:, 1] -= np.min(pos[:, 1])

        scaling = max(np.max(pos[:, 0]), np.max(pos[:, 1])) + pos[0, 2]
        pos /= scaling
        pos[:, :2] += 0.03
        pos[:, :2] *= 0.97 / 1.03
        pos[:, 2:] *= 0.94

        f.close()

        self.kind = kind
        self.pos = pos
        self.names = names

# if __name__ == '__main__':
#
#     layout = Layout()
#
#     import pylab as pl
#     pl.rcParams['axes.edgecolor'] = 'w'
#     pl.close('all')
#     pl.figure(facecolor='k', )
#
#     for i in range(5):
#     # for i in range(len(pos)):
#         ax = pl.axes(layout.pos[i], axisbg='k')
#         ax.plot(np.random.randn(3), 'w')
#         pl.xticks([], ())
#         pl.yticks([], ())
#         pl.gca().grid(color='w')
#
#     pl.show()
