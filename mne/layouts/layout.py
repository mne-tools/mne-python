import os.path as op

import numpy as np
import pylab as pl


class Layout(object):
    """Sensor layouts"""
    def __init__(self, kind='Vectorview-all'):
        """
        kind : 'Vectorview-all' | 'CTF-275' | 'Vectorview-grad' | 'Vectorview-mag'
        """
        lout_fname = op.join(op.dirname(__file__), kind + '.lout')

        f = open(lout_fname)
        f.readline() # skip first line

        names = []
        pos = []

        for line in f:
            _, x, y, dx, dy, chkind, nb = line.split()
            name = chkind + ' ' + nb
            pos.append(np.array([x, y, dx, dy], dtype=np.float))
            names.append(name)

        pos = np.array(pos)
        pos[:, 0] -= np.min(pos[:, 0])
        pos[:, 1] -= np.min(pos[:, 1])

        scaling = max(np.max(pos[:, 0]), np.max(pos[:, 1]))
        pos /= scaling
        pos[:,:2] += 0.03
        pos[:,:2] *= 0.97 / 1.03
        pos[:,2:] *= 0.94

        f.close()

        self.kind = kind
        self.pos = pos
        self.names = names

if __name__ == '__main__':

    layout = Layout()

    import pylab as pl
    pl.rcParams['axes.edgecolor'] = 'w'
    pl.close('all')
    pl.figure(facecolor='k', )

    for i in range(5):
    # for i in range(len(pos)):
        ax = pl.axes(layout.pos[i], axisbg='k')
        ax.plot(np.random.randn(3), 'w')
        pl.xticks([], ())
        pl.yticks([], ())
        pl.gca().grid(color='w')

    pl.show()
