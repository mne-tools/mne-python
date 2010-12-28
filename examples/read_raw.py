import fiff

fname = 'sm02a5_raw.fif'

# fid, tree, directory = fiff.fiff_open(fname, verbose=True)

raw = fiff.setup_read_raw(fname)
# data, times = fiff.read_raw_segment(raw, from_=None, to=None, sel=None)

# import pylab as pl
# pl.plot(data['evoked']['times'], data['evoked']['epochs'][:306,:].T)
# pl.show()


