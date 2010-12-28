import pylab as pl
import fiff

fname = 'sm02a1-cov.fif'

fid, tree, _ = fiff.fiff_open(fname)

cov_type = 1
cov = fiff.read_cov(fid, tree, cov_type)

print "cov size: %s x %s" % cov['data'].shape

pl.matshow(cov['data'])
pl.show()
