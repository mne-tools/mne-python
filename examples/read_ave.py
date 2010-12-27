import fiff

fname = 'sm02a1-ave.fif'
fid, tree, directory = fiff.fiff_open(fname, verbose=True)
