from mne import ndvar
reload(mne.dimensions)
reload(ndvar)

fname = '/Users/christian/Documents/Eclipse/projects/mne-python/examples/MNE-sample-data/MEG/sample/sample_audvis-meg'
stc = mne.read_source_estimate(fname)

stcs = ndvar.from_stc([stc for _ in xrange(10)])
stc = ndvar.from_stc(stc)

stc.subdata(time=(0.1, 0.2))
