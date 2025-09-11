from mne.datasets import testing

# testing.data_path(force_update=True, download=True)
testing.data_path(force_update=False)

path = testing.data_path()
print(path)
