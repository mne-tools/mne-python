# %%
import os
from pathlib import Path
import hashlib
import mne.channels
from mne.channels.channels import _BUILTIN_CHANNEL_ADJACENCIES


builtin_adj_dir = Path(mne.channels.__file__).parent / 'data' / 'neighbors'
ft_neighbors_dir = Path(os.environ['FT_NEIGHBORS_DIR'])

# First, check only within MNE
for adj in _BUILTIN_CHANNEL_ADJACENCIES:
    fname = adj.fname

    if not (ft_neighbors_dir / fname).exists():
        print(f'{fname} only ships with MNE-Python, but not with FieldTrip')
        continue

    hash_mne = hashlib.sha256()
    hash_ft = hashlib.sha256()

    with open(builtin_adj_dir / fname, 'rb') as f:
        while True:
            data = f.read(65536)
            if not data:
                break
            hash_mne.update(data)

    with open(ft_neighbors_dir / fname, 'rb') as f:
        while True:
            data = f.read(65536)
            if not data:
                break
            hash_ft.update(data)

    if hash_mne.hexdigest() != hash_ft.hexdigest():
        raise ValueError(
            f'Hash mismatch between built-in and FieldTrip neighbors '
            f'for {fname}'
        )
