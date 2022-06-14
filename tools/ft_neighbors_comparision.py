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
    hash = adj.hash
    hash_type = adj.hash_type
    fname = adj.fname

    if hash_type == 'sha256':
        actual_hash = hashlib.sha256()
    else:
        raise ValueError(f'Unknown hash type {hash_type}')

    with open(builtin_adj_dir / fname, 'rb') as f:
        while True:
            data = f.read(65536)
            if not data:
                break
            actual_hash.update(data)

    if actual_hash.hexdigest() != hash:
        raise ValueError(
            f'Hash mismatch between built-in definition and built-in file for '
            f'{fname}'
        )


# Now, compare between MNE and FT
for adj in _BUILTIN_CHANNEL_ADJACENCIES:
    hash = adj.hash
    hash_type = adj.hash_type
    fname = adj.fname

    if hash_type == 'sha256':
        ft_hash = hashlib.sha256()
    else:
        raise ValueError(f'Unknown hash type {hash_type}')

    if not (ft_neighbors_dir / fname).exists():
        print(f'{fname} only ships with MNE-Python, but not with FieldTrip')
        continue

    with open(ft_neighbors_dir / fname, 'rb') as f:
        while True:
            data = f.read(65536)
            if not data:
                break
            ft_hash.update(data)

    if ft_hash.hexdigest() != hash:
        raise ValueError(
            f'Hash mismatch between built-in hash and FieldTrip file for '
            f'{fname}; probably FT files have been updated'
        )
