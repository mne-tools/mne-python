import numpy as np
import mne
# 1. Load sample Raw data (ensures other channels have positions for interpolation)
# We crop the data to 1 second to make it fast.
print("Step 1: Loading sample MEG data...")
data_path = mne.datasets.sample.data_path()
raw_fname = data_path / 'MEG' / 'sample' / 'sample_audvis_raw.fif'
raw = mne.io.read_raw_fif(raw_fname, preload=True).pick_types(meg=True).crop(0, 1)

sfreq = raw.info['sfreq']
n_samples = raw.n_times

# 2. Create a new "empty" channel that *lacks sensor positions*
new_ch_name = 'MEG_BAD_NO_POS'
empty_ch_data = np.zeros((1, n_samples))

# The key is here: The created info object for a new channel does NOT have sensor positions (loc).
info_empty = mne.create_info(ch_names=[new_ch_name], sfreq=sfreq, ch_types=['mag'])
raw_ch = mne.io.RawArray(data=empty_ch_data, info=info_empty)

# 3. Add the new channel to the main Raw object
raw.add_channels([raw_ch], force_update_info=True)

# 4. Mark the channel as bad
raw.info['bads'] = [new_ch_name]
idx = raw.ch_names.index(new_ch_name)
# 5. Interpolate bad channels (This runs *silently* without error/warning)
print("\nStep 5: Running raw.interpolate_bads()...")
raw_interp = raw.copy().interpolate_bads()

# 6. Check the interpolated channel data (Replication of the error)
interp_data = raw_interp.get_data(picks=new_ch_name)[0]
is_nan = np.isnan(interp_data).all()
print(interp_data)
print("\n--- Replication Result ---")
print(f"Is the interpolated channel '{new_ch_name}' entirely NaN? {is_nan}")
if is_nan:
    print("✅ The bug is replicated: Interpolation failed silently, resulting in NaN data.")
print(raw.info['chs'][idx]['loc'])