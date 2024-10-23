from dataclasses import dataclass
from mne.io import Raw, read_raw_fif


# TODO: description of what this is
ESG_CHANS = [
    "S35",
    "S24",
    "S36",
    "Iz",
    "S17",
    "S15",
    "S32",
    "S22",
    "S19",
    "S26",
    "S28", 
    "S9", 
    "S13", 
    "S11", 
    "S7", 
    "SC1", 
    "S4", 
    "S18", 
    "S8", 
    "S31", 
    "SC6", 
    "S12", 
    "S16", 
    "S5", 
    "S30", 
    "S20", 
    "S34", 
    "AC", 
    "S21", 
    "S25", 
    "L1", 
    "S29", 
    "S14", 
    "S33", 
    "S3", 
    "AL", 
    "L4", 
    "S6", 
    "S23",
]

# Set variables
fs = 1000  # sampling rate

# For heartbeat epochs
iv_baseline = [-300 / 1000, -200 / 1000]
iv_epoch = [-400 / 1000, 600 / 1000]

# Setting paths
input_path = "/data/pt_02569/tmp_data/prepared_py/sub-001/esg/prepro/"
fname = f"noStimart_sr{fs}_median_withqrs_pchip"
raw = read_raw_fif(input_path + fname + ".fif", preload=True)
