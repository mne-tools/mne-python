# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

import numpy as np
from mne.source_estimate import SourceEstimate

# Create dummy data for the SourceEstimate
n_vertices_lh = 10  # Number of vertices in the left hemisphere
n_vertices_rh = 12  # Number of vertices in the right hemisphere
n_times = 5         # Number of time points

# Random data for the left and right hemispheres
data = np.random.rand(n_vertices_lh + n_vertices_rh, n_times)

# Vertices for the left and right hemispheres
vertices = [np.arange(n_vertices_lh), np.arange(n_vertices_rh)]

# Time parameters
tmin = 0.0  # Start time in seconds
tstep = 0.1  # Time step in seconds

# Subject name
subject = "sample_subject"

# Create a SourceEstimate object
stc = SourceEstimate(data, vertices, tmin, tstep, subject=subject)

# Save the SourceEstimate in different formats
output_dir = "./output_files"  # Directory to save the files
import os
os.makedirs(output_dir, exist_ok=True)

stc.save("test.h5",overwrite=True)
stc.save("test.stc",overwrite=True)
# # Save as .stc file
# stc.save(f"{output_dir}/dummy", ftype="stc", overwrite=True)

# # Save as .h5 file
# stc.save(f"{output_dir}/dummy.h5", ftype="h5", overwrite=True)

# print(f"Dummy files saved in {output_dir}")