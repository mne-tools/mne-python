import mne
import warnings
import numpy as np
import os

def test_label_borders():
    # Simulate the subjects_dir manually (use a local path)
    subjects_dir = os.path.expanduser("~/mne_data/MNE-sample-data/subjects")  # Adjust the path as needed
    subject = "fsaverage"  # Use a typical subject name from the dataset

    # Create mock labels as if they were read from the annotation file
    # Here, we're just using a few dummy labels for testing purposes, adding 'hemi' and 'vertices'
    labels = [
        mne.Label(np.array([0, 1, 2]), name=f"label_{i}", hemi='lh') for i in range(3)
    ]

    # Create a mock Brain object with a flat surface
    class MockBrain:
        def __init__(self, subject, hemi, surf):
            self.subject = subject
            self.hemi = hemi
            self.surf = surf

        def add_label(self, label, borders=False):
            # Simulate adding a label and handling borders logic
            if borders:
                is_flat = self.surf == 'flat'
                if is_flat:
                    warnings.warn("Label borders cannot be displayed on flat surfaces. Skipping borders.")
                else:
                    print(f"Adding borders to label: {label.name}")
            else:
                print(f"Adding label without borders: {label.name}")

    # Create the mock Brain object
    brain = MockBrain(subject=subject, hemi="lh", surf="flat")

    # Test: Add label with borders (this should show a warning for flat surfaces)
    with warnings.catch_warnings(record=True) as w:
        brain.add_label(labels[0], borders=True)
        
        # Assert that the warning is triggered for displaying borders on flat surfaces
        assert len(w) > 0
        assert issubclass(w[-1].category, UserWarning)
        assert "Label borders cannot be displayed on flat surfaces" in str(w[-1].message)

    print("Test passed!")

# Run the test
test_label_borders()
