
MRI Processing
==============

.. currentmodule:: mne

Step by step instructions for using :func:`gui.coregistration`:

 - `Coregistration for subjects with structural MRI
   <https://www.slideshare.net/mne-python/mnepython-coregistration>`_
 - `Scaling a template MRI for subjects for which no MRI is available
   <https://www.slideshare.net/mne-python/mnepython-scale-mri>`_

.. autosummary::
   :toctree: generated/

   coreg.get_mni_fiducials
   coreg.estimate_head_mri_t
   io.read_fiducials
   io.write_fiducials
   get_montage_volume_labels
   gui.coregistration
   gui.locate_ieeg
   create_default_subject
   head_to_mni
   head_to_mri
   read_freesurfer_lut
   read_lta
   read_talxfm
   scale_mri
   scale_bem
   scale_labels
   scale_source_space
   transforms.apply_volume_registration
   transforms.compute_volume_registration
   vertex_to_mni
   warp_montage_volume
   coreg.Coregistration
