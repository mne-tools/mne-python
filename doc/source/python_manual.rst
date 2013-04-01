======
Manual
======

.. _mne-coreg-info:

MRI-Head Coregistration
=======================

Subjects with MRI
-----------------

In order to perform head mri coregistration in mne-python, the fiducials 
corresponding to the mri need to be specified in a file. If no such file 
exists, it can be created using the :func:`mne.gui.fiducials` gui.

After the fiducials have been saved, the coregistration can be performed using
:func:`mne.gui.coregistration`::

    >>> mne.gui.coregistration('path/to/raw.fif', subject='sample')

In the gui:

#. The position of head and mri are initially aligned using the nasion. The 
   nasion position will be held constant during the fitting procedure. Use the
   "Digitizer Position Adjustment" fields to adjust the nasion position of the 
   digitizer head shape relative to the mri.
#. Press "Fit" to estimate rotation parameters that minimize the distance from
   the digitizer points to the mri. Press "Fit Fiducials" to estimate rotation
   parameters that minimize the distance between the LAPs and RAPs. Manually 
   adjust the movement and rotation parameters   
#. Once a satisfactory coregistration is achieved, hit "Save trans" to save
   the trans file.


Subjects without MRI
--------------------

For subjects for which no structural MRI model is available, a generic brain model 
can be substituted. 

The default is to use the *fsaverage* brain that comes with Freesurfer_. 
However, any valid MRI subject can be used.
The convenience function :func:`mne.create_default_subject` can be used to copy
the *fsaverage* brain form the freesurfer directory to the subjects directory 
and add some auxiliary files form the mne directory. 

The mne-python coregistration GUI can be used to 
scale a generic brain to better match each subject's head shape. The
GUI is launched with::

    >>> mne.gui.coregistration()

In the window that opens, look for the "Data Source" section. Make sure your 
subjects directory is properly set. Then you should be able to choose 
"fsaverage" as a subject upon which you should see the fsaverage head in the 
view on the left. 
Then load your raw file. Now you should also see your digitizer head 
shape points on the left. The coregistration proceeds as follows: 

#. The position of head and MRI are initially aligned using the nasion. The 
   nasion position will be held constant with most fitting functions. Use the
   "Translation" fields to adjust the initial nasion position of the digitizer 
   head shape relative to the MRI.
#. For the "MRI Scaling" parameter, select whether to scale the brain with one 
   parameter (same scaling factor along all axes) or with three parameters
   (separate scaling factor for each axis).
#. Now, the relevant scaling parameters can be manually specified, or they can 
   be estimated in different ways using the 3 buttons below the scaling 
   parameters (hover the mouse pointer over the buttons to get a description of
   what they do). The translation and rotation parameters can also be manually
   corrected until a satisfactory coregistration is achieved.
#. Once a satisfactory coregistration is achieved, hit "Save" to save the MRI
   as well as the trans file. In the dialog that opens, specify the name of the
   new subject (i.e. the folder that is going to be created in the subjects 
   directory), and which mne utilities to run (normally the default should be 
   ok), then hit "ok". 


.. _Freesurfer: http://surfer.nmr.mgh.harvard.edu