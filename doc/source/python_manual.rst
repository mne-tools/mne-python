======
Manual
======

.. _mne-coreg-info:

MRI-Head Coregistration
=======================

The coregistration GUI is invoked with::

    >>> mne.gui.coregistration()

Different procedures are outlined below for :ref:`coreg-with-mri` and 
:ref:`coreg-without-mri`.

.. Note::
    For many GUI elements like buttons, contextual help is available by 
    hovering the mouse pointer over the relevant controls.


.. _coreg-with-mri:

Subjects with MRI
-----------------

#. **Data Source** Panel:

   #. Make sure that ``subjects_dir`` is set correctly, to the directory
      containing your MRIs.
   #. Load your raw file. Now you should see your digitizer head shape points 
      on the left.
   #. You should then be able to select the MRI in the ``Subject`` drop-down 
      menu. 


.. _coreg-gui-fiducials:

#. **MRI Fiducials** Panel: 

   #. If the MRI already has a fiducials file they will be automatically 
      loaded and "Lock" will be selected. In this case you can skip this panel.
      If fiducials are not loaded, "Edit" will be selected and "Lock" will not
      be selectable until the fiducials are defined.
   #. If a fiducials file exists for the MRI, it can be loaded. Alternatively, 
      the fiducial points can be set manually by selecting the relevant point
      (nasion, LAP, RAP) and then clicking on the head model.
   #. Once all fiducials are set, the result can be saved (optionally) and then
      accepted by selecting "Lock" at the top of the panel.

#. **Coregistration** Panel: Use the lower half of the panel to fit the head 
   model to the MRI without scaling the MRI. 
   
   #. The position of head and MRI are initially aligned using the nasion. The 
      nasion position will be held constant during the fitting procedure. Use 
      the "Digitizer Position Adjustment" fields to adjust the nasion position 
      of the digitizer head shape relative to the MRI.
   #. Press one of the "Fit..." buttons to find a transformation that brings 
      the head shape as close as possible to the MRI:
      
      -  **Fit Head Shape** estimates rotation parameters that minimize the 
         distance from each digitizer point to the closest MRI point.
      -  **Fit LAP/RAP** estimates rotation parameters that minimize the 
         distance between the LAPs and RAPs
      -  **Fit Fiducials** estimates rotation and translation that minimizes
         the distance between all three fiducials.
      
   #. Once a satisfactory coregistration is achieved, hit "Save" to save
      the trans file.


.. _coreg-without-mri:

Subjects without MRI
--------------------

For subjects for which no structural MRI model is available, a generic brain 
model can be substituted. The default is to use the *fsaverage* brain that 
comes with Freesurfer_, however, any valid MRI subject can be used. 


#. **Data Source** panel:

   #. Make sure that ``subjects_dir`` is set correctly, to the directory
      which contains (or is to contain) your MRIs.
   #. Load your raw file. Now you should see your digitizer head shape points 
      on the left.
   #. You should then be able to select any existing MRI in the ``Subject`` 
      drop-down menu and "Create FsAverage" if the subjects dir does not 
      already contain an item of that name.

#. **MRI Fiducials** Panel: Since the fsaverage brain comes with preset 
   fiducials, "Lock" should be selected automatically and you will not have to
   do anything. If you are using a brain without fiducials, refer to the 
   :ref:`MRI Fiducials Panel section <coreg-gui-fiducials>` above. 

#. **Coregistration** Panel: Select whether to scale with one factor (same 
   scaling along x, y and z axes) or with 3 factors (separate scaling factor 
   for each axis). Then, use the upper half of the panel to find a proper 
   scaling and the lower half of the panel to adjust the positioning without 
   affecting the scale.
   
   #. The position of head and MRI are initially aligned using the nasion. The
      nasion position will be held constant with most fitting functions. Use 
      the "Translation" fields to adjust the initial nasion position of the 
      digitizer head shape relative to the MRI.
   #. If the initial rotation of the head shape seems very off, use the 
      "Fit LAP/RAP" button to bring the heads into approximately overlapping
      orientation.  
   #. Use the upper row of "Fit..." buttons to find a transformation that
      fits the MRI within the head shape. 
      
      -  **Fit Head Shape** estimates scaling and rotation parameters that 
         minimize the distance from each head shape point to the closest MRI 
         point.
      -  **Fit LAP/RAP** estimates scaling and rotation parameters that 
         minimize the distance between the LAPs and RAPs.
      -  **Fit Fiducials** estimates scaling, translation and rotation 
         parameters that minimize the distance between all three fiducials.
      -  All the parameters can also be manually adjusted in the text fields.

   #. Once a satisfactory coregistration is achieved, hit "Save" to save the 
      MRI as well as the trans file. In the dialog that opens, specify the name
      of the new subject (i.e. the folder that is going to be created in the
      subjects directory), and which mne utilities should be run (normally the
      default should be ok), then hit "ok". Source spaces can also be produced 
      later by running :func:`mne.setup_source_space`.
   #. While the MRI is processed in the background you can keep working with 
      the GUI, you can coregister another subject in the same Window.

If new labels are added to the fsaverage brain after scaling it, these labels 
can be transferred to scaled copies using :func:`mne.transforms.scale_labels`.  


.. _Freesurfer: http://surfer.nmr.mgh.harvard.edu