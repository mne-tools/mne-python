---
title: 'Intracranial Electrode Location and Analysis in MNE-Python'
tags:
  - Python
  - BIDS
  - MNE
  - neuroimaging
  - iEEG
  - sEEG
  - ECoG
authors:
  - name: Alexander P. Rockhill
    orcid: 0000-0003-3868-7453
    affiliation: 1
  - name: Eric Larson
    orcid: 0000-0003-4782-5360
    affiliation: 2
  - name: Brittany Stedelin
    orcid: 0000-0001-5368-4299
    affiliation: 3
  - name: Alessandra Mantovani
    affiliation: 3
  - name: Ahmed M. Raslan
    orcid: 0000-0002-8421-7105
    affiliation: 3
  - name: Alexandre Gramfort
    orcid: 0000-0001-9791-4404
    affiliation: 4
  - name: Nicole C. Swann
    orcid: 0000-0003-2463-5134
    affiliation: 1

affiliations:
 - name: Department of Human Physiology, University of Oregon, Eugene OR, USA
   index: 1
 - name: Institute for Learning and Brain Sciences, University of Washington, Seattle, WA, USA
   index: 2
 - name: Department of Neurological Surgery, Oregon Health & Science University, Portland, Oregon.
   index: 3
 - name: Université Paris-Saclay, Inria, CEA, Palaiseau, France
   index: 4

date: \today
bibliography: paper.bib
---

# Summary

Intracranial electrophysiology analysis requires precise location and anatomical labeling of electrode recording contacts before a signal processing analysis of the data can be interpreted. Signal processing techniques are common to other electrophysiology modalities, such as magnetoencephalography (MEG) and electroencephalography (EEG), so ideally locating and labelling intracranial electrodes would be an integrated part of a single analysis software. We added intracranial electrode localization and intracranial-specific analyses to MNE-Python [@GramfortEtAl2013] so that its suite of signal processing and analysis tools can be used more easily by intracranial electrophysiology researchers.

For brain electrophysiology research using stereoelectroencephalography (sEEG) and electrocorticography (ECoG), the raw electrophysiology, neurosurgical planning and neuroimaging needs to be integrated before meaningful interpretation of signal processing analyses can occur. Typically, for a patient with an intracranial electrophysiology recording, two anatomical images are collected. A preoperative magnetic resonance (MR) image is collected to obtain detailed individual brain anatomy, and a postoperative computed tomography (CT) image is collected to get a high-resolution image with 3D contact locations (although this can also be provided by a postoperative MR). The surgical plans then link the names of the contacts in the recording with their positions.

To integrate these electrophysiology and imaging data, first, the CT must be aligned to the MR in order to determine the positions of the electrode contacts in the CT relative to brain structures in the MR. In MNE-Python, these are aligned using a mutual information histogram-matching approach using DIPY [@GaryfallidisEtAl2014] that works without manual pre-alignment using the function `mne.transforms.apply_volume_registration`.

![Figure 1. A CT image correctly aligned to the MR image. The darker areas in the MR are the skull whereas the lighter, outermost regions are subcutaneous fat. The CT is shown with a red scale, and red areas shown in the slice are the skull which has high intensity in the CT.](figures/Figure_1.png)

Next, electrode contact locations are selected in merged CT-MR coordinates using a graphical user interface (GUI) implemented in PyQt (`mne.gui.locate_ieeg`). In the GUI, users click or scroll through slices to find the location of contacts. When the user navigates to the location of a channel, the channel name can be selected from a menu listing the names extracted from the recording file. Then, the user can mark the channel as associated with that location. The channel is then rendered in the 3D view as well as colored on the slice plots.

![Figure 2. A screenshot of the GUI during a sEEG electrode location. For sEEG, the 3D brain view is typically similar to the surgical plans, making it easier to determine which electrode is which. With only 2D slice plots, locations and directions must be pieced together which is substantially more difficult.](figures/Figure_2.png)

![Figure 3. A screenshot of the GUI during an ECoG electrode location. For ECoG, the 3D brain view makes it easier to select each contact in a large grid. Without the 3D view, it is difficult to continue in one row without jumping to another row incorrectly when viewing in 2D. These mistakes make it much slower to identify contacts.](figures/Figure_3.png)

Located electrode contacts in merged CT-MR coordinates for an individual subject can be morphed to a template brain for group analysis using `mne.transforms.compute_volume_registration` and `mne.transforms.apply_volume_registration`. This is done using a symmetric diffeomorphic registration [SDR; @GaryfallidisEtAl2014]. This non-linear mapping tends to be more accurate than a linear transformation such as the Talairach transform [@DaleEtAl1999] as shown in Figure 4.

![Figure 4. a) Example sEEG electrodes in a 3D rendering of individual subject anatomy. b) The electrodes mapped to the Freesurfer ``fsaverage`` brain using the SDR morph. c) The electrodes mapped to the Freesurfer ``fsaverage`` brain using the Talairach transform. The linear transformation in ``c`` is less accurate than the SDR transform in ``b`` which can be seen by the electrode in the temporal pole floating outside the pial surface in ``c``.](figures/Figure_4.png)

For ECoG electrodes, "brain shift", caused by changes in pressure during the craniotomy, can also be accounted for using MNE-Python. This shift causes grid and strip electrodes to be deeper in the post-operative CT than they were in the preoperative MR. This makes it so that the electrodes are inside the pial surface in the preoperative MRI. In MNE-Python, using `mne.preprocessing.ieeg.project_sensors_onto_brain`, this is compensated for by projecting the grid to the leptomeningeal surface.

![Figure 5. a) A 3D rendering of an ECoG grid without correction for "brain shift". In this case, the electrode contacts are under the pial surface of the preoperative brain because the change in pressure shifted the brain during the operation. b) The same 3D rendering with the "brain shift" correction.](figures/Figure_5.png)

Once the electrode locations are found both in relation to individual subject anatomy and in relation to a template brain, there are several visualization functions specific to sEEG and ECoG in MNE-Python. For sEEG, plotting the anatomical labels that the electrode shaft passes through indicates which areas are being recorded from using `mne.viz.plot_channel_labels_circle` (see Figure 6a). These areas can also be rendered in different colors in 3D for precise visualization of the trajectory and location of the electrodes using `mne.viz.Brain.add_volume_labels` as shown in Figure 6b. For ECoG, viewing a time series superimposed on a view of the 3D rendering enables the data to be displayed in relation to nearby channels using `mne.viz.snapshot_brain_montage` or `mne.stc_near_sensors`, as shown in Figure 6c.

![Figure 6. a) The anatomical labels for an sEEG electrode shaft are shown as the contacts progress from deep (starting with 1) to superficial regions. b) The anatomical surfaces that sEEG electrodes pass through are rendered along with the trajectory of the electrode shaft. c) A time-frequency decomposition of the ECoG data rendered on top of a 3D image of the brain showing the location of the grid implant.](figures/Figure_6.png)

These intracranial location and analysis steps are shown in the MNE-Python tutorials "Locating intracranial electrode contacts", "Working with sEEG data" and "Working with ECoG data".

# Statement of need

Integrating intracranial electrode location and analysis into MNE-Python allows researchers to go from raw data to visualizations of results in an all-in-one package that follows modern coding best practices, including unit tests, continuous integration, and thorough documentation. This integration provides immediate access to a wide variety of algorithms for different analyses of interest.

Previous work on intracranial software has typically come in the form of standalone packages [@HamiltonEtAl2017; @GroppeEtAl2017]. However, these are difficult to maintain as the group of developers and number of users tends to be relatively smaller, thereby hampering software maintenance as the developers transition to other projects. MNE-Python is a general-purpose electrophysiology analysis package which makes it easier to retain a larger core group of developers and makes it more likely that this package and the intracranial functionality will be maintained and improved in the long-term. MNE-Python also has stability due to the funding it receives directly for development from institutions such as National Institutes of Health, the Chan-Zuckerberg open-source initiative, the European Research Council and Agence Nationale de la Recherche in France.

Other general purpose packages, written in MATLAB, provide similar functionality [@OostenveldEtAl2011; @TadelEtAl2011] but can be difficult for researchers to integrate into Python-based analyses and requires a MATLAB license. Compared to alternatives, MNE-Python has extensive unit tests and continuous integration across Windows, Mac OS and Ubuntu operating systems, which helps ensure the stability of the code base. Previous work in Python also provides similar functionality [@HamiltonEtAl2017] but was not designed for sEEG electrode location. Adding the ability to control various aspects of the visualization (e.g., electrode contact marker size, opacity, multiple views, etc.) makes this task much easier, both for sEEG and ECoG. Additionally, compared to this previous work, MNE-Python has added key features. First it utilizes an SDR morph, which is orders of magnitude faster (approximately 15 minutes compared to 15 hours) and is comparably accurate [@AvantsEtAl2008]. Second, it has a "snap to center" of the electrode contact feature to both increase accuracy and repeatability. This feature uses the center of mass of all voxels that monotonically decrease from the highest intensity voxel nearby the selected location. Next, it has an integrated 3D view which updates upon selection. Finally, it uses a predetermined list of channel names from the recording file to speed up the localization process and to eliminate matching errors. In general, the intracranial electrode location and analysis in MNE-Python is implemented using coding best practices, it is feature complete for intracranial analysis, and it has user-friendly features. Importantly it is poised to be a well-maintained tool into the future.

# Acknowledgements

We acknowledge Liberty Hamilton for her work [@HamiltonEtAl2017] which was especially helpful to use as a comparison and to build off conceptually in the development of this project.

# References