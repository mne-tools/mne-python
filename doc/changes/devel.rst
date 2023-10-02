`1.6.0.dev235+gc154d8aa7 <https://github.com/mne-tools/mne-python/main>`__ (2023-10-02)
---------------------------------------------------------------------------------------

**Enhancements**

-  MAINT: Warn when fitting cHPI amplitudes on Maxwell filtered data `#12038 <https://github.com/mne-tools/mne-python/pull/12038>`__ (`larsoner <https://github.com/larsoner>`__)
-  ENH: Add Forward.save and hdf5 support `#12036 <https://github.com/mne-tools/mne-python/pull/12036>`__ (`larsoner <https://github.com/larsoner>`__)
-  [ENH] update version and checksum in config.py for testing datasets `#12032 <https://github.com/mne-tools/mne-python/pull/12032>`__ (`KristijanArmeni <https://github.com/KristijanArmeni>`__)
-  Adding nan method to interpolate channels `#12027 <https://github.com/mne-tools/mne-python/pull/12027>`__ (`anaradanovic <https://github.com/anaradanovic>`__)
-  [MRG] Improve docstring format and support floats for conductivity in make_bem_model `#12020 <https://github.com/mne-tools/mne-python/pull/12020>`__ (`mscheltienne <https://github.com/mscheltienne>`__)
-  Add mne.preprocessing.unify_bad_channels `#12014 <https://github.com/mne-tools/mne-python/pull/12014>`__ (`anaradanovic <https://github.com/anaradanovic>`__)
-  ENH: Build API entry usage graphs `#12013 <https://github.com/mne-tools/mne-python/pull/12013>`__ (`larsoner <https://github.com/larsoner>`__)

**Bugfixes**

-  Do not set annotation channel when missing from input data when reading EDF `#12044 <https://github.com/mne-tools/mne-python/pull/12044>`__ (`paulroujansky <https://github.com/paulroujansky>`__)
-  Mark tests as network tests `#12041 <https://github.com/mne-tools/mne-python/pull/12041>`__ (`mbalatsko <https://github.com/mbalatsko>`__)
-  BUG: Fix bug with validation of info[“bads”] `#12039 <https://github.com/mne-tools/mne-python/pull/12039>`__ (`larsoner <https://github.com/larsoner>`__)
-  Stack vertices in plot_volume_source_estimates `#12025 <https://github.com/mne-tools/mne-python/pull/12025>`__ (`mscheltienne <https://github.com/mscheltienne>`__)
-  FIX: allow user to pick “eyegaze” or “pupil” (fix in triage_eyetrack_picks) `#12019 <https://github.com/mne-tools/mne-python/pull/12019>`__ (`scott-huberty <https://github.com/scott-huberty>`__)
-  Raise warning instead of error when loading channel-specific annotations for missing channels `#12017 <https://github.com/mne-tools/mne-python/pull/12017>`__ (`paulroujansky <https://github.com/paulroujansky>`__)
-  Correctly prune channel-specific annotations when creating Epochs `#12010 <https://github.com/mne-tools/mne-python/pull/12010>`__ (`mscheltienne <https://github.com/mscheltienne>`__)

**Documentation**

-  DOC: Remove make test `#12042 <https://github.com/mne-tools/mne-python/pull/12042>`__ (`larsoner <https://github.com/larsoner>`__)
-  MAINT: Fix example `#12035 <https://github.com/mne-tools/mne-python/pull/12035>`__ (`larsoner <https://github.com/larsoner>`__)
-  adding crossref from forward to create trans `#12033 <https://github.com/mne-tools/mne-python/pull/12033>`__ (`anaradanovic <https://github.com/anaradanovic>`__)
-  Fix minor typo found by codespell `#12029 <https://github.com/mne-tools/mne-python/pull/12029>`__ (`DimitriPapadopoulos <https://github.com/DimitriPapadopoulos>`__)
-  MAINT: Fix typos found by codespell `#12021 <https://github.com/mne-tools/mne-python/pull/12021>`__ (`DimitriPapadopoulos <https://github.com/DimitriPapadopoulos>`__)
-  MRG: Suggest to use conda-libmamba-solver instead of mamba in install docs `#11944 <https://github.com/mne-tools/mne-python/pull/11944>`__ (`hoechenberger <https://github.com/hoechenberger>`__)

**Code health**

-  MAINT: Speed up doc build `#12040 <https://github.com/mne-tools/mne-python/pull/12040>`__ (`larsoner <https://github.com/larsoner>`__)
-  MAINT: Use PyPI upload action for published releases `#12037 <https://github.com/mne-tools/mne-python/pull/12037>`__ (`larsoner <https://github.com/larsoner>`__)
-  [pre-commit.ci] pre-commit autoupdate `#12016 <https://github.com/mne-tools/mne-python/pull/12016>`__ (`pre-commit-ci[bot] <https://github.com/apps/pre-commit-ci>`__)

**Merged pull requests:**

-  Bump actions/checkout from 3 to 4 `#12046 <https://github.com/mne-tools/mne-python/pull/12046>`__ (`dependabot[bot] <https://github.com/apps/dependabot>`__)
-  MAINT, DOC: add eyetracking & Dipole convention for loc array to Info class API `#12023 <https://github.com/mne-tools/mne-python/pull/12023>`__ (`scott-huberty <https://github.com/scott-huberty>`__)
-  BUG: Fix doc building bugs `#12009 <https://github.com/mne-tools/mne-python/pull/12009>`__ (`larsoner <https://github.com/larsoner>`__)
-  MAINT: Link to sg_execution_times `#12008 <https://github.com/mne-tools/mne-python/pull/12008>`__ (`larsoner <https://github.com/larsoner>`__)
-  Add EvokedField.plotter `#12005 <https://github.com/mne-tools/mne-python/pull/12005>`__ (`wmvanvliet <https://github.com/wmvanvliet>`__)
-  Add the missing overwrite and verbose parameters to Transform.save `#12004 <https://github.com/mne-tools/mne-python/pull/12004>`__ (`wmvanvliet <https://github.com/wmvanvliet>`__)
-  [MRG] Don’t look for an offset in an eyelink message if the message contains only 2 elements `#12003 <https://github.com/mne-tools/mne-python/pull/12003>`__ (`mscheltienne <https://github.com/mscheltienne>`__)
-  [pre-commit.ci] pre-commit autoupdate `#12002 <https://github.com/mne-tools/mne-python/pull/12002>`__ (`pre-commit-ci[bot] <https://github.com/apps/pre-commit-ci>`__)
-  BUG: Fix bug with get_view `#12000 <https://github.com/mne-tools/mne-python/pull/12000>`__ (`larsoner <https://github.com/larsoner>`__)
-  BUG: Fix bug with clip box setting `#11999 <https://github.com/mne-tools/mne-python/pull/11999>`__ (`larsoner <https://github.com/larsoner>`__)
-  more dev reports infrastructure `#11997 <https://github.com/mne-tools/mne-python/pull/11997>`__ (`drammock <https://github.com/drammock>`__)
-  Fix \_check_edflib_installed `#11996 <https://github.com/mne-tools/mne-python/pull/11996>`__ (`mscheltienne <https://github.com/mscheltienne>`__)
-  MAINT: Replace in1d with isin `#11994 <https://github.com/mne-tools/mne-python/pull/11994>`__ (`larsoner <https://github.com/larsoner>`__)
-  MAINT: Switch to scheduled runs for macOS [skip azp] [skip actions] `#11993 <https://github.com/mne-tools/mne-python/pull/11993>`__ (`larsoner <https://github.com/larsoner>`__)
-  Add mscheltienne to codeowners `#11989 <https://github.com/mne-tools/mne-python/pull/11989>`__ (`mscheltienne <https://github.com/mscheltienne>`__)
-  [pre-commit.ci] pre-commit autoupdate `#11988 <https://github.com/mne-tools/mne-python/pull/11988>`__ (`pre-commit-ci[bot] <https://github.com/apps/pre-commit-ci>`__)
-  update codeowners agramfort `#11987 <https://github.com/mne-tools/mne-python/pull/11987>`__ (`agramfort <https://github.com/agramfort>`__)
-  Add ADAM2392 to codeowners `#11983 <https://github.com/mne-tools/mne-python/pull/11983>`__ (`adam2392 <https://github.com/adam2392>`__)
-  MAINT: Fix notebook `#11982 <https://github.com/mne-tools/mne-python/pull/11982>`__ (`larsoner <https://github.com/larsoner>`__)
-  Add Richard to CODEOWNERS `#11981 <https://github.com/mne-tools/mne-python/pull/11981>`__ (`hoechenberger <https://github.com/hoechenberger>`__)
-  Bump actions/checkout from 3 to 4 `#11980 <https://github.com/mne-tools/mne-python/pull/11980>`__ (`dependabot[bot] <https://github.com/apps/dependabot>`__)
-  Add cbrnr to codeowners `#11979 <https://github.com/mne-tools/mne-python/pull/11979>`__ (`cbrnr <https://github.com/cbrnr>`__)
-  Deprecate complex spectrum obj `#11978 <https://github.com/mne-tools/mne-python/pull/11978>`__ (`drammock <https://github.com/drammock>`__)
-  [MAINT] Add Marijn to codeowners `#11976 <https://github.com/mne-tools/mne-python/pull/11976>`__ (`wmvanvliet <https://github.com/wmvanvliet>`__)
-  MAINT: Add maintenance tool to check status `#11975 <https://github.com/mne-tools/mne-python/pull/11975>`__ (`larsoner <https://github.com/larsoner>`__)
-  [MAINT] Add to codeowners `#11974 <https://github.com/mne-tools/mne-python/pull/11974>`__ (`alexrockhill <https://github.com/alexrockhill>`__)
-  MAINT: Add rob-luke to CODEOWNERS `#11972 <https://github.com/mne-tools/mne-python/pull/11972>`__ (`larsoner <https://github.com/larsoner>`__)
-  mailmap updates `#11971 <https://github.com/mne-tools/mne-python/pull/11971>`__ (`drammock <https://github.com/drammock>`__)
-  MAINT: Refactor codeowners `#11970 <https://github.com/mne-tools/mne-python/pull/11970>`__ (`larsoner <https://github.com/larsoner>`__)
-  MAINT: Test notebook on Windows `#11965 <https://github.com/mne-tools/mne-python/pull/11965>`__ (`larsoner <https://github.com/larsoner>`__)
-  Make ``_read_annotations_edf()`` and ``_read_annotations_txt()`` return an ``mne.Annotations`` instance `#11964 <https://github.com/mne-tools/mne-python/pull/11964>`__ (`paulroujansky <https://github.com/paulroujansky>`__)
-  Use newer ipympl package for drawing matplotlib figures into a notebook `#11962 <https://github.com/mne-tools/mne-python/pull/11962>`__ (`wmvanvliet <https://github.com/wmvanvliet>`__)
-  Handle channel information in annotations when loading data from and exporting to EDF file `#11960 <https://github.com/mne-tools/mne-python/pull/11960>`__ (`paulroujansky <https://github.com/paulroujansky>`__)
-  Fix encoding in read_annotations_edf `#11958 <https://github.com/mne-tools/mne-python/pull/11958>`__ (`adgilbert <https://github.com/adgilbert>`__)
-  update MNE-C install docs `#11957 <https://github.com/mne-tools/mne-python/pull/11957>`__ (`drammock <https://github.com/drammock>`__)
-  MAINT: Update to use trame backend `#11956 <https://github.com/mne-tools/mne-python/pull/11956>`__ (`larsoner <https://github.com/larsoner>`__)
-  remove compat code for versions we no longer support `#11955 <https://github.com/mne-tools/mne-python/pull/11955>`__ (`drammock <https://github.com/drammock>`__)
-  [pre-commit.ci] pre-commit autoupdate `#11954 <https://github.com/mne-tools/mne-python/pull/11954>`__ (`pre-commit-ci[bot] <https://github.com/apps/pre-commit-ci>`__)
-  Refactor writing raw file `#11953 <https://github.com/mne-tools/mne-python/pull/11953>`__ (`dmalt <https://github.com/dmalt>`__)
-  Fix bug with subject_info when loading from and exporting to EDF file `#11952 <https://github.com/mne-tools/mne-python/pull/11952>`__ (`paulroujansky <https://github.com/paulroujansky>`__)
-  MAINT: Simplify logic for radius `#11951 <https://github.com/mne-tools/mne-python/pull/11951>`__ (`larsoner <https://github.com/larsoner>`__)
-  Fix realign_raw and test_realign `#11950 <https://github.com/mne-tools/mne-python/pull/11950>`__ (`qian-chu <https://github.com/qian-chu>`__)
-  MAINT: Work around seaborn<->pandas warning `#11948 <https://github.com/mne-tools/mne-python/pull/11948>`__ (`larsoner <https://github.com/larsoner>`__)
-  Fix bug with Brain.add_annotation when reading from file `#11946 <https://github.com/mne-tools/mne-python/pull/11946>`__ (`wmvanvliet <https://github.com/wmvanvliet>`__)
-  Interactive version of plot_evoked_fieldmap `#11942 <https://github.com/mne-tools/mne-python/pull/11942>`__ (`wmvanvliet <https://github.com/wmvanvliet>`__)
-  MAINT: Fix examples for newer pandas `#11940 <https://github.com/mne-tools/mne-python/pull/11940>`__ (`larsoner <https://github.com/larsoner>`__)
-  API: Deprecate mne maxfilter `#11939 <https://github.com/mne-tools/mne-python/pull/11939>`__ (`larsoner <https://github.com/larsoner>`__)
-  Fix some security issues and add Bandit to our CIs `#11937 <https://github.com/mne-tools/mne-python/pull/11937>`__ (`drammock <https://github.com/drammock>`__)
-  DOC: update name of changelog in contributing guide `#11935 <https://github.com/mne-tools/mne-python/pull/11935>`__ (`scott-huberty <https://github.com/scott-huberty>`__)
-  BUG: Fix bug with CTF data `#11934 <https://github.com/mne-tools/mne-python/pull/11934>`__ (`larsoner <https://github.com/larsoner>`__)
-  MAINT: Use automatic weakref decorator `#11932 <https://github.com/mne-tools/mne-python/pull/11932>`__ (`larsoner <https://github.com/larsoner>`__)

`v1.5.1 <https://github.com/mne-tools/mne-python/main>`__ (2023-09-06)
----------------------------------------------------------------------
