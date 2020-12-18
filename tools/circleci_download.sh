#!/bin/bash -ef

if [ "$CIRCLE_BRANCH" == "master" ] || [[ $(cat gitlog.txt) == *"[circle full]"* ]]; then
    echo "Doing a full dev build";
    echo html_dev-memory > build.txt;
    python -c "import mne; mne.datasets._download_all_example_data()";
elif [ "$CIRCLE_BRANCH" == "maint/0.22" ]; then
    echo "Doing a full stable build";
    echo html_stable-memory > build.txt;
    python -c "import mne; mne.datasets._download_all_example_data()";
else
    echo "Doing a partial build";
    if ! git remote -v | grep upstream ; then git remote add upstream git://github.com/mne-tools/mne-python.git; fi
    git fetch upstream
    FNAMES=$(git diff --name-only $(git merge-base $CIRCLE_BRANCH upstream/master) $CIRCLE_BRANCH);
    if [[ $(cat gitlog.txt) == *"[circle front]"* ]]; then
        FNAMES="tutorials/source-modeling/plot_mne_dspm_source_localization.py tutorials/machine-learning/plot_receptive_field.py examples/connectivity/plot_mne_inverse_label_connectivity.py tutorials/machine-learning/plot_sensors_decoding.py tutorials/stats-source-space/plot_stats_cluster_spatio_temporal.py tutorials/evoked/plot_20_visualize_evoked.py "${FNAMES};
        python -c "import mne; print(mne.datasets.testing.data_path(update_path=True))";
    fi;
    echo FNAMES="$FNAMES";
    for FNAME in $FNAMES; do
        if [[ `expr match $FNAME "\(tutorials\|examples\)/.*plot_.*\.py"` ]] ; then
            echo "Checking example $FNAME ...";
            PATTERN=`basename $FNAME`"\\|"$PATTERN;
            if [[ $(cat $FNAME | grep -x ".*datasets.*sample.*" | wc -l) -gt 0 ]]; then
                python -c "import mne; print(mne.datasets.sample.data_path(update_path=True))";
            fi;
            if [[ $(cat $FNAME | grep -x ".*datasets.*fetch_fsaverage.*" | wc -l) -gt 0 ]]; then
                python -c "import mne; print(mne.datasets.fetch_fsaverage(verbose=True))";
            fi;
            if [[ $(cat $FNAME | grep -x ".*datasets.*spm_face.*" | wc -l) -gt 0 ]]; then
                python -c "import mne; print(mne.datasets.spm_face.data_path(update_path=True))";
            fi;
            if [[ $(cat $FNAME | grep -x ".*datasets.*somato.*" | wc -l) -gt 0 ]]; then
                python -c "import mne; print(mne.datasets.somato.data_path(update_path=True))";
            fi;
            if [[ $(cat $FNAME | grep -x ".*datasets.*eegbci.*" | wc -l) -gt 0 ]]; then
                python -c "import mne; print(mne.datasets.eegbci.load_data(1, [3, 6, 10, 14], update_path=True))";
                python -c "import mne; print(mne.datasets.eegbci.load_data(2, [3], update_path=True))";
                python -c "import mne; print(mne.datasets.eegbci.load_data(3, [3], update_path=True))";
                python -c "import mne; print(mne.datasets.eegbci.load_data(4, [3], update_path=True))";
            fi;
            if [[ $(cat $FNAME | grep -x ".*datasets.*sleep_physionet.*" | wc -l) -gt 0 ]]; then
                python -c "import mne; print(mne.datasets.sleep_physionet.age.fetch_data([0, 1], recording=[1], update_path=True))";
            fi;
            if [[ $(cat $FNAME | grep -x ".*datasets.*hf_sef.*" | wc -l) -gt 0 ]]; then
                python -c "import mne; print(mne.datasets.hf_sef.data_path(update_path=True))";
            fi;
            if [[ $(cat $FNAME | grep -x ".*brainstorm.*bst_auditory.*" | wc -l) -gt 0 ]]; then
                python -c "import mne; print(mne.datasets.brainstorm.bst_auditory.data_path(update_path=True, accept=True))";
            fi;
            if [[ $(cat $FNAME | grep -x ".*brainstorm.*bst_resting.*" | wc -l) -gt 0 ]]; then
                python -c "import mne; print(mne.datasets.brainstorm.bst_resting.data_path(update_path=True, accept=True))";
            fi;
            if [[ $(cat $FNAME | grep -x ".*brainstorm.*bst_raw.*" | wc -l) -gt 0 ]]; then
                python -c "import mne; print(mne.datasets.brainstorm.bst_raw.data_path(update_path=True, accept=True))";
            fi;
            if [[ $(cat $FNAME | grep -x ".*brainstorm.*bst_phantom_ctf.*" | wc -l) -gt 0 ]]; then
                python -c "import mne; print(mne.datasets.brainstorm.bst_phantom_ctf.data_path(update_path=True, accept=True))";
            fi;
            if [[ $(cat $FNAME | grep -x ".*brainstorm.*bst_phantom_elekta.*" | wc -l) -gt 0 ]]; then
                python -c "import mne; print(mne.datasets.brainstorm.bst_phantom_elekta.data_path(update_path=True, accept=True))";
            fi;
            if [[ $(cat $FNAME | grep -x ".*datasets.*hcp_mmp_parcellation.*" | wc -l) -gt 0 ]]; then
                python -c "import mne; print(mne.datasets.sample.data_path(update_path=True))";
                python -c "import mne; print(mne.datasets.fetch_hcp_mmp_parcellation(subjects_dir=mne.datasets.sample.data_path() + '/subjects'), accept=True)";
            fi;
            if [[ $(cat $FNAME | grep -x ".*datasets.*misc.*" | wc -l) -gt 0 ]]; then
                python -c "import mne; print(mne.datasets.misc.data_path(update_path=True))";
            fi;
            if [[ $(cat $FNAME | grep -x ".*datasets.*testing.*" | wc -l) -gt 0 ]]; then
                python -c "import mne; print(mne.datasets.testing.data_path(update_path=True))";
            fi;
            if [[ $(cat $FNAME | grep -x ".*datasets.*kiloword.*" | wc -l) -gt 0 ]]; then
                python -c "import mne; print(mne.datasets.kiloword.data_path(update_path=True))";
            fi;
            if [[ $(cat $FNAME | grep -x ".*datasets.*mtrf.*" | wc -l) -gt 0 ]]; then
                python -c "import mne; print(mne.datasets.mtrf.data_path(update_path=True))";
            fi;
            if [[ $(cat $FNAME | grep -x ".*datasets.*fieldtrip_cmc.*" | wc -l) -gt 0 ]]; then
                python -c "import mne; print(mne.datasets.fieldtrip_cmc.data_path(update_path=True))";
            fi;
            if [[ $(cat $FNAME | grep -x ".*datasets.*multimodal.*" | wc -l) -gt 0 ]]; then
                python -c "import mne; print(mne.datasets.multimodal.data_path(update_path=True))";
            fi;
            if [[ $(cat $FNAME | grep -x ".*datasets.*fnirs_motor.*" | wc -l) -gt 0 ]]; then
                python -c "import mne; print(mne.datasets.fnirs_motor.data_path(update_path=True))";
            fi;
            if [[ $(cat $FNAME | grep -x ".*datasets.*opm.*" | wc -l) -gt 0 ]]; then
                python -c "import mne; print(mne.datasets.opm.data_path(update_path=True))";
            fi;
            if [[ $(cat $FNAME | grep -x ".*datasets.*phantom_4dbti.*" | wc -l) -gt 0 ]]; then
                python -c "import mne; print(mne.datasets.phantom_4dbti.data_path(update_path=True))";
            fi;
            if [[ $(cat $FNAME | grep -x ".*datasets.*limo.*" | wc -l) -gt 0 ]]; then
                python -c "import mne; print(mne.datasets.limo.data_path(subject=1, update_path=True))";
            fi;
            if [[ $(cat $FNAME | grep -x ".*datasets.*refmeg_noise.*" | wc -l) -gt 0 ]]; then
                python -c "import mne; print(mne.datasets.refmeg_noise.data_path(update_path=True))";
            fi;
        fi;
    done;
    echo PATTERN="$PATTERN";
    if [[ $PATTERN ]]; then
        PATTERN="\(${PATTERN::-2}\)";
        echo html_dev-pattern-memory > build.txt;
    else
        echo html_dev-noplot > build.txt;
    fi;
fi;
echo "$PATTERN" > pattern.txt;