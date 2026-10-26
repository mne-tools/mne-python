#!/bin/bash -e

set -o pipefail

# Works out what this build needs, before anything is restored or downloaded:
# build.txt and pattern.txt for the doc build, wanted_datasets.txt for
# circleci_download.sh, and cache_keys/<cache>.txt for the cache keys in
# .circleci/config.yml. A cache marked noop hashes to a key nothing has ever
# saved, so its restore misses and costs nothing -- which is what most PRs want,
# since they touch no example and so need no dataset at all.

DATASETS=.circleci/datasets.txt
WANTED=wanted_datasets.txt

# config.yml cannot read $DATASETS, so make sure nobody has let the two drift:
# a path only in config.yml is archived but never checked, and one only here is
# checked but never archived
if ! diff <(awk '!/^#/ && NF {print $3}' $DATASETS | sort -u) \
          <(grep -o '~/mne_data/[^ ]*' .circleci/config.yml | sed 's|~/mne_data/||' | sort -u); then
    echo "The directories in $DATASETS and the save_cache paths in config.yml disagree."
    exit 1
fi

want() {
    grep -qxF "$1" $WANTED || echo "$1" >> $WANTED
}

: > $WANTED
echo "export OPENBLAS_NUM_THREADS=4" >> $BASH_ENV
echo "export MNE_DOC_BUILD_N_JOBS=1" >> $BASH_ENV

if { [[ "$CIRCLE_BRANCH" == "main" ]] || [[ $(cat gitlog.txt) == *"[circle full]"* ]] || [[ "$CIRCLE_BRANCH" == "maint/"* ]] ; } && [[ "$CIRCLE_PROJECT_USERNAME" == "mne-tools" ]]; then
    echo "Doing a full build";
    echo html-memory > build.txt;
    echo "export OPENBLAS_NUM_THREADS=1" >> $BASH_ENV
    echo "export MNE_DOC_BUILD_N_JOBS=4" >> $BASH_ENV
    # the full build downloads more than DATASETS covers (infant_template, the
    # phantom and parcellation fetchers, ...), so it is all or nothing
    want all
else
    echo "Doing a partial build";
    FNAMES=$(git diff --name-only $(git merge-base $CIRCLE_BRANCH upstream/main) $CIRCLE_BRANCH);
    if [[ $(cat gitlog.txt) == *"[circle front]"* ]]; then
        FNAMES="tutorials/inverse/30_mne_dspm_loreta.py tutorials/machine-learning/30_strf.py tutorials/machine-learning/50_decoding.py tutorials/stats-source-space/20_cluster_1samp_spatiotemporal.py tutorials/evoked/20_visualize_evoked.py "${FNAMES};
        want testing
    fi;
    echo FNAMES="$FNAMES";
    for FNAME in $FNAMES; do
        if [[ $(echo "$FNAME" | grep -P '^(tutorials|examples)(/.*)?/((?!sgskip).)*\.py$') ]] ; then
            echo "Checking example $FNAME ...";
            PATTERN=$(basename $FNAME)"\\|"$PATTERN;
            while read -r NAME CACHE DIR MATCH; do
                if [[ $NAME != \#* ]] && grep -qx "$MATCH" $FNAME; then
                    want $NAME
                fi
            done < $DATASETS
        fi;
    done;
    echo PATTERN="$PATTERN";
    echo html-pattern-memory > build.txt;
    if [[ $PATTERN ]]; then
        PATTERN="\(${PATTERN::-2}\)";
    else
        PATTERN="run_no_examples_or_tutorials"
    fi;
fi;
echo "$PATTERN" > pattern.txt;

mkdir -p cache_keys
for CACHE in $(awk '!/^#/ && NF {print $2}' $DATASETS | sort -u); do
    echo noop > cache_keys/$CACHE.txt
done
while read -r NAME CACHE DIR MATCH; do
    if [[ $NAME != \#* ]] && grep -qxF -e "$NAME" -e all $WANTED; then
        echo real > cache_keys/$CACHE.txt
    fi
done < $DATASETS

echo "Datasets wanted: $(tr '\n' ' ' < $WANTED)"
grep -H . cache_keys/*.txt
