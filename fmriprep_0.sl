
# module load apptainer/fMRIPrep/23.2.3

echo PMT

subjects_string=`cat /your_path/subjects.csv`

IFS=',' read -r -a array <<< "$subjects_string"

participant_id="${array[${SLURM_ARRAY_TASK_ID}]}"

echo "participant id $participant_id"

work_dir_abs_path="/your_path/fmriprep_PMT_$participant_id"

export APPTAINER_BINDPATH="/your_path/"

image_path="/your_path/fMRIPREP/images/fmriprep_24.1.1.simg"

bids_path="/your_path/"

out_path="/your_path/"

work_dir_path="/your_path/fmriprep_PMT_$participant_id"

license_path="/your_path/license.txt"

fs_sub_dir="/your_path/"

rm -rf ${work_dir_abs_path}
mkdir ${work_dir_abs_path}

trap "rm -rf $work_dir_abs_path" EXIT

apptainer run --cleanenv ${image_path} \
${bids_path}  \
${out_path} \
participant \
--participant_label ${participant_id} \
--mem_mb 50000 \
--level full \
--fs-license-file ${license_path} \
-w ${work_dir_path} \
--clean-workdir \
--output-spaces T1w MNI152NLin2009cAsym \
--cifti-output \
--write-graph \
--notrack \
--fs-subjects-dir ${fs_sub_dir}


echo "finished $participant_id , removing working directory"
rm -rf ${work_dir_abs_path}
