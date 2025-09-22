
echo mriqc_PMT  

subjects_string=`cat /your_path/subjects.csv`

IFS=',' read -r -a array <<< "$subjects_string"

participant_id="${array[${SLURM_ARRAY_TASK_ID}]}"

echo "participant id $participant_id"

export APPTAINER_BINDPATH="/your_path/"

image_path="/your_path/fMRIPREP/images/mriqc_24.0.2.simg"

bids_path="/your_path/"

out_path="/your_path/"

apptainer run --cleanenv ${image_path} \
${bids_path} \
${out_path} \
participant \
--notrack --no-sub \
--participant_label ${participant_id} \
--write-graph \
--fd_thres 0.5

echo "finished $participant_id"
