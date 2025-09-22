
echo mriqc_PMT  

export APPTAINER_BINDPATH="/your_path/"

image_path="/your_path/fMRIPREP/images/mriqc_24.0.2.simg"

bids_path="/your_path/"

out_path="/your_path/"

apptainer run --cleanenv ${image_path} \
${bids_path} \
${out_path} \
group \
--notrack --no-sub \
--fd_thres 0.5

echo "finished"
