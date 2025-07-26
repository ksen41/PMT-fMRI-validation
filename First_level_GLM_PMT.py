#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import json
import gc
import numpy as np
import pandas as pd
import logging
import traceback
import numpy as np
import pandas as pd
import nibabel as nib
import matplotlib.pyplot as plt
from nilearn import plotting
from nilearn import image
from nilearn.image import clean_img
from nilearn import signal
from nilearn.glm.first_level import make_first_level_design_matrix, run_glm
from nilearn.glm.contrasts import compute_contrast
from nilearn.plotting import plot_contrast_matrix, plot_design_matrix
from nilearn.signal import clean
from nilearn.maskers import NiftiMasker
from nilearn.glm.first_level import FirstLevelModel
from nilearn.glm.contrasts import compute_fixed_effects
from nilearn.reporting import make_glm_report


# In[2]:


## create mask:
from nilearn import image
from templateflow import api as tflow
from nilearn.maskers import NiftiMasker

mni_gm = tflow.get('MNI152NLin2009cAsym', desc='brain',# label='GM', 
resolution=2, suffix='T1w', 
extension='nii.gz')
gray_matter_binary = image.math_img('i1 > 0.5', i1=image.load_img(mni_gm))
masker = NiftiMasker(smoothing_fwhm=6, standardize='zscore_sample').fit(gray_matter_binary)
masker.generate_report()


# In[3]:


def load_repetition_time(data_dir, sub_label, task_label, run_id):
    """
    Load repetition time (TR) from the fmriprep JSON file.

    Parameters
    ----------
    data_dir: str
        Directory where data is stored.
    sub_label: str
        Subject identifier 
    task_label: str
        Task name 

    Returns
    -------
    t_r : float
        Repetition time (in seconds).
    """
    json_file = os.path.join(
        data_dir,
        f"derivatives/fmriprep_24.1.1/{sub_label}/func/"
        f"{sub_label}_task-{task_label}_run-{run_id}_"
        "space-MNI152NLin2009cAsym_desc-preproc_bold.json"
    )

    with open(json_file, "r") as f:
        t_r = json.load(f).get("RepetitionTime", None)
    return t_r


# In[4]:


def manage_confounds(original_confs, dummy_num):
    """
    Extract desired confound regressors (12 motion parameters + Cosine + aCompCor)

    Parameters
    ----------
    original_confs : pd.DataFrame
        The original confounds DataFrame from fmriprep.

    Returns
    -------
    confs_final : pd.DataFrame
        Confounds after selecting target regressors.
    """
    import numpy as np

    # 12 motion regressors (6 real, 6 derivatives)
    movement_regs = [
        'rot_x', 'rot_x_derivative1', 'rot_y', 'rot_y_derivative1',
        'rot_z', 'rot_z_derivative1', 'trans_x', 'trans_x_derivative1',
        'trans_y', 'trans_y_derivative1', 'trans_z', 'trans_z_derivative1'
    ]
    # Cosine terms for high-pass filtering
    cosine_regs = original_confs.filter(regex='^cosine').columns.tolist()
    # aCompCor (top 10)
    c_comp_cor = original_confs.filter(regex='^c_comp_cor_').iloc[:, :5].columns.tolist()
    w_comp_cor = original_confs.filter(regex='^w_comp_cor_').iloc[:, :5].columns.tolist()

    desired_confounds = movement_regs + cosine_regs + c_comp_cor + w_comp_cor

    available_confounds = [col for col in desired_confounds if col in original_confs.columns]

    cleaned_confounds = original_confs[available_confounds].fillna(0)

    # Remove dummy scans
    fin_confounds = cleaned_confounds.loc[dummy_num:]
    fin_confounds = fin_confounds.reset_index(drop=True)

    # Add a linear trend
    fin_confounds['linear_trend'] = np.arange(len(fin_confounds))

    return fin_confounds


# In[5]:


def prepare_run_data(sub_label, run_id, task_label, data_dir, t_r, dummy_num):
    """
    Load and prepare data for a single run (NIfTI-based).

    Parameters
    ----------
    sub_label : str
        Subject ID
    run_id : str
        Run ID (e.g., '1').
    task_label : str
        Task label 
    data_dir : str
        Directory for data.
    t_r : float
        Repetition time (in seconds).
    dummy_num : int
        Number of dummy scans to discard.

    Returns
    -------
    events : pd.DataFrame
        The task-event information.
    confounds_final : pd.DataFrame
        Cleaned confounds to include in the design matrix.
    nifti_img : nib.Nifti1Image
        The original 4D NIfTI image
    """

    import nilearn
    from nilearn.image import clean_img

    # File paths
    fmriprep_bold = os.path.join(
        data_dir,
        f"derivatives/fmriprep_24.1.1/{sub_label}/func/"
        f"{sub_label}_task-{task_label}_run-{run_id}_"
        "space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz"
    )

    events_file = os.path.join(
        data_dir, 
        f"/your_path/{sub_label}/func/{sub_label}_task-{task_label}_run-{run_id}_events.tsv"
    )

    conf_file = os.path.join(
        data_dir,
        f"derivatives/fmriprep_24.1.1/{sub_label}/func/"
        f"{sub_label}_task-{task_label}_run-{run_id}_desc-confounds_timeseries.tsv"
    )

    # Load events and confounds
    events_raw = pd.read_csv(events_file, sep="\t")
    confounds = pd.read_csv(conf_file, sep="\t")
    confounds_final = manage_confounds(confounds, dummy_num).fillna(0)

    # Adjust events timing for dummy scans
    events = events_raw.copy()
    # 1. shift all onsets back by the dummy-scan interval
    shift = dummy_num * t_r
    events['onset'] -= shift
    # 2. compute each event’s end time after the shift
    events['end'] = events['onset'] + events['duration']
    # 3. drop any events that end on or before time 0 (fully in the dummy period)
    events = events[events['end'] > 0].copy()
    # 4. for events that now start before 0, trim them to start at 0
    mask = events['onset'] < 0
    events.loc[mask, 'duration'] = events.loc[mask, 'end']
    events.loc[mask, 'onset'] = 0
    # 5. clean up
    events.drop(columns='end', inplace=True)
    events.reset_index(drop=True, inplace=True)


    # Load the 4D NIfTI image and 3D brain mask
    nifti_img = nib.load(fmriprep_bold)
    # Drop the first 5 dummies
    nifti_img_trimmed = nib.Nifti1Image(nifti_img.get_fdata()[..., dummy_num:], affine=nifti_img.affine, header=nifti_img.header)

    return events, confounds_final, nifti_img_trimmed



# In[7]:


def prepare_contrasts(fitted_model):
    """
    Prepare a dictionary of named contrasts from a design matrix.

    Parameters
    ----------
    fitted_model: output from nilearn FirstLevelMode() function
    
    Returns
    -------
    contrasts : dict
        Mapping of contrast names to contrast vectors (np.array).
    """
    """
    Prepare a dictionary of named contrasts from a design matrix.

    Parameters
    ----------
    fitted_model: output from nilearn FirstLevelMode() function
    
    Returns
    -------
    contrasts : dict
        Mapping of contrast names to contrast vectors (np.array).
    """
    event_design_map = fitted_model.design_matrices_[0]
    # Identity matrix for each column in the design matrix
    contrast_matrix = np.eye(event_design_map.shape[1])
    contrasts = {column: contrast_matrix[i]
                 for i, column in enumerate(event_design_map.columns)}

    # Custom contrasts
    custom_contrasts = {
        # 1–15: each condition vs Fixation (cross)
        'ADD1_REST': contrasts['ADD1'] - contrasts['rest'],
        'ADD2_REST': contrasts['ADD2'] - contrasts['rest'],
        'ADD3_REST': contrasts['ADD3'] - contrasts['rest'],
        'SUB1_REST': contrasts['SUB1'] - contrasts['rest'],
        'SUB2_REST': contrasts['SUB2'] - contrasts['rest'],
        'SUB3_REST': contrasts['SUB3'] - contrasts['rest'],
        'MUL1_REST': contrasts['MUL1'] - contrasts['rest'],
        'MUL2_REST': contrasts['MUL2'] - contrasts['rest'],
        'MUL3_REST': contrasts['MUL3'] - contrasts['rest'],
        'DIV1_REST': contrasts['DIV1'] - contrasts['rest'],
        'DIV2_REST': contrasts['DIV2'] - contrasts['rest'],
        'DIV3_REST': contrasts['DIV3'] - contrasts['rest'],
        'CON1_REST': contrasts['CON1'] - contrasts['rest'],
        'CON2_REST': contrasts['CON2'] - contrasts['rest'],
        'CON3_REST': contrasts['CON3'] - contrasts['rest'],
    
        # 16–27: each condition vs its Control taskn
        'ADD1_CON1': contrasts['ADD1'] - contrasts['CON1'],
        'ADD2_CON2': contrasts['ADD2'] - contrasts['CON2'],
        'ADD3_CON3': contrasts['ADD3'] - contrasts['CON3'],
        'SUB1_CON1': contrasts['SUB1'] - contrasts['CON1'],
        'SUB2_CON2': contrasts['SUB2'] - contrasts['CON2'],
        'SUB3_CON3': contrasts['SUB3'] - contrasts['CON3'],
        'MUL1_CON1': contrasts['MUL1'] - contrasts['CON1'],
        'MUL2_CON2': contrasts['MUL2'] - contrasts['CON2'],
        'MUL3_CON3': contrasts['MUL3'] - contrasts['CON3'],
        'DIV1_CON1': contrasts['DIV1'] - contrasts['CON1'],
        'DIV2_CON2': contrasts['DIV2'] - contrasts['CON2'],
        'DIV3_CON3': contrasts['DIV3'] - contrasts['CON3'],
    
        # 28–42: within-operation, between-levels
        'ADD2_ADD1': contrasts['ADD2'] - contrasts['ADD1'],
        'ADD3_ADD1': contrasts['ADD3'] - contrasts['ADD1'],
        'ADD3_ADD2': contrasts['ADD3'] - contrasts['ADD2'],
        'SUB2_SUB1': contrasts['SUB2'] - contrasts['SUB1'],
        'SUB3_SUB1': contrasts['SUB3'] - contrasts['SUB1'],
        'SUB3_SUB2': contrasts['SUB3'] - contrasts['SUB2'],
        'MUL2_MUL1': contrasts['MUL2'] - contrasts['MUL1'],
        'MUL3_MUL1': contrasts['MUL3'] - contrasts['MUL1'],
        'MUL3_MUL2': contrasts['MUL3'] - contrasts['MUL2'],
        'DIV2_DIV1': contrasts['DIV2'] - contrasts['DIV1'],
        'DIV3_DIV1': contrasts['DIV3'] - contrasts['DIV1'],
        'DIV3_DIV2': contrasts['DIV3'] - contrasts['DIV2'],
        'CON2_CON1': contrasts['CON2'] - contrasts['CON1'],
        'CON3_CON1': contrasts['CON3'] - contrasts['CON1'],
        'CON3_CON2': contrasts['CON3'] - contrasts['CON2'],
    
        # 43–45: all-operations by level
        'ALLOPS2_ALLOPS1':
            (contrasts['ADD2']+contrasts['SUB2']+contrasts['MUL2']+contrasts['DIV2'])/4
          - (contrasts['ADD1']+contrasts['SUB1']+contrasts['MUL1']+contrasts['DIV1'])/4,
        'ALLOPS3_ALLOPS1':
            (contrasts['ADD3']+contrasts['SUB3']+contrasts['MUL3']+contrasts['DIV3'])/4
          - (contrasts['ADD1']+contrasts['SUB1']+contrasts['MUL1']+contrasts['DIV1'])/4,
        'ALLOPS3_ALLOPS2':
            (contrasts['ADD3']+contrasts['SUB3']+contrasts['MUL3']+contrasts['DIV3'])/4
          - (contrasts['ADD2']+contrasts['SUB2']+contrasts['MUL2']+contrasts['DIV2'])/4,
    
        # 46–57: cross-operation at level 1
        'ADD1_SUB1': contrasts['ADD1'] - contrasts['SUB1'],
        'ADD1_MUL1': contrasts['ADD1'] - contrasts['MUL1'],
        'ADD1_DIV1': contrasts['ADD1'] - contrasts['DIV1'],
        'SUB1_ADD1': contrasts['SUB1'] - contrasts['ADD1'],
        'SUB1_MUL1': contrasts['SUB1'] - contrasts['MUL1'],
        'SUB1_DIV1': contrasts['SUB1'] - contrasts['DIV1'],
        'MUL1_ADD1': contrasts['MUL1'] - contrasts['ADD1'],
        'MUL1_SUB1': contrasts['MUL1'] - contrasts['SUB1'],
        'MUL1_DIV1': contrasts['MUL1'] - contrasts['DIV1'],
        'DIV1_ADD1': contrasts['DIV1'] - contrasts['ADD1'],
        'DIV1_SUB1': contrasts['DIV1'] - contrasts['SUB1'],
        'DIV1_MUL1': contrasts['DIV1'] - contrasts['MUL1'],
    
        # 58–69: cross-operation at level 2
        'ADD2_SUB2': contrasts['ADD2'] - contrasts['SUB2'],
        'ADD2_MUL2': contrasts['ADD2'] - contrasts['MUL2'],
        'ADD2_DIV2': contrasts['ADD2'] - contrasts['DIV2'],
        'SUB2_ADD2': contrasts['SUB2'] - contrasts['ADD2'],
        'SUB2_MUL2': contrasts['SUB2'] - contrasts['MUL2'],
        'SUB2_DIV2': contrasts['SUB2'] - contrasts['DIV2'],
        'MUL2_ADD2': contrasts['MUL2'] - contrasts['ADD2'],
        'MUL2_SUB2': contrasts['MUL2'] - contrasts['SUB2'],
        'MUL2_DIV2': contrasts['MUL2'] - contrasts['DIV2'],
        'DIV2_ADD2': contrasts['DIV2'] - contrasts['ADD2'],
        'DIV2_SUB2': contrasts['DIV2'] - contrasts['SUB2'],
        'DIV2_MUL2': contrasts['DIV2'] - contrasts['MUL2'],
    
        # 70–81: cross-operation at level 3
        'ADD3_SUB3': contrasts['ADD3'] - contrasts['SUB3'],
        'ADD3_MUL3': contrasts['ADD3'] - contrasts['MUL3'],
        'ADD3_DIV3': contrasts['ADD3'] - contrasts['DIV3'],
        'SUB3_ADD3': contrasts['SUB3'] - contrasts['ADD3'],
        'SUB3_MUL3': contrasts['SUB3'] - contrasts['MUL3'],
        'SUB3_DIV3': contrasts['SUB3'] - contrasts['DIV3'],
        'MUL3_ADD3': contrasts['MUL3'] - contrasts['ADD3'],
        'MUL3_SUB3': contrasts['MUL3'] - contrasts['SUB3'],
        'MUL3_DIV3': contrasts['MUL3'] - contrasts['DIV3'],
        'DIV3_ADD3': contrasts['DIV3'] - contrasts['ADD3'],
        'DIV3_SUB3': contrasts['DIV3'] - contrasts['SUB3'],
        'DIV3_MUL3': contrasts['DIV3'] - contrasts['MUL3'],
    
        # 82–93: all-levels by operation
        'ALLADD_ALLSUB':
            (contrasts['ADD1']+contrasts['ADD2']+contrasts['ADD3'])/3
          - (contrasts['SUB1']+contrasts['SUB2']+contrasts['SUB3'])/3,
        'ALLADD_ALLMUL':
            (contrasts['ADD1']+contrasts['ADD2']+contrasts['ADD3'])/3
          - (contrasts['MUL1']+contrasts['MUL2']+contrasts['MUL3'])/3,
        'ALLADD_ALLDIV':
            (contrasts['ADD1']+contrasts['ADD2']+contrasts['ADD3'])/3
          - (contrasts['DIV1']+contrasts['DIV2']+contrasts['DIV3'])/3,
        'ALLSUB_ALLADD':
            (contrasts['SUB1']+contrasts['SUB2']+contrasts['SUB3'])/3
          - (contrasts['ADD1']+contrasts['ADD2']+contrasts['ADD3'])/3,
        'ALLSUB_ALLMUL':
            (contrasts['SUB1']+contrasts['SUB2']+contrasts['SUB3'])/3
          - (contrasts['MUL1']+contrasts['MUL2']+contrasts['MUL3'])/3,
        'ALLSUB_ALLDIV':
            (contrasts['SUB1']+contrasts['SUB2']+contrasts['SUB3'])/3
          - (contrasts['DIV1']+contrasts['DIV2']+contrasts['DIV3'])/3,
        'ALLMUL_ALLADD':
            (contrasts['MUL1']+contrasts['MUL2']+contrasts['MUL3'])/3
          - (contrasts['ADD1']+contrasts['ADD2']+contrasts['ADD3'])/3,
        'ALLMUL_ALLSUB':
            (contrasts['MUL1']+contrasts['MUL2']+contrasts['MUL3'])/3
          - (contrasts['SUB1']+contrasts['SUB2']+contrasts['SUB3'])/3,
        'ALLMUL_ALLDIV':
            (contrasts['MUL1']+contrasts['MUL2']+contrasts['MUL3'])/3
          - (contrasts['DIV1']+contrasts['DIV2']+contrasts['DIV3'])/3,
        'ALLDIV_ALLADD':
            (contrasts['DIV1']+contrasts['DIV2']+contrasts['DIV3'])/3
          - (contrasts['ADD1']+contrasts['ADD2']+contrasts['ADD3'])/3,
        'ALLDIV_ALLSUB':
            (contrasts['DIV1']+contrasts['DIV2']+contrasts['DIV3'])/3
          - (contrasts['SUB1']+contrasts['SUB2']+contrasts['SUB3'])/3,
        'ALLDIV_ALLMUL':
            (contrasts['DIV1']+contrasts['DIV2']+contrasts['DIV3'])/3
          - (contrasts['MUL1']+contrasts['MUL2']+contrasts['MUL3'])/3,

        # additional contrasts
        # 94-97: Operation vs Control
         'ALLADD_ALLCON':
            (contrasts['ADD1'] + contrasts['ADD2'] + contrasts['ADD3']) / 3
          - (contrasts['CON1'] + contrasts['CON2'] + contrasts['CON3']) / 3,
        'ALLSUB_ALLCON':
            (contrasts['SUB1'] + contrasts['SUB2'] + contrasts['SUB3']) / 3
          - (contrasts['CON1'] + contrasts['CON2'] + contrasts['CON3']) / 3,
        'ALLMUL_ALLCON':
            (contrasts['MUL1'] + contrasts['MUL2'] + contrasts['MUL3']) / 3
          - (contrasts['CON1'] + contrasts['CON2'] + contrasts['CON3']) / 3,
        'ALLDIV_ALLCON':
            (contrasts['DIV1'] + contrasts['DIV2'] + contrasts['DIV3']) / 3
          - (contrasts['CON1'] + contrasts['CON2'] + contrasts['CON3']) / 3,
          
        # 98-100: By level vs corresponding control
        'ALLLVL1_CON1':
            (contrasts['ADD1'] + contrasts['SUB1'] + contrasts['MUL1'] + contrasts['DIV1']) / 4
          - contrasts['CON1'],
        'ALLLVL2_CON2':
            (contrasts['ADD2'] + contrasts['SUB2'] + contrasts['MUL2'] + contrasts['DIV2']) / 4
          - contrasts['CON2'],
        'ALLLVL3_CON3':
            (contrasts['ADD3'] + contrasts['SUB3'] + contrasts['MUL3'] + contrasts['DIV3']) / 4
          - contrasts['CON3']
    }

    return custom_contrasts


# In[12]:


def combine_run_level_estimates_fixed_effects(run_outputs):
    """
    Combine run-level effect sizes, variances, and z-maps using a fixed-effects 
    approach for each contrast present across the runs.
    Weights each run’s estimate by the inverse of its variance.

    Parameters
    ----------
    run_outputs : list of dict
        A list, where each element corresponds to one run.
        Each run is a dict with keys = contrast names,
        and values = {"zmap": np.array(...), 
                      "effect_size": np.array(...), 
                      "variance": np.array(...)}.
    
    Returns
    -------
    combined_results : dict
        Dictionary with keys = contrast names,
        and values = {"zmap": combined_z (np.array),
                      "effect_size": combined_beta (np.array),
                      "variance": combined_var (np.array)}.
    """
    # 1) Collect all possible contrasts across runs for a participant
    all_contrasts = set()
    for run_dict in run_outputs:
        all_contrasts.update(run_dict.keys())

    combined_results = {}

    # 2) For each contrast, gather run-level betas and variances
    for contrast in all_contrasts:
        betas = []
        variances = []

        # Collect data from runs that have this contrast
        for run_dict in run_outputs:
            if contrast in run_dict:
                betas.append(run_dict[contrast]["effect_size"])
                variances.append(run_dict[contrast]["variance"])

        # If some runs do not have that contrast, they are skipped.

        if len(betas) < 1:
            # No runs had this contrast, skip
            continue
        if len(betas) == 1:
            # Only one run has this contrast; "combined" = that single run
            single_run_dict = next(r for r in run_outputs if contrast in r)
            combined_results[contrast] = {
                "effect_size": betas[0],
                "variance": variances[0],
                "zmap": single_run_dict[contrast]["zmap"],
            }
            continue
        # 3) Apply Nilearn function
        combined_beta, combined_var, comb_stat, combined_z = compute_fixed_effects(betas, variances, precision_weighted=True, return_z_score=True)

        # 4) Store combined results
        combined_results[contrast] = {
            "effect_size": combined_beta,
            "variance": combined_var,
            "zmap": combined_z,
        }
    return combined_results


# In[84]:


def run_subject_glm_voxel(run_list, sub_label, task_label, data_dir, output_dir, masker, dummy_num):
    """
    Main function to compute GLM contrasts for a subject (NIfTI-based 1st-level).

    Parameters
    ----------
    run_id : str
        Run ID (e.g., '1').
    sub_label : str
        Subject ID 
    task_label : str
        Task label 
    data_dir : str
        Data directory.
    output_dir : str
        Directory to store outputs
    masker: nifti object
        Mask 

    Returns
    -------
    None
    """
    try:
        # Create output directory for the subject if it doesn't exist
        subject_out_dir = os.path.join(output_dir, f'{sub_label}')
        os.makedirs(subject_out_dir, exist_ok=True)

        # Initialize storage for z-scores, effect sizes, and variances
        glm_mult_run = []
        
        for run_id in run_list:
            # Load TR
            t_r = load_repetition_time(data_dir, sub_label, task_label, run_id)
            logging.info(f"Loaded TR for {sub_label}: {t_r}")
            
            # Load & prepare data (masked)
            events, confounds_final, nifti_img = prepare_run_data(sub_label, run_id, task_label, data_dir, t_r, dummy_num)
    
            # Calculate frametimes using confounds file n_rows and T_R
            frame_times = (np.arange(confounds_final.shape[0]) * t_r) + (t_r/2)
    
            model = FirstLevelModel(t_r, hrf_model='spm', drift_model= None, high_pass=0, noise_model='ar1', mask_img=masker, signal_scaling= False)
            # fit the model 
            model.fit(nifti_img, events = events, confounds = confounds_final)
            
            # Save design matrix
            design_plot = plotting.plot_design_matrix(model.design_matrices_[0])
            design_plot.figure.savefig(f"{subject_out_dir}/design_matrix_run-{run_id}.svg")
            model.design_matrices_[0].to_csv(f"{subject_out_dir}/design_matrix_run-{run_id}.csv", index = False)
    
            plt.close('all')

            contrast_results = {}
            # Prepare contrasts
            contrasts = prepare_contrasts(model)
            
            #HTML report
            report = make_glm_report(
                model,
                contrasts=contrasts,
                height_control=None,
                cluster_threshold=0,
                title="1-level GLM Report",
            )
            report.save_as_html(os.path.join(subject_out_dir, f"{sub_label}_task-{task_label}_run-{run_id}_report.html"))
            
            # For each contrast, compute stats and save
            for contrast_id, contrast_val in contrasts.items():
                effect_size = model.compute_contrast(contrast_val, output_type="effect_size")
                zmap = model.compute_contrast(contrast_val,output_type="z_score")
                variance = model.compute_contrast(contrast_val,output_type="effect_variance")

                # Save each result in a dictionary keyed by contrast name 
                contrast_results[contrast_id] = { 
                    "zmap": zmap,
                    "effect_size": effect_size,
                    "variance": variance
                }
    
                # Plot the contrast vector
                contrast_plot = plot_contrast_matrix(
                contrast_val,
                model.design_matrices_[0],
                colorbar=True,
                )
                
                contrast_plot.set_xlabel(contrast_id)
                contrast_plot.figure.set_figheight(2)
                contrast_plot.figure.set_constrained_layout(True)
                #contrast_plot.figure.savefig(f'{subject_out_dir}/{sub_label}_task-{task_label}_run-{run_id}_contrast-{contrast_id}_matrix.svg')
                plt.close('all')

            # Save this run
            glm_mult_run.append(contrast_results)
        
        # Combine run-level effect sizes, variances, and z-maps using a fixed-effects approach
        glm_combined = combine_run_level_estimates_fixed_effects(glm_mult_run)

        # Download results
        for contrast in glm_combined:
            # Save the result images
            zmap = glm_combined[contrast]['zmap']
            effect_size = glm_combined[contrast]['effect_size']
            z_img_fname = (f"{subject_out_dir}/{sub_label}_task-{task_label}_runs-combined_"
                           f"contrast-{contrast}_stat-z.nii.gz")
            eff_img_fname = (f"{subject_out_dir}/{sub_label}_task-{task_label}_runs-combined_"
                             f"contrast-{contrast}_stat-effect.nii.gz")
            zmap.to_filename(z_img_fname)
            effect_size.to_filename(eff_img_fname)
            
            #Plot the z-map in MNI space
            display = plotting.plot_stat_map(
                zmap, title=f'{sub_label} - {contrast}',
                threshold=2.0,  # corresponds to p < 0.05 (uncorrected) for a two-tailed test
                display_mode='ortho',  # shows three orthogonal planes (axial, sagittal, and coronal) in a single plot
                cut_coords=[0, 0, 0],
                cmap='seismic'
            )
            display_fname = (f"{subject_out_dir}/{sub_label}_task-{task_label}_runs-combined_"
                             f"contrast-{contrast}_zmap_preview.png")
            display.savefig(display_fname)
            display.close()
            plt.close('all')
        logging.info(f"Finished subject {sub_label}")
    
    except Exception as e:
        logging.error(f"Error with subject {sub_label}, run {run_id}: {traceback.format_exc()}")


# In[8]:


# Upload list of participants completed the task 
subject_ID_df  = pd.read_csv('/your_path/participants.tsv')
subject_ID = list(subject_ID_df['participant_id'])


# In[9]:


#subject = 'sub-12wave1'
run_list = ['01', '02', '03']
task = 'PMT'
dummy_num = 5
data_directory = '/your_path/'
output_directory = '/your_path/'


# In[87]:


run_subject_glm_voxel(run_list, subject_ID[19], task, data_directory, output_directory, masker, dummy_num)

