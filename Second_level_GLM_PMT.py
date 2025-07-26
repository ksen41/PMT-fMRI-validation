#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import glob
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

import nibabel as nb
import nilearn as nl
from nilearn.image import math_img
from nilearn.glm.second_level import SecondLevelModel
from nilearn.plotting import plot_stat_map, view_img
from nilearn.reporting import make_glm_report

from nilearn.glm import threshold_stats_img
from nilearn.reporting import get_clusters_table

from atlasreader import create_output


# In[2]:


contrast_list = ['ADD1_REST', 'ADD2_REST', 'ADD3_REST', 'SUB1_REST', 'SUB2_REST', 'SUB3_REST', 'MUL1_REST', 'MUL2_REST', 'MUL3_REST', 
                 'DIV1_REST', 'DIV2_REST', 'DIV3_REST', 'CON1_REST', 'CON2_REST', 'CON3_REST', 'ADD1_CON1', 'ADD2_CON2', 'ADD3_CON3', 
                 'SUB1_CON1', 'SUB2_CON2', 'SUB3_CON3', 'MUL1_CON1', 'MUL2_CON2', 'MUL3_CON3', 'DIV1_CON1', 'DIV2_CON2', 'DIV3_CON3', 
                 'ADD2_ADD1', 'ADD3_ADD1', 'ADD3_ADD2', 'SUB2_SUB1', 'SUB3_SUB1', 'SUB3_SUB2', 'MUL2_MUL1', 'MUL3_MUL1', 'MUL3_MUL2', 
                 'DIV2_DIV1', 'DIV3_DIV1', 'DIV3_DIV2', 'CON2_CON1', 'CON3_CON1', 'CON3_CON2', 'ALLOPS2_ALLOPS1', 'ALLOPS3_ALLOPS1', 
                 'ALLOPS3_ALLOPS2', 'ADD1_SUB1', 'ADD1_MUL1', 'ADD1_DIV1', 'SUB1_ADD1', 'SUB1_MUL1', 'SUB1_DIV1', 'MUL1_ADD1', 'MUL1_SUB1', 
                 'MUL1_DIV1', 'DIV1_ADD1', 'DIV1_SUB1', 'DIV1_MUL1', 'ADD2_SUB2', 'ADD2_MUL2', 'ADD2_DIV2', 'SUB2_ADD2', 'SUB2_MUL2', 'SUB2_DIV2', 
                 'MUL2_ADD2', 'MUL2_SUB2', 'MUL2_DIV2', 'DIV2_ADD2', 'DIV2_SUB2', 'DIV2_MUL2', 'ADD3_SUB3', 'ADD3_MUL3', 'ADD3_DIV3', 'SUB3_ADD3', 
                 'SUB3_MUL3', 'SUB3_DIV3', 'MUL3_ADD3', 'MUL3_SUB3', 'MUL3_DIV3', 'DIV3_ADD3', 'DIV3_SUB3', 'DIV3_MUL3', 'ALLADD_ALLSUB', 
                 'ALLADD_ALLMUL', 'ALLADD_ALLDIV', 'ALLSUB_ALLADD', 'ALLSUB_ALLMUL', 'ALLSUB_ALLDIV', 'ALLMUL_ALLADD', 'ALLMUL_ALLSUB', 
                 'ALLMUL_ALLDIV', 'ALLDIV_ALLADD', 'ALLDIV_ALLSUB', 'ALLDIV_ALLMUL']




# In[4]:


def gather_contrast_files(base_path, contrast_name):
    """
    Gathers files matching the specified pattern within subdirectories and returns
    a dictionary, with subject IDs as keys and lists of file paths as values.

    Parameters:
    - base_path (str): The root directory to search within.
    - contrast_name (str): The name of the contrast to define the file pattern to match.

    Returns:
    - subject_dict (dict): A dictionary where each key is a subject ID (sub-*wave*),
      and each value is a list of matching contrast image paths for that subject.
    """
    pattern = os.path.join(
        base_path,
        "sub-*",
        f"*contrast-{contrast_name}_stat-effect.nii*"
    )
    
    contrast_imgs = sorted(glob.glob(pattern))

    # Build a dictionary: keys = subject folder, values = list of paths
    subject_dict = {}
    for img in contrast_imgs:
        # Extract folder name as subject ID
        subject_id = os.path.basename(os.path.dirname(img))
        # Store the path 
        subject_dict[subject_id] = img

    return subject_dict



# In[6]:


def run_group_analysis(base_path, output_dir, contrast_name):
    """
    Perform a group-level one-sample t-test (or multiple regression) on the 
    provided first-level contrast maps using nilearn's SecondLevelModel.

    Parameters
    ----------
    base_path : (str): 
        The root directory to contrast files (1-st level outputs).
    output_dir : str
        Directory to save second-level outputs.
    contrast_name : str
        Name of the contrast (used in output filenames).
    """
    os.makedirs(output_dir, exist_ok=True)

    subject_dict = gather_contrast_files(base_path, contrast_name)
    contrast_imgs = list(subject_dict.values())

    # Default intercept-only model
    design_matrix = pd.DataFrame({'intercept': np.ones(len(contrast_imgs))})

    # Initialize SecondLevelModel
    second_level_model = SecondLevelModel(
        smoothing_fwhm=None,   
        mask_img=None,        
    )

    # Fit the second-level model
    second_level_model = second_level_model.fit(
        contrast_imgs,
        design_matrix=design_matrix
    )
    
    # Compute the group-level contrast
    z_map = second_level_model.compute_contrast("intercept", output_type='z_score')
    eff_map = second_level_model.compute_contrast("intercept", output_type='effect_size')
    #var_map = second_level_model.compute_contrast("intercept", output_type='effect_variance')
    
    # Save the result images
    z_map_fname = os.path.join(output_dir, f"group_{contrast_name}_zmap.nii.gz")
    z_map.to_filename(z_map_fname)
    print(f"Z-map saved to: {z_map_fname}")

    eff_map_fname = os.path.join(output_dir, f"group_{contrast_name}_effect_size.nii.gz")
    eff_map.to_filename(eff_map_fname)
    
    #var_map_fname = os.path.join(output_dir, f"group_{contrast_name}_variance.nii.gz")
    #var_map.to_filename(var_map_fname)

    
   # Plot and save the z-score map
    z_display = plot_stat_map(
        z_map,
        title=f"Group {contrast_name} (z-map)",
        threshold=2.0,  # for an uncorrected p<0.05 approx
        display_mode='ortho',
        cmap = 'Spectral_r',
        cut_coords=[0, 0, 0]  
    )
    z_plot_file = os.path.join(output_dir, f"group_{contrast_name}_zmap_raw_preview.png")
    z_display.savefig(z_plot_file)
    z_display.close()
    print(f"Z-map preview saved to: {z_plot_file}")

    #HTML report
    report1 = make_glm_report(
        second_level_model,
        contrasts='intercept',
        height_control='fdr',
        alpha=0.01,
        cluster_threshold=10,
        title="Report fdr correction 0.01",
    )
    report1.save_as_html(os.path.join(output_dir, f"task-{contrast_name}_report.html"))

    # get the clusters table
    # 1. get the FDR threshold 
    thresholded_map, fdr_thr = threshold_stats_img(
        z_map,
        alpha=0.01,
        height_control='fdr',
        cluster_threshold=10,
        two_sided=True
    )

    # 2. plot the thresholded map
    thr_display = plot_stat_map(
        thresholded_map,
        title=f"Group {contrast_name} (FDR < .01, clusters ≥10 voxels)",
        threshold=fdr_thr,
        display_mode='ortho',
        cmap = 'Spectral_r',
        cut_coords=[0, 0, 0],   
    )
    thr_plot_file = os.path.join(
        output_dir,
        f"group_{contrast_name}_zmap_fdr_thresh_preview.png"
    )
    thr_display.savefig(thr_plot_file)
    thr_display.close()
    
    # 3. extract a cluster table at that height and size
    clusters = get_clusters_table(
        stat_img=z_map,
        stat_threshold=fdr_thr,   # use the FDR height
        cluster_threshold=10,      # same cluster size filter
        two_sided=True
    )
    
    # 4. write out the CSV
    clusters.to_csv(
        os.path.join(output_dir, f"group_{contrast_name}_clusters_fdr.csv"),
        index=False
    )

    create_output(
        filename=thresholded_map,          
        cluster_extent=10,       # min voxels per cluster
        outdir = os.path.join(output_dir, f"{contrast_name}")
    )
    
    # Interactive viewer in a notebook or an HTML file:
    view = view_img(thresholded_map, threshold=0, title=f"{contrast_name} FDR")
    view.save_as_html(os.path.join(output_dir, f"group_{contrast_name}_interactive_fdr.html"))

    plt.close('all')
    return thresholded_map
    #return


# In[8]:


# Base directory
base_path = '/your_path/'
output_path = '/your_path/'




for i in range(len(contrast_list)):
    run_group_analysis(base_path, output_path, contrast_list[i])


