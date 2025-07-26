#!/usr/bin/env python
# coding: utf-8

# In[50]:


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
                 'ALLMUL_ALLDIV', 'ALLDIV_ALLADD', 'ALLDIV_ALLSUB', 'ALLDIV_ALLMUL', 'ALLADD_ALLCON', 'ALLSUB_ALLCON', 'ALLMUL_ALLCON', 
                 'ALLDIV_ALLCON', 'ALLLVL1_CON1', 'ALLLVL2_CON2', 'ALLLVL3_CON3']


# In[52]:


for contrast in contrast_list:
    FIRST_LEVEL_PATTERN = f'/path_to_folder/GLM/first_level/**/*contrast-{contrast}_stat-effect.nii.gz'
    
    GROUP_ZMAP = f'/path_to_folder/GLM/second_level_extended/group_{contrast}_zmap.nii.gz'
    
    
    import numpy as np, pandas as pd, nibabel as nib, glob, os
    from nilearn.image import load_img, smooth_img, get_data
    from nilearn.reporting import get_clusters_table
    from nilearn.maskers import NiftiSpheresMasker
    
    # ---------------------------------------------------
    # (a) grab group z-map and find peaks above |z| > 3.29
    # ---------------------------------------------------
    # 1. extract clusters / peaks
    z_img  = load_img(GROUP_ZMAP)
    
    # 1. extract all clusters/peaks above |z| > 3.29
    peaks = get_clusters_table(z_img, stat_threshold=3.29, two_sided=True)
    
    # 2. tidy column names: lower-case & strip spaces
    peaks.columns = peaks.columns.str.lower().str.replace(' ', '_')
    
    # peak-height column in your version is now 'peak_stat'
    stat_col = 'peak_stat'           # confirmed from printout
    
    # 3. keep the 200 strongest peaks
    peaks = peaks.sort_values(stat_col, ascending=False).head(200)
    
    # 4. coords list for sphere masker
    coords = list(zip(peaks['x'], peaks['y'], peaks['z']))
    # ---------------------------------------------------
    # (b) collect subject β-maps (effect-size maps!)
    # ---------------------------------------------------
    beta_paths = sorted(glob.glob(FIRST_LEVEL_PATTERN))
    Nsub       = len(beta_paths)
    
    # 4-mm radius = two voxels in 2 mm space
    masker = NiftiSpheresMasker(
        seeds=coords,
        radius=4,                 # 4-mm sphere
        allow_overlap=True,       
        detrend=False,
        standardize=False,
        mask_img=None
    )
    betas = masker.fit_transform(beta_paths)     # shape = (Nsub, Npeaks)
    
    # ---------------------------------------------------
    # (c) compute Cohen d and its SE for each peak sphere
    # ---------------------------------------------------
    mean_beta = betas.mean(axis=0)
    sd_beta   = betas.std(axis=0, ddof=1)
    d         = mean_beta / sd_beta
    se_d      = np.sqrt((1/Nsub) + (d**2)/(2*(Nsub-1)))    # Hedges & Olkin 1985
    
    df = pd.DataFrame({
            "x": [c[0] for c in coords],
            "y": [c[1] for c in coords],
            "z": [c[2] for c in coords],
            "d": d,
            "se": se_d,
            "n": Nsub      # constant column
    })
    df.to_csv(f"peaks/peak_beta_summary_{contrast}.csv", index=False)
    print("Saved peak_beta_summary.csv with", len(df), "peaks")


# In[ ]:





# In[ ]:





# In[ ]:




