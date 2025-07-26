# PMT-fMRI-validation
Analysis scripts for the paper "Parametric math task (PMT): Behavioral and fMRI cor-relates of one-, two- and three- digit addition, subtraction, multiplication and division"

The files reproduce key statistical outputs but do **not** include raw MRI data.

## Contents

* **First_level_GLM_PMT.py** – builds subject level General Linear Models, merges runs with fixed effects, and saves design matrices, effect maps, and summary reports.  
* **Second_level_GLM_PMT.py** – runs a one‑sample group model for each contrast, applies voxelwise FDR, exports thresholded maps and cluster tables.  
* **Bayesian_sample_size_step1.py** – extracts peak spheres from group z maps, computes Cohen’s *d* and its standard error, and writes a CSV per contrast.  
* **Bayesian_sample_size_step2.R** – draws posterior predictive datasets from a Cauchy prior on *d*, reruns the FDR threshold, and reports the proportion of peaks that remain significant.  
