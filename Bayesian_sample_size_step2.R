# --------------------------------------------------------------------
# 0. Load packages and set working directory
# --------------------------------------------------------------------
library(data.table)
library(stringr)      # to parse contrast names

setwd("/path_to_folder/PMT_GLM/peaks")

# --------------------------------------------------------------------
# 1. Helper: posterior draw for Cohen-d  (Normal approx to Cauchy prior)
# --------------------------------------------------------------------
rposterior_d <- function(mean_d, se_d,
                         prior_mu    = 0,
                         prior_sigma = 1,
                         n_sub, n_draws = 1L) {
  post_mean <- ((mean_d / se_d^2) + (prior_mu / prior_sigma^2)) /
    ((1 / se_d^2)      + (1 / prior_sigma^2))
  post_se   <- sqrt(1 / ((1 / se_d^2) + (1 / prior_sigma^2)))
  matrix(rnorm(length(mean_d) * n_draws, post_mean, post_se),
         nrow  = length(mean_d),
         ncol  = n_draws)
}

# --------------------------------------------------------------------
# 2. Function that returns % replicable peaks for one CSV
# --------------------------------------------------------------------
replicability_prop <- function(csv_path,
                               n_sub   = 20,
                               n_draws = 2000,
                               prior_mu = 0, prior_sigma = 1) {
  peaks <- fread(csv_path)
  n_peaks <- nrow(peaks)
  
  # Monte-Carlo over posterior predictive datasets
  hits <- replicate(n_draws, {
    # a) posterior draw of d for each peak
    d_post <- rposterior_d(peaks$d, peaks$se,
                           prior_mu, prior_sigma,
                           n_sub, n_draws = 1)
    # b) future sample
    sd_future <- sqrt((1/n_sub) + (d_post^2)/(2*(n_sub-1)))
    d_future  <- rnorm(n_peaks, mean = d_post, sd = sd_future)
    
    # c) FDR at q = .05
    z_future  <- d_future * sqrt(n_sub / 2)
    p_future  <- 2 * pnorm(-abs(z_future))
    pass_fdr  <- p_future < p.adjust(p_future, method = "BH")
    
    # d) 95 % CI excludes zero
    ci_low  <- d_future - 1.96 * sd_future
    ci_high <- d_future + 1.96 * sd_future
    pass_ci <- (ci_low > 0) | (ci_high < 0)
    
    mean(pass_fdr & pass_ci)          # fraction of peaks that replicate
  })
  
  mean(hits)                          # final proportion
}

# --------------------------------------------------------------------
# 3. Loop over every contrast file and build a summary table
# --------------------------------------------------------------------
csv_files <- list.files(pattern = "^peak_beta_summary_.*\\.csv$")

results <- rbindlist(lapply(csv_files, function(f) {
  message("Processing ", f)
  prop_rep <- replicability_prop(f)        # default n_sub = 20
  data.table(
    contrast      = str_remove_all(f, "^peak_beta_summary_|\\.csv$"),
    n_peaks       = nrow(fread(f)),
    prop_replicable = round(100 * prop_rep, 1)   # percentage
  )
}))

# --------------------------------------------------------------------
# 4. Save and print
# --------------------------------------------------------------------
fwrite(results, "replicability_summary_all_contrasts.csv")
print(results)
