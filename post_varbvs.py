import os
import pandas as pd
import numpy as np

# this is a directory specific file (not from a pip or conda install)
bd = "/storage/gablab001/data/genus/GIT/genus/bayes/data_sets/files_for_edward/"

if os.getcwd() != bd:
    os.chdir(bd)

import utils

info_dict = dict(
    base_dir = bd,
    gtoi_res = "dev_for_container/gtoi_res",
    itod_res = "dev_for_container/fxvb_run_cv_{}",
    data_file = "brain_snp_covars_meancentered_scaled.h5",
    cv_idx = "dev_for_container/cv_splits.csv",
    bf_cv_file = "dev_for_container/bf_info_cv_{}_for_fxvb.csv",
    fxvb_cv = "dev_for_container/fxvb_run_cv_{}",
    read_imaging = True,
    read_genomic = False,
    read_depvar = True
)

# set some required variables
nsnp, nimg = int(10e4), 170
# load data
data = utils.set_env(info_dict)
# get the prediction accuarcies
classif_results, roc_auc_insample, roc_auc_outsample = utils.predict_all_folds(info_dict, data, "beta", False)
# get the pip values for plotting on the brain
df_with_itod_pips = utils.get_itod_pip(info_dict, data, nimg)
# get pip values for plotting the manhattan plots
gtoi_pip = utils.get_gtoi_pip(info_dict, 1, nsnp, nimg)
# bayes factor for each brain region after the first regression
gtoi_bf = utils.get_gtoi_bayes_factor(info_dict, data, nimg)
