# variational_lsrc

### data:

Input data `brain_snp_covars_meancentered_scaled.h5` is an hdf5 file, see the file `save_h5.py` for more information. 
Cross validation splits were pre-made and saved to a csv file, see `save_cv_splits.py` for more information.

### computing

* this is intended for HCP use, if you only need to run this a single time, you can bypass all of the singularity, and nipype and just use the R files.

docker url:

https://hub.docker.com/r/ysa6/genus/

command to build using singularity: 
```
singularity build genus_img.sqsh docker://ysa6/genus:latest
```

command to run the container for an interactive session
```
singularity shell --bind /storage:/storage genus_img.sqsh
```

General logic of execution:
  1. `gtoi_cv.py` 
      * a) `write_varbvs` function creates directory structure and writes a bash file that calls the `gtoi_cv_template.r` file when inside the singularity image
      * b) `run_sing` function writes batch files for slurm that execute a singularity image with all dependencies and environmental variables to run the analysis. 

* example command to run the first part of the analysis
```
python gtoi_cv.py -dp /storage/gablab001/data/genus/GIT/genus/bayes/data_sets/files_for_edward/brain_snp_covars_meancentered_scaled.h5 \
-sn /storage/gablab001/data/genus/GIT/genus/bayes/data_sets/files_for_edward/dev_for_container/bf_out_cv \
-nr 170 \
-ip /storage/gablab001/data/genus/GIT/genus/genus_img.sqsh \
-bp /storage:/storage \
-cn I_cols \
-cv /storage/gablab001/data/genus/GIT/genus/bayes/data_sets/files_for_edward/dev_for_container/shuffle_split_cv.csv \
-gd /storage/gablab001/data/genus/GIT/genus/bayes/data_sets/files_for_edward/dev_for_container/gtoi_res \
```

that first part should have created a directory where the bayes factor scores are outputed, using the path to that directory you run the second part with this example command, using the same logic execution as above: 

2. `itod_cv.py`
```
python itod_cv.py -dp /storage/gablab001/data/genus/GIT/genus/bayes/data_sets/files_for_edward/brain_snp_covars_meancentered_scaled.h5 \
-sn /storage/gablab001/data/genus/GIT/genus/bayes/data_sets/files_for_edward/dev_for_container/fxvb_out_cv \
-ip /storage/gablab001/data/genus/GIT/genus/genus_img.sqsh \
-bp /storage:/storage \
-bf /storage/gablab001/data/genus/GIT/genus/bayes/data_sets/files_for_edward/dev_for_container/bf_out_cv \
-nc 100 \
-cp /storage/gablab001/data/genus/GIT/genus/bayes/data_sets/files_for_edward/dev_for_container/shuffle_split_cv.csv
```

* take a look at `itody_cv.py` and `gtoi_cv.py` to see what the arguments e.g. `-bf` mean. 

#### Please cite:
```
Batmanghelich N, Dalca A, Quon G, Sabuncu M, Golland P. Probabilistic Modeling of Imaging, Genetics and the Diagnosis. IEEE Transactions on Medical Imaging (TMI), pp 1765-1779, 2016.  doi: 10.1109/TMI.2016.2527784. PMID: 26886973. PMCID: PMC5364030.
```
