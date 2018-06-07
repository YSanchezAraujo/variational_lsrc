# variational_lsrc

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
  1. `gtoi.py` 
      * a) `write_varbvs` function creates directory structure and writes a bash file that calls the `gtoi_template.r` file when inside the singularity image
      * b) `run_sing` function writes batch files for slurm that execute a singularity image with all dependencies and environmental variables to run the analysis. 

* example command to run the first part of the analysis
```
python gtoi.py -dp /storage/gablab001/data/genus/GIT/genus/bayes/data_sets/files_for_edward/brain_snp_covars_meancentered_scaled.h5  \
-sn /storage/gablab001/data/genus/GIT/genus/bayes/data_sets/files_for_edward/dev_for_container/bf_out \
-nr 170 \ 
-ip /storage/gablab001/data/genus/GIT/genus/genus_img.sqsh \
-bp /storage:/storage \
-cn I_cols
```
