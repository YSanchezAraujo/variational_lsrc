# variational_lsrc

General logic of execution:
  1. `gtoi.py` 
      * a) `write_varbvs` function creates directory structure and writes a bash file that calls the `gtoi_template.r` file when inside the singularity image
      * b) `run_sing` function writes batch files for slurm that execute a singularity image with all dependencies and environmental variables to run the analysis. 

