import os
import numpy as np
import scipy.io as sio
import h5py

""" 
assuming all datasets (I, G, Z) are already sorted by the same ID's: 

I = imaging data: Matrix (N, P)
G = SNP (genomic) data: Matrix (N, H)
Z = covariates (age, sex, ...): Matrix (N, Q)
y = dependent variable e.g. diagnosis, values in {1, 0}, eg. y = [1, 0, 1, 0, 0, 1, ....] : Array (N,)
"""


data = sio.loadmat("brain_snp_covars_meancentered_scaled.mat")
Z_cols = data["Z_cols"].astype("S")
I_cols = np.array([str(i[0]) for i in data["I_cols"][0]]).astype("S")
G_cols = np.array([str(i[0]) for i in data["G_cols"][0]]).astype("S")
I, Z, G = data["I"], data["Z"], data["G"]
y = data["y"].ravel()

hf = h5py.File("brain_snp_covars_meancentered_scaled.h5", 'w')
hf.create_dataset("I", data=I)
hf.create_dataset("Z", data=Z)
hf.create_dataset("G", data=G)
dt = h5py.special_dtype(vlen=bytes)
hf.create_dataset("I_cols", data=I_cols, dtype=dt)
hf.create_dataset("Z_cols", data=Z_cols, dtype=dt)
hf.create_dataset("G_cols", data=G_cols, dtype=dt)
hf.create_dataset("y", data=y)
hf.close()
