import os
import numpy as np
import pandas as pd
import h5py
import scipy.stats
from sklearn.model_selection import ShuffleSplit
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns

def pad(data, l, pad_val=123456789):
    padded_data = np.repeat(pad_val, l)
    padded_data[:len(data)] = data
    return padded_data

def make_cv_splits(nsamples, nfolds, test_size, pad_val=123456789):
    N_test = round(test_size*nsamples)
    N_train = int(nsamples - N_test)
    cols = [("train{}".format(k), "test{}".format(k)) for k in range(nfolds)]
    cvdf = pd.DataFrame(data=np.zeros((N_train, int(nfolds*2))), columns=[j for k in cols for j in k])
    X, y = np.random.random((nsamples, 1)), np.random.random(nsamples)
    cv = ShuffleSplit(n_splits=nfolds, test_size=test_size)

    for idx, (train, test) in enumerate(cv.split(X, y)):
        cvdf["train{}".format(idx)] = train+1
        cvdf["test{}".format(idx)] = pad(test+1, N_train, pad_val=pad_val)

    return cvdf

def save_h5(path, data_obj, dts):
    """ path: path to save the data
    data_obj: python object that the data live in
    dts: data type of data_obj
    returns: saved path and keys of data_obj in the h5 file
    """
    if len(data_obj) != len(dts):
        raise ValueError("len of data_obj and dts must be the same")

    with h5py.File(path, "w") as data_store:
        for idx, (key, val) in enumerate(data_obj.items()):
            data_set = data_store.create_dataset(
                    key, val.shape, dtype=dts[idx]
                    )
            data_set[...] = val

    return path, data_obj.keys()

def read_h5(path, key):
    """path: path to the saved h5 file
    key: key in the h5 file that you want to load
    returns: data object (numpy array or pd.DataFrame, or list)
    """
    with h5py.File(path, "r") as file_store:
        data = file_store[key][...]

    return data

def set_env(info_dict):
    """dictionary object with paths to required files, and
    location of data. Also contains boolean values that used
    to load data: e.g. imaging, genomic
    returns: dictionary object of data
    """
    bd = info_dict["base_dir"]
    bf_step_output = os.path.join(bd, info_dict["gtoi_res"])
    data_path = os.path.join(bd, info_dict["data_file"])

    if os.getcwd() != bd:
        os.chdir(bd)

    return_obj = {}
    rel_vars = dict(read_imaging="I", read_genomic="G", read_depvar="y")

    for key in info_dict.keys():
        if "read" in key:
            if info_dict[key]:
                return_obj[rel_vars[key]] = read_h5(data_path, rel_vars[key])

    # cv file was written in R and we are loading in Python, indicies need to be shifted
    return_obj["cv"] = pd.read_csv(os.path.join(bd, info_dict["cv_idx"]), index_col=0) - 1

    return return_obj

def get_gtoi_pip(info_dict, cv_idx, nsnp, nimg):
    """info_dict[gtoi_res] - path to the directory where the first step of the
    analysis output was saved
    cv_idx - required for loading the appropriate indexed results
    nsnp, nimg - number of SNP and imaging features respectively
    """
    pip_data = np.zeros((nsnp, nimg))
    cv = "_cv_{}".format(cv_idx)
    bwd, gtoi_res = info_dict["base_dir"], info_dict["gtoi_res"]

    for dir_name in os.listdir(os.path.join(info_dict["base_dir"], gtoi_res)):
        if cv in dir_name:
            wd = os.path.join(bwd, gtoi_res, dir_name)
            col_idx = int([q for q in wd.split("_") if "idx" in q][0].split("idx")[-1]) - 1
            data = pd.read_csv(os.path.join(wd, "pip_beta.csv"), index_col=None)["pip"]
            pip_data[:, col_idx] = data.values

    return pip_data

def predict_with_param(I, y, param, train, test, use_prob=False):
    """this function is not mean to be used outright, only meant to be called
    by the function predict_all_folds
    """
    f_prob = lambda x: scipy.stats.bernoulli.rvs(x)
    f = lambda x: 1 if x >= .5 else 0
    a, b = np.exp(I[train].dot(param)), np.exp(I[test].dot(param))
    probs_train, probs_test = a/(1 + a), b/(1 + b)

    if use_prob:
        yhat_train, yhat_test = (
            [f_prob(pr) for pr in probs_train], [f_prob(pr) for pr in probs_test]
        )
    else:
        yhat_train, yhat_test = (
            [f(pr) for pr in probs_train], [f(pr) for pr in probs_test]
        )

    return dict(
        train=(probs_train, yhat_train, roc_auc_score(y[train], yhat_train)),
        test=(probs_test, yhat_test, roc_auc_score(y[test], yhat_test))
    )

def predict_all_folds(info_dict, data, use_param, use_prob=False):
    """info_dict - dictionary object with paths to required files, and
    location of data. Also contains boolean values that used
    to load data: e.g. imaging, genomic
    data - dictionary object with data sets, and cv fold indices (response, imaging, ect.)
    """
    I, y = data["I"], data["y"]
    ncv = int(data["cv"].shape[1]/2)
    results = {}
    bd, ir = info_dict["base_dir"], info_dict["itod_res"]

    for cv_idx in range(ncv):
        cv = cv_idx + 1 # to account for python to r differences
        train, test = data["cv"]["train{}".format(cv)], data["cv"]["test{}".format(cv)]

        if use_param == "mu":
            path = os.path.join(bd, ir.format(cv), "fxvb_out_cv_{}".format(cv), "mu.csv")
            param = pd.read_csv(path, index_col=None).values
            results["cv{}".format(cv)] = predict_with_param(I, y, param, train, test, use_prob)
        elif use_param == "beta":
            path = os.path.join(
                bd, ir.format(cv), "fxvb_out_cv_{}".format(cv), "logw_w_sa_logodds_pip_beta.csv"
            )
            param = pd.read_csv(path, index_col=None)["beta"].values
            results["cv{}".format(cv)] = predict_with_param(I, y, param, train, test, use_prob)
        elif use_param == "beta_pip":
            path = os.path.join(
                bd, ir.format(cv), "fxvb_out_cv_{}".format(cv), "logw_w_sa_logodds_pip_beta.csv"
            )
            df_params = pd.read_csv(path, index_col=None)
            results["cv{}".format(cv)] = predict_with_param(
                I*df_params["pip"].values, y, df_params["beta"].values, train, test
            )

    train_mean_rocauc = np.mean([results["cv{}".format(cv+1)]["train"][-1] for cv in range(ncv)])
    test_mean_rocauc = np.mean([results["cv{}".format(cv+1)]["test"][-1] for cv in range(ncv)])

    return results, train_mean_rocauc, test_mean_rocauc

def get_gtoi_bayes_factor(info_dict, data, nimg):
    """info_dict - dictionary object with paths to required files, and
    location of data. Also contains boolean values that used
    to load data: e.g. imaging, genomic
    data - dictionary object with data sets, and cv fold indices (response, imaging, ect.)
    nimg - integer, number of imaging features
    """
    ncv, bd = int(data["cv"].shape[1]/2), info_dict["base_dir"]
    results = {
        "cv{}".format(cv+1):{
            "bf":np.zeros((nimg, 1)), "bflog10":np.zeros((nimg, 1))
         } for cv in range(ncv)
    }
    bfcvf = info_dict["bf_cv_file"]

    for cv_idx in range(ncv):
        cv = cv_idx + 1
        path = os.path.join(bd, bfcvf.format(cv))
        df = pd.read_csv(path, index_col=None)

        for (eidx, file_path) in enumerate(df["path"]):
            bfcvfile, ni = pd.read_csv(file_path, index_col=0), df["idx"][eidx]
            results["cv{}".format(cv)]["bf"][ni-1] = bfcvfile["comp"]["bf"]
            results["cv{}".format(cv)]["bflog10"][ni-1] = bfcvfile["comp"]["ln10bf"]

    return results

def get_itod_pip(info_dict, data, nimg):
    """info_dict - dictionary object with paths to required files, and
    location of data. Also contains boolean values that used
    to load data: e.g. imaging, genomic
    data - dictionary object with data sets, and cv fold indices (response, imaging, ect.)
    nimg - integer, number of imaging features
    """
    ncv, bd = int(data["cv"].shape[1]/2), info_dict["base_dir"]
    results = {}
    fxvbcv = info_dict["fxvb_cv"]

    for cv_idx in range(ncv):
        cv = cv_idx + 1
        path = os.path.join(
            bd, fxvbcv.format(cv), "fxvb_out_cv_{}".format(cv),
            "logw_w_sa_logodds_pip_beta.csv"
        )
        results["cv{}".format(cv)] = pd.read_csv(path, index_col=None)

    return results

"""
TODO:

def manhattan_plot():
    sns.set_style('white')
    colors = np.tile(['darkred','black'], 11)
    name = 'RPS'
"""
