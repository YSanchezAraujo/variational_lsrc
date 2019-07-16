import os
import numpy as np
import pandas as pd
import h5py
import scipy.stats
from sklearn.model_selection import ShuffleSplit
from sklearn.metrics import roc_auc_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

def match_search(sitem, ref):
    sitem_list = sitem.split("_")
    refdict = {i:0 for i in ref}

    for si in sitem_list:
        for j in ref:
            if si in j:
                refdict[j] += 1

    max_num = max(refdict.values())
    outdict = {j:k for j, k in refdict.items() if k == max_num}
    return outdict

def plausible_matches(icols, ref_list):
    return {ic:match_search(ic, ref_list) for ic in icols}

def convert_new_cols(matched_dict):
    special_search = [
        "lh_S_central_thickness_D", "rh_S_central_thickness_D"
    ]

    out_orig = []
    out_comp = []

    for k, d in matched_dict.items():
        m = list(d.keys())
        if len(m) == 1:
            out_orig.append(m[0])
            out_comp.append(m[0].replace(".", "_"))
        else:
           for j in m:
               if k == j.replace(".", "_"):
                   out_orig.append(j)
                   out_comp.append(j.replace(".", "_"))

    return out_orig, out_comp

def read_pickle(path):
    with open(path, "rb") as file_store:
        data = pickle.load(file_store)
    return data

def save_pickle(path, data):
    with open(path, "wb") as file_store:
        pickle.dump(data, file_store, protocol=pickle.HIGHEST_PROTOCOL)
    return path

def pad(data, l, pad_val=123456789):
    padded_data = np.repeat(pad_val, l)
    padded_data[:len(data)] = data
    return padded_data

def regress_covars(I, Z):
    P = np.eye(Z.shape[0]) - Z.dot(np.linalg.pinv(Z.T.dot(Z))).dot(Z.T)
    return P.dot(I)

def make_cv_splits(nsamples, nfolds, test_size, pad_val=123456789):
    N_test = round(test_size*nsamples)
    N_train = int(nsamples - N_test)
    cols = [("train{}".format(k+1), "test{}".format(k+1)) for k in range(nfolds)]
    cvdf = pd.DataFrame(data=np.zeros((N_train, int(nfolds*2))), columns=[j for k in cols for j in k])
    X, y = np.random.random((nsamples, 1)), np.random.random(nsamples)
    cv = ShuffleSplit(n_splits=nfolds, test_size=test_size)

    for idx, (train, test) in enumerate(cv.split(X, y)):
        cvdf["train{}".format(idx+1)] = train+1
        cvdf["test{}".format(idx+1)] = pad(test+1, N_train, pad_val=pad_val)

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

def enforce_5050_keep_n(labels, c1, c2, test_percent):
    n = len(labels)
    about_half = int(np.floor(n/2))
    other_half = int(n - about_half)
    c1n = (labels == c1).sum()

    if c1n > about_half:
        c_to_resample = c2
        c_not_resample = c1
    else:
        c_to_resample = c1
        c_not_resample = c2

    crs = np.random.choice(np.where(labels == c_to_resample)[0], about_half, replace=True)
    cnrs = np.random.choice(np.where(labels == c_not_resample)[0], other_half, replace=False)
    all_idxs = np.array([idx for x in [crs, cnrs] for idx in x])
    num_test = int(np.floor(test_percent*n))
    num_train = int(n - num_test)

    if num_test + num_train != n:
        raise Exception("something went wrong")

    # now because of the resampling, this can get a bit messey so we need
    # to create dummy indices
    dummy_idx = np.arange(n)
    dummy_train = np.random.choice(dummy_idx, num_train, replace=False)
    dummy_test = np.setdiff1d(dummy_idx, dummy_train)
    train = all_idxs[dummy_train]
    test = all_idxs[dummy_test]

    return dict(
        resampled_class_label = c_to_resample,
        resampled_class_idx = crs,
        other_class_idx = cnrs,
        idxs = all_idxs,
        train_idxs = train,
        test_idxs = test
    )

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
    rel_vars = dict(read_imaging="I", read_genomic="G", read_depvar="y", read_covars="Z")

    for key in info_dict.keys():
        if "read" in key:
            if info_dict[key]:
                return_obj[rel_vars[key]] = read_h5(data_path, rel_vars[key])

    return_obj["cv"] = pd.read_csv(os.path.join(bd, info_dict["cv_idx"]))
    return_obj["I_cols"] = read_h5(data_path, "I_cols")
    return_obj["G_cols"] = read_h5(data_path, "G_cols")
    return_obj["Z_cols"] = read_h5(data_path, "Z_cols")
    return_obj["snpinfo"] = pd.read_csv(os.path.join(bd, info_dict["snp_info"]))

    return return_obj

def get_gtoi_pip_percv(info_dict, cv_idx, nsnp, nimg):
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

def get_gtoi_pip_perfeat(info_dict, feat_idx, nsnp, ncv):
    pip_data = np.zeros((nsnp,ncv))
    bwd, gtoi_res = info_dict["base_dir"], info_dict["gtoi_res"]

    use_dirs = [
        dir_name for dir_name in os.listdir(os.path.join(info_dict["base_dir"], gtoi_res))
        if "g2i_result_col_idx{}_cv_".format(feat_idx) in dir_name
    ]

    for dir_name in use_dirs:
        wd = os.path.join(info_dict["base_dir"], gtoi_res, dir_name)
        cv_idx = int(dir_name.split("_")[-1])
        pip_data[:, cv_idx-1] = pd.read_csv(
            os.path.join(wd, "pip_beta.csv"), index_col=None
        )["pip"].values

    return pip_data

def predict_with_param(I, y, param, train, test, use_prob=False):
    """this function is not mean to be used outright, only meant to be called
    by the function predict_all_folds
    """
    f_prob = lambda x: scipy.stats.bernoulli.rvs(x)
    f = lambda x: 1 if x >= .5 else 0
    a, b = np.exp(I[train-1].dot(param)), np.exp(I[test-1].dot(param))
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
        train=(probs_train, yhat_train, roc_auc_score(y[train-1], yhat_train)),
        test=(probs_test, yhat_test, roc_auc_score(y[test-1], yhat_test))
    )

def predict_all_folds(info_dict, data, use_param, use_prob=False, pad_val=123456789):
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
        test = test[test != pad_val]
        print(test.shape)
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

def predict_validation(info_dict, I, y, l1models, l1coefs):
    probvalres = np.zeros(100)
    l1res = np.zeros(100)
    f = lambda x: 1 if x >= 0.5 else 0
    class_map = {1:1, 0:-1}

    for cv_idx in range(100):
        cv = cv_idx + 1
        path = os.path.join(info_dict["base_dir"], info_dict["itod_res"].format(cv),
                            "fxvb_out_cv_{}".format(cv), "logw_w_sa_logodds_pip_beta.csv")
        param = pd.read_csv(path, index_col=None)["beta"].values
        Xb_hat = I.dot(param)
        prob = np.exp(Xb_hat)/(1 + np.exp(Xb_hat))
        probvalres[cv_idx] = roc_auc_score(y, [f(pr) for pr in prob])
        l1models[cv_idx].coef_ = l1coefs[:, cv_idx][None, :]
        l1res[cv_idx] = roc_auc_score([class_map[i] for i in y], l1models[cv_idx].predict(I))

    return probvalres, l1res

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

def get_snpnpip(snp_info, pips, gcols):
    gcols = np.array([i.decode("utf-8") for i in gcols])
    df1 = pd.DataFrame(
        data=np.hstack((gcols[:, None], pips[:, None])),
         columns=["snpid", "pip"]
    )
    if sum(df1["snpid"] == snp_info["snpid"]) == snp_info.shape[0]:
        snp_info["pip"] = df1["pip"].astype(np.float64)
    else:
        raise Exception("couldn't match row labels")
    snp_info["log10pip"] = np.log10(snp_info["pip"])
    snp_info["chrnum"] = [int(i[3:]) for i in snp_info["hg19chrc"]]
    snp_info = snp_info.sort_values("chrnum")
    snp_info["ind"] = np.arange(len(snp_info))
    snp_info = snp_info.reset_index(drop=True)
    grouped = snp_info.groupby(("chrnum"))
    return snp_info, grouped

def manhattan_plot(grouped_snp_info, plot_name, pip_thresh, pip_col, xlims, ylims):
    sns.set_style('white')
    colors = np.tile(['darkred','black'], 11)
    name = 'most_sig_brain_region'
    fig = plt.figure(figsize=(20,12))
    ax = fig.add_subplot(111)
    x_labels, x_labels_pos = [], []

    for num, (name, group) in enumerate(grouped_snp_info):
        group.plot(
            kind="scatter", x="ind", y=pip_col, color=colors[num % len(colors)],
            ax = ax, s=40
        )
        tf_vals = group[pip_col] > pip_thresh

        if np.any(tf_vals):
            x_labels.append(name)
            x_locs = group["ind"][tf_vals].values
            y_locs = group[pip_col][tf_vals].values
            snps = group["snpid"][tf_vals].values
            for i in range(len(x_locs)):
                plt.annotate("{}".format(snps[i]), xy=(x_locs[i], y_locs[i]), size=10) # add value hurts for nonlogpip
        else:
            x_labels.append("")

        x_labels_pos.append(
            (group["ind"].iloc[-1] - (group["ind"].iloc[-1] - group["ind"].iloc[0]) / 2)
        )

    ax.set_xticks(x_labels_pos)
    ax.set_xticklabels(x_labels)
    #ax.set_xlim([xlims[0], xlims[1]])
    #ax.set_ylim([ylims[0], ylims[1]])
    plt.xlabel("Chromosome", fontsize=20)
    plt.ylabel(pip_col, fontsize=20)
    plt.savefig("manhattan_pip_brainreg_{}.png".format(plot_name),dpi=300, bbox_inches="tight")
    plt.close()
