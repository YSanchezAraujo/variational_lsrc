import os
from argparse import ArgumentParser
from nipype import Function, Node, Workflow, IdentityInterface

# TODO: 
#     1. run the brain to diagnosis step with the bayes factor as prior logodds
#     2. save results

import os
from argparse import ArgumentParser
from nipype import Function, Node, Workflow, IdentityInterface

parser = ArgumentParser(prog="itod.py", description=__doc__)
parser.add_argument("-dp", "--data_path", type=str, help="full path to the data")
parser.add_argument("-bf", "--bayes_factor", type=str, help="full path to bayes factor")
parser.add_argument("-sn", "--save_name", type=str, help="name of output file")
args = parser.parse_args()

wf = Workflow(name="varbvs-s2")
wf.base_dir = os.getcwd()

def collect_varbvs(bayes_factor, base_dir):
    import os
    import pandas as pd
    files = [[x]+x.split('-')[1:] for x in os.listdir(bayes_factor)]
    df = pd.DataFrame(files)
    df.columns = ["path","idx", "name"]
    df["path"] = [os.path.join(bayes_factor, p) for p in df["path"]]
    df["exist"] = [os.path.isfile(p) for p in df["path"]]
    df["idx"] = df["idx"].astype(int)
    df = df.sort_values("idx").reset_index(drop=True)
    df["exist"] = df["exist"].astype(int)
    os.chdir(base_dir)
    df.to_csv("bf_info_for_fxvb.csv", index=False)
    info_path = os.path.join(base_dir, "bf_info_for_fxvb.csv")
    return info_path
