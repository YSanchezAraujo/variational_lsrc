import os
from argparse import ArgumentParser
from nipype import Function, Node, Workflow, IdentityInterface

parser = ArgumentParser(prog="itod.py", description=__doc__)
parser.add_argument("-dp", "--data_path", type=str, help="full path to the data")
parser.add_argument("-bf", "--bayes_factor", type=str, help="full path to bayes factor csv files")
parser.add_argument("-sn", "--save_name", type=str, help="name of output file")
parser.add_argument("-bp", "--bind_path", type=str, help="bind path for image")
parser.add_argument("-ip", "--img_path", type=str, help="path to the singularity image")
parser.add_argument("-nc", "--num_cv", type=int, help="number of cv iterations", required=False)
parser.add_argument("-cp", "--cv_path", type=str, help="path to cv file", required=False)
args = parser.parse_args()

wf = Workflow(name="varbvs-s2")
wf.base_dir = os.getcwd()

Iternode = Node(IdentityInterface(fields=["cv_iter"]), name="Iternode")
Iternode.iterables = ("cv_iter", [i+1 for i in range(args.num_cv)])

def collect_varbvs(bayes_factor, base_dir, cv_iter):
    import os
    import pandas as pd
    files = []
    bayes_outputs = os.listdir(bayes_factor)
    match_str = "cv_{}_".format(cv_iter)
    for out in bayes_outputs:
        if  match_str in out:
            files.append([out] + out.split('-')[1:])
    df = pd.DataFrame(files)
    df.columns = ["path", "idx", "name"]
    df["path"] = [os.path.join(bayes_factor, p) for p in df["path"]]
    df["exist"] = [os.path.isfile(p) for p in df["path"]]
    df["idx"] = df["idx"].astype(int)
    df = df.sort_values("idx").reset_index(drop=True)
    os.chdir(base_dir)
    df.to_csv("bf_info_cv_{}_for_fxvb.csv".format(cv_iter), index=False)
    info_path = os.path.join(base_dir, "bf_info_cv_{}_for_fxvb.csv".format(cv_iter))
    return info_path, cv_iter

def fxvb(info_path, base_dir, data_path, save_name, bind_path, img_path, cv_iter, cv_path):
    import os
    from subprocess import call
    f_name = "fxvb_run_cv_{}.csv".format(cv_iter)
    shfile = "\n".join([
        "#!/bin/bash", ("R_DEFAULT_PACKAGES= Rscript "
        "{script} {data_path} {bf_path} {save_path} {cv_path} {cv_idx}")
    ])
    file_name = os.path.join(base_dir, "itod_cv_template.r")
    r_file = shfile.format(
        script=file_name, data_path=data_path, bf_path=info_path,
        save_path=save_name, cv_path=cv_path, cv_idx=cv_iter
    )
    if os.getcwd() != base_dir:
        os.chdir(base_dir)
    wd = f_name.split(".")[0]
    os.makedirs(wd)
    os.chdir(wd)
    rf = "r_file_{}.sh".format(wd)
    with open(rf, 'w') as f:
        f.write(r_file)
    script_path = os.path.join(os.getcwd(), rf)
    singfile = "\n".join([
        "#!/bin/bash", "#SBATCH --mem=4G", "#SBATCH -c 2",
        "#SBATCH --time=04:00:00", ("singularity exec --bind "
        "{bind_path} {img_path} {cmd}")
    ])
    singfile = singfile.format(bind_path=bind_path, img_path=img_path,
                               cmd="bash {}".format(script_path))
    sif = os.path.join(os.getcwd(), "sbatch_file-fxvb.sh")
    with open(sif, 'w') as sf:
        sf.write(singfile)
    call("sbatch {}".format(sif).split())
    return None

Collect_Varbvs = Node(interface=Function(
        input_names=["bayes_factor", "base_dir", "cv_iter"],
        output_names=["info_path", "cv_iter"],
        function=collect_varbvs
    ),
    name="Collect_Varbvs"
)

Fxvb = Node(interface=Function(
        input_names=[
            "info_path", "base_dir", "data_path",
            "save_name", "bind_path", "img_path",
            "cv_iter", "cv_path"
        ],
        output_names=[],
        function=fxvb
    ),
    name="Fxvb"
)

wf.connect(Iternode, "cv_iter", Collect_Varbvs, "cv_iter")
Collect_Varbvs.inputs.base_dir = wf.base_dir
Collect_Varbvs.inputs.bayes_factor = args.bayes_factor
wf.connect(Collect_Varbvs, "info_path", Fxvb, "info_path")
wf.connect(Collect_Varbvs, "cv_iter", Fxvb, "cv_iter")
Fxvb.inputs.base_dir = wf.base_dir
Fxvb.inputs.data_path = args.data_path
Fxvb.inputs.save_name = args.save_name
Fxvb.inputs.bind_path = args.bind_path
Fxvb.inputs.img_path = args.img_path
Fxvb.inputs.cv_path = args.cv_path
wf.run()
