import os
from argparse import ArgumentParser
from nipype import Function, Node, Workflow, IdentityInterface

# TODO: 
#     1. check if it works

parser = ArgumentParser(prog="itod.py", description=__doc__)
parser.add_argument("-dp", "--data_path", type=str, help="full path to the data")
parser.add_argument("-bf", "--bayes_factor", type=str, help="full path to bayes factor")
parser.add_argument("-sn", "--save_name", type=str, help="name of output file")
parser.add_argument("-bp", "--bind_path", type=str, help="bind path for image")
parser.add_argument("-ip", "--img_path", type=str, help="path to the singularity image")
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

def fxvb(info_path, base_dir, data_path, save_name, bind_path, img_path):
    import os
    from subprocess import call
    f_name = "fxvb_run.csv"
    shfile = "\n".join([
        "#!/bin/bash", ("R_DEFAULT_PACKAGES= Rscript "
        "{script} {data_path} {bf_path} {save_path}")
    ])
    file_name = os.path.join(base_dir, "itod_template.r")
    r_file = shfile.format(
        script=file_name, data_path=data_path, bf_path=info_path,
        save_path=save_name
    )
    if os.getwd() != base_dir:
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
        input_names=["bayes_factor", "base_dir"],
        output_names=["info_path"]
        function=collect_varbvs
    ),
    name="Collect_Varbvs"
)

Fxvb = Node(Interface=Function(
        input_names=["info_path", "base_dir", "data_path", "save_name", "bind_path", "img_path"],
        output_names=[],
        function=fxvb
    ),
    name="Fxvb"
)

Collect_Varbvs.inputs.base_dir = wf.base_dir
Collect_Varbvs.inputs.base_factor = args.bayes_factor
wf.connect(Collect_Varbvs, "info_path", Fxvb, "info_path")
Fxvb.inputs.base_dir = wf.base_dir
Fxvb.inputs.data_path = args.data_path
Fxvb.inputs.save_name = args.save_name
Fxvb.inputs.bind_path = args.bind_path
Fxvb.inputs.img_path = args.img_path
wf.run()
