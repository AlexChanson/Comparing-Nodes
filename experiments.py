import os
from subprocess import PIPE, Popen
import subprocess
import platform
from multiprocessing import Pool
import random
from os import path


log_dir = "/works/logs/cn_experiments/"
script_path = "./main.py"
python_bin = "python3"
threads = 16
nb_samples = 20
datasets = ["airports", "iris"]
heuristics = ["ls", "smart-start"]


def run_exp(conf):
    path_str = "_".join(map(str, conf)) + ".log"
    if path.exists(path_str):
        print("Ignored", conf)
        return
    with open(path_str, mode="w") as out:
        print(conf)
        with Popen([python_bin, script_path, "--dataset", conf[0], "-h", conf[1], "-k", conf[2]], stdout=out, stderr=PIPE) as process:
            output = process.communicate()[0].decode("utf-8")
            print(output, file=out)
        print("Finished", conf)

if __name__ == '__main__':
    configs = []
    # Generate all test configs
    for data_config in datasets:
        for heuristic in heuristics:
            for k in range(2,5):
                for sample_id in range(nb_samples):
                        configs.append((data_config, heuristic, k, sample_id))

    random.shuffle(configs)

    print("Created", len(configs), "experimental configuration. Saving to", log_dir + "_to_run.cfg")

    with open(log_dir + "_to_run.cfg", mode="w") as out:
        for c in configs:
            print(c, file=out)

    with Pool(threads) as pool:
        print(pool.map(run_exp, configs))