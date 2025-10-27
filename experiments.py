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
nb_samples = 5
datasets = ["officer" ] #"actors", "movies", "directors", "city", "country", "entity", "officer", "airports"
heuristics = ["lp", "sls"] # "ls", "sls", "lp", "rd"


def run_exp(conf):
    path_str = log_dir + "_".join(map(str, conf)) + ".log"
    if path.exists(path_str):
        print("Ignored", conf)
        return path_str

    with open(path_str, mode="w", buffering=1) as out:  # line-buffered
        print(conf, file=out, flush=True)
        with Popen(
            [python_bin, script_path, "--dataset", conf[0], "--method", conf[1], "-k", str(conf[2])],
            stdout=out,
            stderr=out,      # merge stderr into the same log
            text=True        # get str, not bytes
        ) as process:
            rc = process.wait()
        print("Finished", conf, "rc=", rc, file=out, flush=True)
    return path_str

if __name__ == '__main__':
    configs = []
    # Generate all test configs
    for data_config in datasets:
        for heuristic in heuristics:
            for k in range(2,6):
                if heuristic != "exp":
                    sids = range(nb_samples)
                else:
                    sids = [1]
                for sample_id in sids:
                        configs.append((data_config, heuristic, k, sample_id))

    random.shuffle(configs)

    print("Created", len(configs), "experimental configuration. Saving to", log_dir + "_to_run.cfg")

    with open(log_dir + "_to_run.cfg", mode="w") as out:
        for c in configs:
            print(c, file=out)

    #with Pool(threads) as pool:
       # print(pool.map(run_exp, configs))
    for c in configs:
        print(run_exp(c))