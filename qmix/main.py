import os, sys
import numpy as np
import torch
import yaml

from train import train

print(__file__)

def my_main(config_dict):
    np.random.seed(config_dict["seed"])
    torch.manual_seed(config_dict["seed"])

    train(config_dict)

if __name__ == '__main__':
    with open(os.path.join(os.path.dirname(__file__), "config", "default.yaml"), "r") as f:
        try:
            config_dict = yaml.safe_load(f)
        except yaml.YAMLError as exc:
            assert False, "default.yaml error: {}".format(exc)
    print(config_dict)

    learner_name = None
    if "learner" in config_dict:
        learner_name = config_dict['learner']
    else:
        learner_name = "iql"

    with open(os.path.join(os.path.dirname(__file__), "config", "algo", "{}.yaml".format(learner_name)), "r") as f:
        try:
            learner_config_dict = yaml.safe_load(f)
        except yaml.YAMLError as exc:
            assert False, "{}.yaml error: {}".format("algo", exc)
    config_dict.update(learner_config_dict)
    my_main(config_dict)

