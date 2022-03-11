import os
import torch
import numpy as np
import argparse
from config import TfConfig



def get_cmd():
    # paras:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", default="CVPR", help="CVPR (4096-dim) YC2 dataset")
    parser.add_argument("--model_type", default="full", help="model types")
    parser.add_argument("--train_test", default="test", help="training or testing")
    parser.add_argument("--model_dir", default="./checkpoint/", help="model directory")
    parser.add_argument("--initialize", default=None, help="initialization")
    parser.add_argument("--train_mode", default="ACL_Spatial", help="train mode")
    args = parser.parse_args()
    return args


def main(args, paras):
    components = args.model_name[paras.model_type]
    full_model = {}
    for item in components:
        compo_file_dir = paras.model_dir + item + '_Best.pkl'
        if not os.path.exists(compo_file_dir):
            print('Missing component of '+ item + ', exiting...')
            exit()
        full_model[item] = torch.load(paras.model_dir+ item + '_Best.pkl')
    torch.save(full_model, paras.model_dir + 'Model_Best.pkl')
    print('Best model successfully generated in ' + paras.model_dir + 'Model_Best.pkl')


if __name__ == "__main__":
    paras = get_cmd()
    args = TfConfig()
    main(args, paras)