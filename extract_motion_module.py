import argparse
from collections import OrderedDict
import torch


def extract_motion_module(unet_path, mm_path):
    unet = torch.load(unet_path, map_location="cpu")
    mm_state_dict = OrderedDict()
    state_dict = unet['state_dict']
    state_dict = {key.replace('module.', '', 1): value for key, value in state_dict.items()}
    for key in state_dict:
        if "motion_module" in key:
            mm_state_dict[key] = state_dict[key]

    torch.save(mm_state_dict, mm_path)

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--unet_path", type=str, required=True)
    argparser.add_argument("--mm_path", type=str, required=True)
    args = argparser.parse_args()
    extract_motion_module(args.unet_path, args.mm_path)