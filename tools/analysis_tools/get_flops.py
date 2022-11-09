# Copyright (c) OpenMMLab. All rights reserved.
import argparse

import torch
from fvcore.nn import FlopCountAnalysis
from fvcore.nn import flop_count_table

from mmcv import Config
from mmcv.cnn.utils import get_model_complexity_info

from mmselfsup.models import build_algorithm


def parse_args():
    parser = argparse.ArgumentParser(description='Get model flops and params')
    parser.add_argument('config', help='config file path')
    parser.add_argument(
        '--shape',
        type=int,
        nargs='+',
        default=[224, 224],
        help='input image size')
    args = parser.parse_args()
    return args


def main():

    args = parse_args()

    if len(args.shape) == 1:
        input_shape = (1, 3, args.shape[0], args.shape[0])
    elif len(args.shape) == 2:
        input_shape = (1, 3, ) + tuple(args.shape)
    else:
        raise ValueError('invalid input shape')

    cfg = Config.fromfile(args.config)
    model = build_algorithm(cfg.model)
    model.eval()

    flops = FlopCountAnalysis(model, (torch.rand(input_shape),))
    print(flop_count_table(flops))


if __name__ == '__main__':
    main()