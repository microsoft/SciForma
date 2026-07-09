# Referenced https://github.com/open-mmlab/mmdetection
import argparse
import os
import os.path as osp

from mmengine import Config, DictAction


def parse_config(config_dir=None, train=False):
    if config_dir is None:
        parser = argparse.ArgumentParser()
        parser.add_argument('config_dir', type=str)
        parser.add_argument(
            '--cfg_options',
            nargs='+',
            action=DictAction,
            help='override some settings in the used config, the key-value pair '
            'in xxx=yyy format will be merged into config file. If the value to '
            'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
            'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
            'Note that the quotation marks are necessary and that no white space '
            'is allowed.')
        args = parser.parse_args()
        config = Config.fromfile(args.config_dir)
        config.config_dir = args.config_dir
        if args.cfg_options is not None:
            config.merge_from_dict(args.cfg_options)
    else:
        config = Config.fromfile(config_dir)
        config.config_dir = config_dir
    if train:
        config.logging_dir = osp.join(config.model_output_dir, config.logging_dir)
    else:
        config.logging_dir = osp.join(config.image_output_dir, config.logging_dir)
    #config.logging_dir = osp.join(config.output_dir, config.logging_dir)

    if config_dir is None:
        # When using PyTorch version >= 2.0.0, the `torch.distributed.launch`
        # will pass the `--local-rank` parameter to `tools/train.py` instead
        # of `--local_rank`.
        parser.add_argument('--local_rank', '--local-rank', type=int, default=-1)
        args = parser.parse_args()
        if 'LOCAL_RANK' not in os.environ:
            os.environ['LOCAL_RANK'] = str(args.local_rank)

    print(config)
    return config
