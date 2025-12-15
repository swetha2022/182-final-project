"""
Code adapted from https://github.com/khurramjaved96/mrcl/blob/master/model/learner.py
For more information, see https://proceedings.neurips.cc/paper_files/paper/2019/file/f4dd765c12f2ef67f98f3558c282a9cd-Paper.pdf
"""

import logging
import math

import torch
from torch import nn
from torch.nn import functional as F

logger = logging.getLogger("experiment")


class Model(nn.Module):

    def __init__(self, config: list[dict]):
        super(Model, self).__init__()
        self.config = config
        self.vars = nn.ParameterList()
        self.vars = self.parse_config(self.config, nn.ParameterList())

    def parse_config(self, config, vars_list):

        for i, info_dict in enumerate(config):

            if info_dict["name"] == 'conv2d':
                conv_config = info_dict["config"]
                out_channels = conv_config["out-channels"]
                in_channels = conv_config["in-channels"]
                kernel_size = conv_config["kernal"]
                
                # Initialize weight tensor: (out_channels, in_channels, kernel_height, kernel_width)
                w = nn.Parameter(torch.empty(out_channels, in_channels, kernel_size, kernel_size))
                nn.init.kaiming_uniform_(w, a=math.sqrt(5))
                
                # Initialize bias tensor: (out_channels,)
                b = nn.Parameter(torch.empty(out_channels))
                fan_in = in_channels * kernel_size * kernel_size
                bound = 1 / math.sqrt(fan_in)
                nn.init.uniform_(b, -bound, bound)
                
                vars_list.append(w)
                vars_list.append(b)

            elif info_dict["name"] == 'linear':
                param_config = info_dict["config"]
                out_features = param_config["out"]
                in_features = param_config["in"]
                
                # Initialize weight tensor: (out_features, in_features)
                w = nn.Parameter(torch.empty(out_features, in_features))
                nn.init.kaiming_uniform_(w, a=math.sqrt(5))
                
                # Initialize bias tensor: (out_features,)
                b = nn.Parameter(torch.empty(out_features))
                fan_in = in_features
                bound = 1 / math.sqrt(fan_in)
                nn.init.uniform_(b, -bound, bound)
                
                vars_list.append(w)
                vars_list.append(b)

            elif info_dict["name"] in ['tanh', 'rep', 'relu', 'upsample', 'avg_pool2d', 'max_pool2d',
                                       'flatten', 'reshape', 'leakyrelu', 'sigmoid', 'rotate']:
                continue
            else:
                print(info_dict["name"])
                raise NotImplementedError
        return vars_list

    def add_rotation(self):
        self.rotate = nn.Parameter(torch.ones(2304,2304))
        torch.nn.init.uniform_(self.rotate)
        self.rotate_inverse = nn.Parameter(torch.inverse(self.rotate))
        logger.info("Inverse computed")


    def forward(self, x, vars=None, config=None, sparsity_log=False, rep=False):
        x = x.float()
        if vars is None:
            vars = self.vars

        if config is None:
            config = self.config

        idx = 0

        for layer_counter, info_dict in enumerate(config):
            name = info_dict["name"]
            if name == 'conv2d':
                w, b = vars[idx], vars[idx + 1]
                x = F.conv2d(x, w, b, stride=info_dict['config']['stride'], padding=info_dict['config']['padding'])
                idx += 2

            elif name == 'linear':
                w, b = vars[idx], vars[idx + 1]
                x = F.linear(x, w, b)
                idx += 2

            elif name == 'flatten':
                x = x.view(x.size(0), -1)

            elif name == 'rotate':
                # pass
                x = F.linear(x, self.rotate)
                x = F.linear(x, self.rotate_inverse)

            elif name == 'reshape':
                continue

            elif name == 'rep':
                if rep:
                    return x

            elif name == 'relu':
                x = F.relu(x)

            else:
                raise NotImplementedError
        assert idx == len(vars)
        return x

    def update_weights(self, vars):
        for old, new in zip(self.vars, vars):
            old.data = new.data

    def parameters(self):
        """
        override this function since initial parameters will return with a generator.
        :return:
        """
        return self.vars

