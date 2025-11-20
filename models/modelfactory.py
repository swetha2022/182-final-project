"""
Code adapted from https://github.com/khurramjaved96/mrcl/blob/master/model/modelfactory.py
For more information, see https://proceedings.neurips.cc/paper_files/paper/2019/file/f4dd765c12f2ef67f98f3558c282a9cd-Paper.pdf
"""

import numpy as np

class ModelFactory():
    def __init__(self):
        pass

    @staticmethod
    def get_model(dataset):

        if dataset == "omniglot":
            channels = 256

            return [
                {"name": 'conv2d',
                 "config": {"out-channels": channels, "in-channels": 1, "kernal": 3, "stride": 2, "padding": 0}},
                {"name": 'relu'},

                {"name": 'conv2d',
                 "config": {"out-channels": channels, "in-channels": channels, "kernal": 3, "stride": 1,
                            "padding": 0}},
                {"name": 'relu'},

                {"name": 'conv2d',
                 "config": {"out-channels": channels, "in-channels": channels, "kernal": 3, "stride": 2,
                            "padding": 0}},
                {"name": 'relu'},
                #
                {"name": 'conv2d',
                 "config": {"out-channels": channels, "in-channels": channels, "kernal": 3, "stride": 1,
                            "padding": 0}},
                {"name": 'relu'},

                {"name": 'conv2d',
                 "config": {"out-channels": channels, "in-channels": channels, "kernal": 3, "stride": 2,
                            "padding": 0}},
                {"name": 'relu'},

                {"name": 'conv2d',
                 "config": {"out-channels": channels, "in-channels": channels, "kernal": 3, "stride": 2,
                            "padding": 0}},
                {"name": 'relu'},

                {"name": 'flatten'},
                # {"name": 'rotate'},
                {"name": 'rep'},

                {"name": 'linear',
                 "config": {"out": 1623, "in": 9 * channels}}
            ]

        else:
            print("Unsupported model; either implement the model in model/ModelFactory or choose a different model")
            assert (False)