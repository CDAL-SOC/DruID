import numpy as np
import pandas as pd

import logging
import os

import torch

from functools import cached_property


from sklearn.metrics import pairwise_distances


DATA_BASE_DIR = "../data/"


class BaseDruidModel(torch.nn.Module):
    """
    Base Class for all Druid models - provides necessary template with some default
    implementations to simplify comparisons among models

    """

    SAVE_BASE_PATH = DATA_BASE_DIR

    @cached_property
    def device(self):
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def preprocess(self, dataset):
        """
        Make necessary modifications to the input dataset required for making inference

        """
        return dataset

    def postprocess(self, dataset, model_out):
        """
        Hook to format/transform the output from the model that is ready for further analysis
        Eg. tranform the tensor output into a human-consumable dataframe

        """
        return model_out

    @cached_property
    def model_state_dir(self):
        return f"{self.SAVE_BASE_PATH}local_checkpoints/{type(self).__name__}/"

    @cached_property
    def model_state_path(self):
        return f"{self.model_state_dir}model.pth"

    def load_model(self):
        if os.path.exists(self.model_state_dir):
            self.load_state_dict(
                torch.load(self.model_state_path, map_location=str(self.device),)
            )
        else:
            print(f"Model state not found at {self.model_state_path}")

    def save_model(self):
        if not os.path.exists(self.model_state_dir):
            os.makedirs(self.model_state_dir)

        torch.save(self.state_dict(), self.model_state_path)

    def train_model(self, epoch):
        raise NotImplementedError(
            f"{type(self).__name__} has not implemented train_model()"
        )

    def forward(self, dataset):
        raise NotImplementedError(
            f"{type(self).__name__} has not implemented forward()"
        )

