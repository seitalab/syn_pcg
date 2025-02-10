import os
import sys
from argparse import Namespace
from typing import Dict, Tuple

import torch
from scipy.special import softmax
from sklearn.metrics import classification_report

sys.path.append("..")
from codes.trainer import ModelTrainer
from common.utils import get_timestamp

class ModelEvaluator(ModelTrainer):

    def __init__(self, args: Namespace, dump_loc: str, device: str) -> None:
        """
        Args:
            args (Namespace):
            dump_loc (str):
            device (str):
        Returns:
            None
        """
        self.args = args
        self.args.device = device

        self.device = device
        self.model = None

        timestamp = get_timestamp()
        self.dump_loc = os.path.join(dump_loc, timestamp)

        os.makedirs(self.dump_loc, exist_ok=True)

    def set_weight(self, weight_file):
        assert (self.model is not None)

        self.model.to("cpu")

        # Temporal solution.
        state_dict = dict(torch.load(weight_file, map_location="cpu")) # OrderedDict -> dict

        old_keys = list(state_dict.keys())
        for key in old_keys:
            new_key = key.replace("module.", "")
            state_dict[new_key] = state_dict.pop(key)
        self.model.load_state_dict(state_dict)

        self.model.to(self.device)

    def run(self, loader, dump_errors=False) -> Tuple[float, float]:
        """
        Args:
            loader
        Returns:
            eval_score (float):
            eval_loss (float):
        """
        assert not dump_errors
        result_dict = self._evaluate(loader, store_sample=False)
        report = classification_report(
            result_dict["y_trues"], 
            softmax(result_dict["y_preds"], axis=1).argmax(axis=1), 
            digits=5, 
            zero_division=0.0
        )
        return result_dict, report    