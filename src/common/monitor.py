import os
import json
import math
import pickle
from typing import Dict
# from PIL import Image

import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import (
    f1_score, 
    roc_auc_score, 
    confusion_matrix,
    accuracy_score, 
    average_precision_score,
    recall_score,
    precision_score
)
from scipy.special import softmax

def sigmoid(a):
    return 1 / (1 + np.exp(-a))

def specificity_score(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    specificity = tn / (tn + fp) if (tn + fp) != 0 else 0
    return specificity

class Monitor:

    def __init__(self) -> None:
        """
        Args:
            None
        Returns:
            None
        """
        self.num_data = 0
        self.total_loss = 0
        self.ytrue_record = None
        self.ypred_record = None

        self.inputs = None

    def _concat_array(self, record, new_data: np.array) -> np.ndarray:
        """
        Args:
            record ()
            new_data (np.ndarray):
        Returns:
            concat_data (np.ndarray):
        """
        if record is None:
            return new_data
        else:
            return np.concatenate([record, new_data])

    def store_loss(self, loss: float) -> None:
        """
        Args:
            loss (float): Mini batch loss value.
        Returns:
            None
        """
        self.total_loss += loss

    def store_num_data(self, num_data: int) -> None:
        """
        Args:
            num_data (int): Number of data in mini batch.
        Returns:
            None
        """
        self.num_data += num_data

    def store_result(self, y_trues: np.ndarray, y_preds: np.ndarray) -> None:
        """
        Args:
            y_trues (np.ndarray):
            y_preds (np.ndarray): Array with 0 - 1 values.
        Returns:
            None
        """
        y_trues = y_trues.cpu().detach().numpy()
        y_preds = y_preds.cpu().detach().numpy()

        self.ytrue_record = self._concat_array(self.ytrue_record, y_trues)
        self.ypred_record = self._concat_array(self.ypred_record, y_preds)
        assert(len(self.ytrue_record) == len(self.ypred_record))

    def store_input(self, input_x):

        input_x = input_x.cpu().detach().numpy()

        self.inputs = self._concat_array(self.inputs, input_x)

    def average_loss(self) -> float:
        """
        Args:
            None
        Returns:
            average_loss (float):
        """
        return self.total_loss / self.num_data

    def _pred_val_to_label(self) -> np.ndarray:
        y_preds = softmax(self.ypred_record, axis=1)
        y_preds = np.argmax(y_preds, axis=1)
        return y_preds

    def calc_f1(self) -> float:
        """
        Args:
            None
        Returns:
            score (float): F1 score.
        """
        y_preds = self._pred_val_to_label()
        score = f1_score(self.ytrue_record, y_preds)
        return score

    def accuracy(self) -> float:
        """
        Args:
            None
        Returns:
            score (float):
        """            
        y_preds = self._pred_val_to_label()
        score = accuracy_score(self.ytrue_record, y_preds)
        return score
    
    def recall_score(self):
        y_preds = self._pred_val_to_label()
        score = recall_score(self.ytrue_record, y_preds, zero_division=0)
        return score

    def precision_score(self):
        y_preds = self._pred_val_to_label()
        score = precision_score(self.ytrue_record, y_preds, zero_division=0)
        return score

    def specificity_score(self):
        y_preds = self._pred_val_to_label()
        score = specificity_score(self.ytrue_record, y_preds)
        return score

    def roc_auc_score(self) -> float:
        """
        Args:
            None
        Returns:
            score (float): AUC-ROC score.
        """
        # y_preds = sigmoid(self.ypred_record[:,1])
        y_preds = softmax(self.ypred_record, axis=1)[:, 1]

        score = roc_auc_score(self.ytrue_record, y_preds)
        return score
    
    def count(self, target):
        y_preds = self._pred_val_to_label()
        tn, fp, fn, tp = confusion_matrix(self.ytrue_record, y_preds).ravel()
        if target == "tp":
            return tp
        elif target == "fp":
            return fp
        elif target == "fn":
            return fn
        elif target == "tn":
            return tn
        else:
            raise
    
    def average_precision_score(self):
        """
        Args:

        Returns:    

        """
        y_preds = sigmoid(self.ypred_record[:,1])
        score = average_precision_score(
            self.ytrue_record, y_preds)
        return score

    def get_confmat_text(self):
        y_preds = self._pred_val_to_label()

        confmat = confusion_matrix(
            self.ytrue_record, y_preds)
        return np.array2string(confmat, separator=', ')

    def show_result(self) -> None:
        """
        Args:
            is_multilabel (bool): 
        Returns:
            None
        """
        y_preds = self._pred_val_to_label()
        conf_matrix = confusion_matrix(self.ytrue_record, y_preds)
        print("Confusion Matrix")
        print(conf_matrix)
        print(f"Macro F1: {self.calc_f1():.4f}")
        print(f"Average Loss: {self.average_loss():.4f}")

class Storer:

    def __init__(
        self, 
        save_dir: str, 
        store_interim_model: bool=False
    ):
        """
        Args:
            save_dir (str): Path to save dir
        Returns:
            None
        """
        os.makedirs(save_dir, exist_ok=True)
        if store_interim_model:
            os.makedirs(save_dir+"/interims", exist_ok=True)
        self.save_dir = save_dir

        self.trains = {"loss": {}, "score": {}}
        self.evals = {"loss": {}, "score": {}}
        self.save_samples = True

    def save_params(self, params) -> None:
        """
        Save parameters.
        Args:
            params
        Returns:
            None
        """
        savename = self.save_dir + "/params.pkl"
        with open(savename, "wb") as fp:
            pickle.dump(params, fp)
        
        # Convert to json format
        params_json = json.dumps(vars(params), indent=4)
        with open(self.save_dir + "/params.json", "w") as f:
            f.write(params_json)

    def save_model(self, model: nn.Module, score: float) -> None:
        """
        Save current model (overwrite existing model).
        Args:
            model (nn.Module):
            score (float):
        Returns:
            None
        """
        savename = self.save_dir + "/net.pth"
        torch.save(model.state_dict(), savename)

        with open(self.save_dir + "/best_score.txt", "w") as f:
            f.write(f"{score:.5f}")

    def store_epoch_result(
        self, 
        epoch: int, 
        epoch_result_dict: Dict, 
        is_eval: bool = False,
        store_all: bool = False
    ) -> None:
        """
        Args:
            epoch (int):
            score (float):
        Returns:
            None
        """
        if not store_all:
            if is_eval:
                self.evals["loss"][epoch] = epoch_result_dict["loss"]
                self.evals["score"][epoch] = epoch_result_dict["score"]
            else:
                self.trains["loss"][epoch] = epoch_result_dict["loss"]
                self.trains["score"][epoch] = epoch_result_dict["score"]
        else:
            target_types = [float, int, np.float64]
            if is_eval:
                for key in epoch_result_dict.keys():
                    if type(epoch_result_dict[key]) not in target_types:
                        continue
                    if key not in self.evals.keys():
                        self.evals[key] = {}
                    self.evals[key][epoch] = epoch_result_dict[key]
            else:
                for key in epoch_result_dict.keys():
                    if type(epoch_result_dict[key]) not in target_types:
                        continue
                    if key not in self.trains.keys():
                        self.trains[key] = {}
                    self.trains[key][epoch] = epoch_result_dict[key]

    def store_logs(self, n_samples=None):
        """
        Args:
            None
        Returns:
            None
        """

        with open(self.save_dir + "/train_scores.json", "w") as ft:
            json.dump(self.trains, ft, indent=4)

        with open(self.save_dir + "/eval_scores.json", "w") as fe:
            json.dump(self.evals, fe, indent=4)

    def save_model_interim(self, model, n_sample, denom=1e6):
        """
        Args:

        Returns:
            None
        """
        if n_sample < denom:
            # calculate appropriate denominator based on n_sample.
            # eg. n_sample = 1500, denom = 1e2, n_sample = 150, denom=1e1
            denom = 10 ** (len(str(n_sample)) - 2)

        power = round(math.log(denom, 10), 3)
        n_sample_d = n_sample / denom
        info = f"{int(n_sample_d):06d}E{power:.2f}"

        savename = self.save_dir + f"/interims/net_{info}.pth"
        torch.save(model.state_dict(), savename)


class EarlyStopper:

    def __init__(self, mode: str, patience: int):
        """
        Args:
            mode (str): max or min
            patience (int):
        Returns:
            None
        """
        assert (mode in ["max", "min"])
        self.mode = mode

        self.patience = patience
        self.num_bad_count = 0

        if mode == "max":
            self.best = -1 * np.inf
        else:
            self.best = np.inf

    def stop_training(self, metric: float):
        """
        Args:
            metric (float):
        Returns:
            stop_train (bool):
        """
        if self.mode == "max":

            if metric <= self.best:
                self.num_bad_count += 1
            else:
                self.num_bad_count = 0
                self.best = metric

        else:

            if metric >= self.best:
                self.num_bad_count += 1
            else:
                self.num_bad_count = 0
                self.best = metric

        if self.num_bad_count > self.patience:
            stop_train = True
            print("Early stopping applied, stop training")
        else:
            stop_train = False
            print(f"Patience: {self.num_bad_count} / {self.patience} (Best: {self.best:.4f})")
        return stop_train