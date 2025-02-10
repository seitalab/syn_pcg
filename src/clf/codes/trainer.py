import sys
from typing import Dict, Optional, Tuple, Iterable

import torch
import numpy as np
from tqdm import tqdm

from codes.data.dataloader import prepare_dataloader
sys.path.append("..")
from common.base_trainer import BaseTrainer
from common.monitor import Monitor, EarlyStopper
from common.model.model import prepare_clf_model

class ModelTrainer(BaseTrainer):

    def prepare_dataloader(
        self, 
        data_split: str, 
        is_train: bool=False,
    ) -> Tuple[Iterable, Iterable]:
        """
        Args:
            data_split (str): 
            is_train (bool): 
        Returns:
            loader (Iterable):
        """

        # Prepare dataloader.
        loader = prepare_dataloader(
            self.args, 
            data_split, 
            is_train=is_train,
        )
        return loader

    def set_model(self):

        model = prepare_clf_model(self.args)
        model = model.to(self.args.device)
        self.model = model

    def set_pretrained_model(self, weight_file: str, freeze: bool=False):
        """
        Args:
            weight_file (str):
            freeze (bool):
        Returns:
            None
        """
        if self.args.modelname == "mae_base":
            self.set_pretrained_mae(weight_file, freeze)
        elif self.args.modelname.startswith("resnet"):
            self.set_pretrained_resnet(weight_file, freeze)
        elif self.args.modelname.startswith("effnet"):
            self.set_pretrained_effnet(weight_file, freeze)
        elif self.args.modelname.startswith("gru"):
            self.set_pretrained_rnn(weight_file, freeze)            
        elif self.args.modelname.startswith("lstm"):
            self.set_pretrained_rnn(weight_file, freeze)            
        elif self.args.modelname.startswith("transformer"):
            self.set_pretrained_transformer(weight_file, freeze)            
        elif self.args.modelname.startswith("s4"):
            self.set_pretrained_transformer(weight_file, freeze)            
        else:
            raise NotImplementedError
        
    def set_pretrained_transformer(self, weight_file: str, freeze: bool=False):
        """
        Set trained weight to model.
        Args:
            weight_file (str):
        Returns:
            None
        """
        assert (self.model is not None)

        self.set_pretrained_mae(weight_file, freeze)

    def set_pretrained_rnn(self, weight_file: str, freeze: bool=False):
        """
        Set trained weight to model.
        Args:
            weight_file (str):
        Returns:
            None
        """
        # effnet func. works with gru too.
        self.set_pretrained_effnet(weight_file, freeze)

    def set_pretrained_resnet(
        self, 
        weight_file: str, 
        freeze: bool=False,
    ):
        """
        Set trained weight to model.
        Args:
            weight_file (str):
        Returns:
            None
        """
        # MAE func. works with resnet too.
        self.set_pretrained_mae(weight_file, freeze)

    def set_pretrained_effnet(
        self, 
        weight_file: str, 
        freeze: bool=False,
    ):
        """
        Set trained weight to model.
        Args:
            weight_file (str):
        Returns:
            None
        """
        assert (self.model is not None)

        self.model.backbone.to("cpu")

        # Temporal solution.
        state_dict = dict(
            torch.load(weight_file, map_location="cpu")
        ) # OrderedDict -> dict

        old_keys = list(state_dict.keys())

        for key in old_keys:
            if not key.startswith("backbone."):
                del state_dict[key]
                continue

            new_key = key.replace("module.", "")
            new_key = new_key.replace("backbone.", "")
            state_dict[new_key] = state_dict.pop(key)
        self.model.backbone.load_state_dict(state_dict)

        if freeze:
            for p in self.model.backbone.parameters():
                p.requires_grad = False

        self.model.backbone.to(self.args.device)
        self.model.to(self.args.device)

    def set_pretrained_mae(
        self, 
        weight_file: str, 
        freeze: bool=False,
    ):
        """
        Set trained weight to model.
        Args:
            weight_file (str):
        Returns:
            None
        """
        assert (self.model is not None)

        self.model.backbone.to("cpu")

        # Temporal solution.
        state_dict = dict(
            torch.load(weight_file, map_location="cpu")
        ) # OrderedDict -> dict

        old_keys = list(state_dict.keys())
        try:
            for key in old_keys:
                new_key = key.replace("module.", "")
                state_dict[new_key] = state_dict.pop(key)
            self.model.backbone.load_state_dict(state_dict)
        except:
            print("Error in processing state_dict.")
            for key in old_keys:
                # new_key = "backbone." + key# .replace("backbone.", "")
                if not key.startswith("backbone."):
                    del state_dict[key]
                    continue
                new_key = key.replace("backbone.backbone.", "backbone.")

                state_dict[new_key] = state_dict.pop(key)
            self.model.backbone.load_state_dict(state_dict)

        if freeze:
            for p in self.model.backbone.parameters():
                p.requires_grad = False

        self.model.backbone.to(self.args.device)
        self.model.to(self.args.device)

    def set_lossfunc(self, class_info: Optional[Dict]=None):

        if class_info is not None:
            n_samples = sum(class_info.values())
            class_w = {
                i: n_samples / (len(class_info) * class_info[i]) 
                if i in class_info else 1
                for i in range(len(class_info))
            }
            class_info = np.array([
                class_w[i] 
                for i in range(len(class_info))
            ])
            class_info = torch.tensor(class_info).to(self.args.device).float()

        loss_fn = torch.nn.CrossEntropyLoss(weight=class_info)

        self.loss_fn = loss_fn.to(self.args.device)

    def _evaluate(self, loader, store_sample=True) -> Dict:
        """
        Args:
            loader :
        Returns:
            result_dict (Dict):
        """
        monitor = Monitor()
        self.model.eval()

        with torch.no_grad():

            for X, y in tqdm(loader):

                X = X.to(self.args.device).float()
                y = y.to(self.args.device).long()

                # Only save for the first batch.
                # if store_sample:
                #     self.storer.save_sample(X)

                pred_y = self.model(X)
                minibatch_loss = self.loss_fn(pred_y, y)

                monitor.store_loss(float(minibatch_loss) * len(X))
                monitor.store_num_data(len(X))
                monitor.store_result(y, pred_y)

        monitor.show_result()
        result_dict = {
            "score": monitor.calc_f1(), # For tracking.
            "loss": monitor.average_loss(),
            "f1score": monitor.calc_f1(), # For final csv.
            "Recall": monitor.recall_score(),
            "Precision": monitor.precision_score(),
            "AUROC": monitor.roc_auc_score(),
            "AUPRC": monitor.average_precision_score(),
            "y_trues": monitor.ytrue_record,
            "y_preds": monitor.ypred_record,
            "confusion_matrix": monitor.get_confmat_text()
        }            
        return result_dict

    def _train(self, loader) -> Dict:
        """
        Run train mode iteration.
        Args:
            loader:
        Returns:
            result_dict (Dict):
        """

        monitor = Monitor()
        self.model.train()

        for X, y in tqdm(loader):

            self.optimizer.zero_grad()
            X = X.to(self.args.device).float()
            y = y.to(self.args.device).long()#.squeeze(-1)

            pred_y = self.model(X).squeeze(-1)

            minibatch_loss = self.loss_fn(pred_y, y)

            minibatch_loss.backward()
            self.optimizer.step()

            monitor.store_loss(float(minibatch_loss) * len(X))
            monitor.store_num_data(len(X))
            monitor.store_result(y, pred_y)

        monitor.show_result()
        result_dict = {
            "score": monitor.calc_f1(), 
            "loss": monitor.average_loss(),
            "y_trues": monitor.ytrue_record,
            "y_preds": monitor.ypred_record
        }
        return result_dict

    def run(self, train_loader, valid_loader):
        """
        Args:
            train_loader (Iterable): Dataloader for training data.
            valid_loader (Iterable): Dataloader for validation data.
            mode (str): definition of best (min or max).
        Returns:
            None
        """
        self.best = np.inf * self.flip_val # Sufficiently large or small

        if self.trial is None:
            early_stopper = EarlyStopper(
                mode=self.mode, patience=self.args.patience)

        for epoch in range(1, self.args.epochs + 1):
            print("-" * 80)
            print(f"Epoch: {epoch:03d}")
            train_result = self._train(train_loader)
            self.storer.store_epoch_result(
                epoch, train_result, is_eval=False)

            if epoch % self.args.eval_every == 0:
                eval_result = self._evaluate(valid_loader)
                self.storer.store_epoch_result(
                    epoch, eval_result, is_eval=True)

                if self.mode == "max":
                    monitor_target = eval_result["score"]
                    # self.scheduler.step(eval_result["score"])
                else:
                    monitor_target = eval_result["loss"]
                    # self.scheduler.step(eval_result["loss"])

                # Use pruning if hyperparameter search with optuna.
                # Use early stopping if not hyperparameter search (= trial is None).
                if self.trial is not None:
                    self.trial.report(monitor_target, epoch)
                    if self.trial.should_prune():
                        raise TrialPruned()
                else:
                    if early_stopper.stop_training(monitor_target):
                        break

                self._update_best_result(monitor_target, eval_result)

            self.storer.store_logs()

    def _update_best_result(self, monitor_target, eval_result):
        """
        Args:

        Returns:
            None
        """
        
        if monitor_target * self.flip_val < self.best_val * self.flip_val:
            print(f"Val metric improved {self.best_val:.4f} -> {monitor_target:.4f}")
            self.best_val = monitor_target
            self.best_result = eval_result
            self.storer.save_model(self.model, monitor_target)
        else:
            message = (
                f"Val metric did not improve ({monitor_target:.4f}). "
                f"Current best {self.best_val:.4f}"
            )
            print(message)