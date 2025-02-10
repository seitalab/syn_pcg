from argparse import Namespace
from typing import Iterable, Optional, Dict

import torch
import torch.optim as optim
import numpy as np
from optuna.trial import Trial

from common.monitor import Storer
from common.model.model import prepare_clf_model

class BaseTrainer:

    def __init__(
        self,
        args: Namespace,
        save_dir: str,
        mode: str="min"
    ) -> None:
        """
        Args:
            args (Namespace):
            save_dir (str): Directory to output model weights and parameter info.
            mode (str): 
        Returns:
            None
        """

        self.args = args

        self.storer = Storer(
            save_dir, hasattr(args, "save_model_every"))
        self.model = None
        
        
        assert mode in ["max", "min"]
        self.mode = mode
        self.flip_val = -1 if mode == "max" else 1

        self.best_result = None
        self.best_val = np.inf * self.flip_val # Overwritten during training.

    def set_optimizer(self) -> None:
        """
        Args:
            None
        Returns:
            None
        """
        assert self.model is not None

        if self.args.optimizer == "sgd":
            self.optimizer = optim.SGD(
                self.model.parameters(), 
                lr=self.args.learning_rate
            )
        elif self.args.optimizer == "rmsprop":
            self.optimizer = optim.RMSprop(
                self.model.parameters(), 
                lr=self.args.learning_rate
            )
        elif self.args.optimizer == "adam":
            self.optimizer = optim.Adam(
                self.model.parameters(), 
                lr=self.args.learning_rate
            )
        else:
            raise NotImplementedError
        
        if hasattr(self.args, "scheduler"):
            if self.args.scheduler is not None:
                self._set_scheduler()

    def _set_scheduler(self) -> None:
        """
        Args:
            None
        Returns:
            None
        """

        if self.args.scheduler.startswith("plateau-"):
            if self.args.scheduler == "plateau-01":
                self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                    self.optimizer, 
                    mode="min",
                    patience=10, 
                    factor=0.1
                )
            else:
                raise
        elif self.args.scheduler.startswith("cosine-"):
            if self.args.scheduler == "cosine-01":
                self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                    self.optimizer, 
                    T_max=10, 
                    eta_min=0
                )
            else:
                raise
        elif self.args.scheduler.startswith("exp-"):
            if self.args.scheduler == "exp-01":
                self.scheduler = optim.lr_scheduler.ExponentialLR(
                    self.optimizer, 
                    gamma=0.9
                )
            else:
                raise
        elif self.args.scheduler.startswith("cyclic-"):
            if self.args.scheduler == "cyclic-01":
                self.scheduler = optim.lr_scheduler.CyclicLR(
                    self.optimizer, 
                    base_lr=self.args.learning_rate*0.1,
                    max_lr=self.args.learning_rate*2,
                )
            else:
                raise            
        else:
            raise

    def set_model(self) -> None:
        """
        Args:
            None
        Returns:
            None
        """

        model = prepare_clf_model(self.args)
        model = model.to(self.args.device)
        self.model = model

    def set_pretrained_model(
        self, 
        weight_file: str,
        freeze: bool=False,
    ) -> None:
        """
        Args:
            weight_file (str):
            freeze (bool):
        Returns:
            None
        """
        if self.args.pretrained_model == "mae":
            self._set_pretrained_mae(weight_file, freeze)
        else:
            raise NotImplementedError

    def _set_pretrained_mae(
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

        # Update state_dict.
        old_keys = list(state_dict.keys())
        for key in old_keys:
            new_key = key.replace("module.", "")
            state_dict[new_key] = state_dict.pop(key)
        self.model.backbone.load_state_dict(state_dict)

        # Freeze backbone.
        if freeze:
            for p in self.model.backbone.parameters():
                p.requires_grad = False

        # Send to device.
        self.model.backbone.to(self.args.device)
        self.model.to(self.args.device)

    def set_lossfunc(self, class_info: Optional[Dict]=None):
        """
        Args:
            class_info (Optional[Dict]):
        Returns:
            None
        """
 
        if class_info is not None:
            n_samples = sum(class_info.values())
            raise NotImplementedError
            class_info = torch.tensor(class_info).to(self.args.device).float()
        loss_fn = torch.nn.CrossEntropyLoss(weight=class_info)
 
        self.loss_fn = loss_fn.to(self.args.device)

    def set_trial(self, trial: Trial) -> None:
        """
        Args:
            trial (Trial): Optuna trial.
        Returns:
            None
        """
        self.trial = trial

    def save_params(self) -> None:
        """
        Save parameters.
        Args:
            params
        Returns:
            None
        """
        self.storer.save_params(self.args)

    def get_best_loss(self) -> float:
        """
        Args:
            None
        Returns:
            best_value (float):
        """
        return self.best_val, self.best_result

    def _train(self, iterator: Iterable):
        raise NotImplementedError

    def _evaluate(self, iterator: Iterable):
        raise NotImplementedError

    def run(self, train_loader: Iterable, valid_loader: Iterable):
        raise NotImplementedError

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