import os
import sys
from glob import glob

import yaml

sys.path.append("..")
from codes.run_eval import run_eval
from codes.run_train import run_train
from common.experiment_manager import ExperimentManager

cfg_file = "../../config.yaml"
with open(cfg_file) as f:
    config = yaml.safe_load(f)

class ClassificationExperimentManager(ExperimentManager):

    exe_mode = "clf_exp01"

    def _fetch_config_file(self, exp_id: str):
        """
        Args:

        Returns:
            
        """

        exp_config_file = os.path.join(
            "./resources",
            f"exp{exp_id//100:02d}s",
            f"exp{exp_id:04d}.yaml"
        )

        return exp_config_file

    def _run_train(self, params, save_loc):
        """
        Args:

        Returns:
            None
        """
        if params.dataset == "syn":
            params.dataset_ver_norm = \
                config["experiment"]\
                    [self.exe_mode]["syn_dataset_ver"]["Normal"]
            params.dataset_ver_dx = \
                config["experiment"]\
                    [self.exe_mode]["syn_dataset_ver"][params.target_dx]            
            params.data_lim = 10000
            params.val_data_lim = 2500
        
        if params.batch_size == "per_model":
            params.batch_size = \
                config["experiment"]\
                    [self.exe_mode]["model_to_batchsize"][params.modelname]

        if params.finetune_target is not None:
            params = self._update_finetune_target(params)
        best_result, save_dir = run_train(params, save_loc)

        # Rerun if matching condition.
        # training dataset is syn, and best_result > 0.5
        if params.dataset == "syn":
            best_result, save_dir = self._train_loop(
                best_result["loss"], params, save_dir, save_loc)

        return best_result, save_dir
    
    def _train_loop(self, best_loss, params, save_dir, save_loc):
        """
        Args:

        Returns:
            None
        """
        if not hasattr(params, "rerun"):
            return best_loss, save_dir

        for _ in range(5):
            params.learning_rate = params.learning_rate / 2

            params.aug_mask_ratio = self._update_aug_params(
                params.aug_mask_ratio, False)
            params.max_shift_ratio = self._update_aug_params(
                params.max_shift_ratio, False)
            params.flip_rate = self._update_aug_params(
                params.flip_rate, False)
            params.breathing_scale = self._update_aug_params(
                params.breathing_scale, True)
            params.scale_ratio = self._update_aug_params(
                params.scale_ratio, True)
            params.stretch_ratio = self._update_aug_params(
                params.stretch_ratio, True)

            best_result, save_dir = run_train(params, save_loc)
            best_loss = best_result["loss"]
            if best_loss < 0.5:
                break
        return best_result, save_dir
    
    def _update_aug_params(self, param_val, zero_to_one: bool):
        """
        Args:

        Returns:
            updated_param_val
        """
        if not zero_to_one:
            if param_val < 1:
                param_val *= 2.
            else:
                param_val *= 0.5
        else:
            param_val *= 0.5
        return param_val

    def _update_finetune_target(self, params):
        """
        Args:

        Returns:
            None
        """
        finetune_target = os.path.join(
            params.finetune_target, 
            "multirun",
            "train",
            f"seed{params.seed:04d}",
        )
        finetune_target = glob(finetune_target + "/*")[-1]
        params.finetune_target = finetune_target

        return params

    def _run_eval(self, eval_target, device, dump_loc, multiseed_run):
        """
        Args:

        Returns:
            None
        """        
        return run_eval(eval_target, device, dump_loc, multiseed_run)

if __name__ == "__main__":

    from argparse import ArgumentParser

    parser = ArgumentParser()

    parser.add_argument(
        '--exp', 
        default=0
    )
    parser.add_argument(
        '--device', 
        default="cuda:0"
    )
    parser.add_argument(
        '--debug', 
        action="store_true"
    )
    parser.add_argument(
        '--multirun', 
        action="store_true"
    )    
    args = parser.parse_args()

    print(args)

    executer = ClassificationExperimentManager(
        int(args.exp), 
        args.device,
        debug=args.debug
    )
    executer.main(not args.multirun)
