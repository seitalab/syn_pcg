import os
import sys
from glob import glob

import yaml

sys.path.append("..")
from codes.run_eval import run_eval
from common.experiment_manager import ExperimentManager
from common.utils import ResultManager, get_timestamp

cfg_file = "../../config.yaml"
with open(cfg_file) as f:
    config = yaml.safe_load(f)

class ClassificationExperimentEvaluator(ExperimentManager):

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

    def _run_eval(
        self, 
        eval_target, 
        device, 
        dump_loc, 
        multiseed_run, 
        overwrite_params
    ):
        """
        Args:

        Returns:
            None
        """        
        return run_eval(
            eval_target, 
            device, 
            dump_loc, 
            multiseed_run,
            overwrite_params=overwrite_params
        )

    def _fetch_trained_model_loc(self, params):
        """
        Args:

        Returns:
            None
        """
        # check if params has a `eval_target` attr.
        if "eval_target" not in params:
            raise ValueError("`eval_target` not found in params.")

        # Fetch trained model location.
        trained_model_root = os.path.join(
            params.eval_target,
            "multirun",
            "train",
            f"seed{params.seed:04d}",
        )
        trained_model_loc = glob(trained_model_root + "/*")[-1]

        return trained_model_loc

    def main(self, single_run=False):
        """
        Args:

        Returns:
            None
        """
        # Prepare result storer.
        columns = \
            ["seed", "dataset"] + \
            config["experiment"][self.exe_mode]["result_cols"]

        savename = os.path.join(
            self.save_loc, "ResultTableMultiSeed.csv")
        result_manager = ResultManager(
            savename=savename, columns=columns)

        seeds = config["experiment"][self.exe_mode]["seed"]["multiseed"]
        for _, seed in enumerate(seeds):
            self.param_manager.update_params({"seed": seed})

            # Run training and store result.
            trained_model_loc = \
                self._fetch_trained_model_loc(
                    self.param_manager.get_params())

            # Eval.
            save_loc_eval = os.path.join(
                self.save_loc, 
                "multirun", 
                "eval", 
                f"seed{seed:04d}"
            )
            os.makedirs(save_loc_eval, exist_ok=True)

            overwrite_params = {
                "seed": seed,
                "data_lim": None,
                "val_data_lim": None,
                "dataset": self.param_manager.get_params().dataset
            }
            val_result, test_result = self._run_eval(
                eval_target=trained_model_loc, 
                device=self.device,
                dump_loc=save_loc_eval, 
                multiseed_run=True,
                overwrite_params=overwrite_params
            )
            result_row = self._form_result_row(
                seed, "val", columns, val_result)
            result_manager.add_result(result_row)

            result_row = self._form_result_row(
                seed, "test", columns, test_result)
            result_manager.add_result(result_row)
            
            result_manager.save_result(is_temporal=True)

            if single_run:
                break

        result_manager.save_result()
        return result_manager.savename



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

    executer = ClassificationExperimentEvaluator(
        int(args.exp), 
        args.device,
        debug=args.debug
    )
    executer.main(not args.multirun)
