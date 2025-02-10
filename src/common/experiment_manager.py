import os
from typing import Optional
from itertools import product

import yaml

from common.utils import ResultManager, get_timestamp
from common.param_manager import ParamManager
from common.hyperparameter_search import run_hps

cfg_file = "../../config.yaml"
with open(cfg_file) as f:
    config = yaml.safe_load(f)

class ExperimentManager:

    exe_mode = None

    def __init__(
        self, 
        exp_id: int, 
        device: str,
        use_cpu: bool=False,
        debug: bool=False
    ):
        """

        Args:
            exp_id (int): _description_
            device (str): cuda device or cpu to use.
            use_cpu (bool, optional): _description_. Defaults to False.
            debug (bool, optional): _description_. Defaults to False.
        """

        exp_config_file = self._fetch_config_file(exp_id)    
        self._prepare_save_loc(exp_id)
        self._select_device(device, use_cpu)

        self.param_manager = ParamManager(
            self.exe_mode, exp_config_file, self.device)

        # For debugging.
        if debug:
            update_dict = {
                "epochs": 2,
                "eval_every": 1
            }
            self.param_manager.update_params(update_dict)
        self.debug = debug

    def _fetch_config_file(self, exp_id: str):
        """
        Args:

        Returns:
            
        """
        raise NotImplementedError

    def _select_device(self, device: str, use_cpu: bool):
        """
        Args:

        Returns:
            None
        """
        if use_cpu:
            device = "cpu"
        self.device = device

    def _prepare_save_loc(self, exp_id: int):
        """
        Args:

        Returns:
            None
        """
        self.save_loc = os.path.join(
            config["path"]["processed_data"],
            config["experiment"][self.exe_mode]["save_root"],
            f"exp{exp_id//100:02d}s",
            f"exp{exp_id:04d}",
            get_timestamp()
        )
        os.makedirs(self.save_loc, exist_ok=True)

    def _save_config(self):
        """

        Args:
            config_file (str): _description_
        Returns:
            None
        """
        # Convert `self.fixed_params` to dict.
        fixed_params = vars(self.param_manager.get_params())

        # Save dict to yaml.
        save_loc = os.path.join(self.save_loc, "exp_config_src.yaml")
        with open(save_loc, "w") as f:
            yaml.dump(fixed_params, f)

    def _run_train(self):
        """
        Args:
            None
        Returns:
            None
        """
        raise NotImplementedError
    
    def _run_eval(self):
        """
        Args:
            None
        Returns:
            None
        """
        raise NotImplementedError

    def run_gs_experiment(self) -> str:
        """
        Args:
            None

        Returns:
            csv_path (str): _description_
        """
        # Prepare parameters.
        self.param_manager.update_params(
            {"seed": config["experiment"][self.exe_mode]["seed"]["gs"]}
        )

        # Prepare search space.
        search_space_dict = self.param_manager.search_params
        search_space = list(
            product(*search_space_dict.values())
        )
        search_keys = list(search_space_dict.keys())

        # Prepare result storer.
        columns = \
            search_keys + \
            config["experiment"][self.exe_mode]["result_cols"] +\
            ["save_loc"]

        savename = os.path.join(
            self.save_loc, "result_table_gs.csv")
        result_manager = ResultManager(
            savename=savename, columns=columns)

        # Execute grid search.
        for param_comb in search_space:
            
            update_dict = {
                search_keys[i]: param
                for i, param in enumerate(param_comb)
            }
            self.param_manager.update_params(update_dict)
            save_loc = os.path.join(self.save_loc, "gs")
            
            # Run training and store result.
            best_val_result, result_save_dir = self._run_train(
                self.param_manager.get_params(), 
                save_loc,
            )

            # Form result row.
            result_row = list(param_comb)
            for key in columns[len(search_keys):]:
                if key not in best_val_result:
                    continue
                result_row.append(best_val_result[key])
            result_row.append(result_save_dir)

            # Store result row.
            result_manager.add_result(result_row)
            result_manager.save_result(is_temporal=True)
        
        result_manager.save_result()
        return result_manager.savename

    def run_hps_experiment(self) -> str:
        """
        Args:
            None
        Returns:
            savename (str): _description_
        """
        self.param_manager.update_params(
            {"seed": config["experiment"][self.exe_mode]["seed"]["hps"]}
        )

        csv_name = run_hps(
            self.param_manager.get_params(), 
            self.save_loc, 
            self.param_manager.search_params
        )

        # Copy hyperparameter result file.
        savename = os.path.join(
            self.save_loc, f"ResultTableHPS.csv")
        os.system(f"cp {csv_name} {savename}")

        return savename

    def run_multiseed_experiment(
        self, 
        score_sheet: Optional[str], 
        single_run: bool,
    ):
        """_summary_

        Args:
            score_sheet (str): _description_
        """
        if score_sheet is not None:
            self.param_manager.update_by_search_result(
                score_sheet, 
            )
        
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
            save_loc_train = os.path.join(
                self.save_loc, "multirun", "train", f"seed{seed:04d}")
            os.makedirs(save_loc_train, exist_ok=True)

            # Run training and store result.
            _, trained_model_loc = self._run_train(
                self.param_manager.get_params(), 
                save_loc_train, 
            )

            # Eval.
            save_loc_eval = os.path.join(
                self.save_loc, "multirun", "eval", f"seed{seed:04d}")
            os.makedirs(save_loc_eval, exist_ok=True)

            val_result, test_result = self._run_eval(
                eval_target=trained_model_loc, 
                device=self.device,
                dump_loc=save_loc_eval, 
                multiseed_run=True
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

    def _form_result_row(self, seed, dataset, columns, result_dict):
        """
        Args:

        Returns:

        """

        result_row = [seed, dataset]
        for key in columns[2:]:
            if key not in result_dict:
                result_row.append(None)
            else:
                result_row.append(result_dict[key])
        return result_row

    def main(self, single_run=False, hps_result=None):
        """
        Args:
            None
        Returns:
            None
        """
        # Overwrite.
        # if self.param_manager.search_mode == "hps":
        #     assert hps_result is None
        #     hps_result = self.fixed_params.hps_result

        # Search.
        if hps_result is None:
            if self.param_manager.search_mode == "hps": 
                csv_path = self.run_hps_experiment()
            elif self.param_manager.search_mode == "gs":
                csv_path = self.run_gs_experiment()
            else:
                csv_path = None
        else:
            csv_path = hps_result

        self.param_manager.update_params(
            {

            }
        )
        self._save_config()
        
        # Multi seed eval.
        self.run_multiseed_experiment(
            csv_path, single_run)
        
        # End
