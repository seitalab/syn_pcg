import os
from datetime import datetime
from typing import Dict, List
from argparse import Namespace

import optuna
import pandas as pd

def get_timestamp() -> str:
    """
    Get timestamp in `yymmdd-hhmmss` format.
    Args:
        None
    Returns:
        timestamp (str): Time stamp in string.
    """
    timestamp = datetime.now()
    timestamp = timestamp.strftime('%Y%m%d-%H%M%S')[2:]
    return timestamp

class ResultManager:

    def __init__(self, savename: str, columns: List):
        self.savename = savename
        self.columns = columns
        self.results = []

    def add_result(self, row: List):
        """
        Add one row to results.
        
        Args:
            row (List): _description_
        Returns: 
            None
        """
        self.results.append(row)

    def get_result_df(self) -> pd.DataFrame:
        """
        Args:
            None
        Returns:
            df_result: 
        """
        df_result = pd.DataFrame(
            self.results, columns=self.columns)
        return df_result
    
    def save_result(self, is_temporal: bool=False):
        """

        Args:
            is_temporal (bool, optional): _description_. Defaults to False.
        """
        df_result = pd.DataFrame(
            self.results, columns=self.columns)
        
        savename = self.savename
        if is_temporal:
            savename = savename.replace(".csv", "_tmp.csv")
        df_result.to_csv(savename)

class TemporalResultSaver:

    def __init__(self, save_loc: str) -> None:
        """
        Args:
            save_loc (str): 
        Returns:
            None
        """
        self.save_loc = save_loc

    def save_temporal_result(self, study, frozen_trial):
        """
        Arguments are required by optuna.
        Args:
            study: 
            frozen_trial: <REQUIRED BY OPTUNA>
        Returns:
            None
        """
        filename = "tmp_result_hps.csv"
        csv_name = os.path.join(
            self.save_loc, filename)
        df_hps = study.trials_dataframe()
        df_hps.to_csv(csv_name)
