import yaml
import numpy as np
import pandas as pd

from prep_01_gs import (
    load_template,
    get_all_combinations,
    replace_value,
    save_exp_yaml
)

from prep_03_clf import insert_hps_result

hps_result_dict2 = {
    # GRU, LSTM Hyper Parameter Search results
}

pow_key = ["emb_dim", "rnn_hidden"]

def insert_hps_result2(exp_yaml):
    """
    Args:

    Returns:

    """
    modelname = exp_yaml["modelname"]["param_val"]
    hps_result = hps_result_dict2[modelname]
    df_hps = pd.read_csv(hps_result, index_col=0)
    min_idx = np.argmin(df_hps.value.values)
    best_row = df_hps.iloc[min_idx] # -> pd.Series

    for key in exp_yaml.keys():
        
        if exp_yaml[key]["param_val"] != "<HPS-2>":
            continue

        val = int(best_row[f"params_{key}"])
        if key in pow_key:
            val = int(2 ** val)

        exp_yaml[key]["param_val"] = val

    return exp_yaml

def generate_exp_yaml(
    exp_id_start, 
    template_id, 
    val_replace_dict
):
    """
    Args:
        exp_id (_description_): _description_
        template_id (_description_): _description_

    Returns:
        str: _description_
    """
    template = load_template(template_id)
    
    all_combinations = get_all_combinations(val_replace_dict)
    for n_proc, comb in enumerate(all_combinations):
        exp_yaml = template
        exp_id = exp_id_start + n_proc
        for key, val in comb.items():
            exp_yaml = replace_value(exp_yaml, key, val)

        # Convert to dict
        exp_yaml = yaml.safe_load(exp_yaml)

        # Replace HPS vals (augmentation results).
        exp_yaml = insert_hps_result(exp_yaml)

        # Replace HPS vals (augmentation results).
        exp_yaml = insert_hps_result2(exp_yaml)

        # Convert to str.
        exp_yaml = yaml.dump(exp_yaml)

        save_exp_yaml(exp_yaml, exp_id)

    return exp_yaml

    
if __name__ == "__main__":
    exp_yaml_start = 601
    template_id = 11
    val_replace_dict = {
        "VAL01": ["AS", "AR", "MR"],
        "VAL02": ["buet", "private", "syn"],
        "VAL03": [
            "gru", "lstm"
        ]
    }

    generate_exp_yaml(
        exp_yaml_start, 
        template_id, 
        val_replace_dict
    )
    
