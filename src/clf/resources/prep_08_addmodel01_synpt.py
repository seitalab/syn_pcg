import yaml

from prep_01_gs import (
    load_template,
    get_all_combinations,
    replace_value,
    save_exp_yaml
)

from prep_03_clf import insert_hps_result
from prep_07_addmodel01_synpt_eval import syndata_pretrained_model_dict

def insert_pretrained_model(exp_yaml):

    dx = exp_yaml["target_dx"]["param_val"]
    modelname = exp_yaml["modelname"]["param_val"]

    finetune_target = syndata_pretrained_model_dict[modelname][dx]
    exp_yaml["finetune_target"]["param_val"] = finetune_target

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

        # Replace HPS vals.
        exp_yaml = insert_hps_result(exp_yaml)

        # Replace pretrained model.
        exp_yaml = insert_pretrained_model(exp_yaml)

        # Convert to str.
        exp_yaml = yaml.dump(exp_yaml)

        save_exp_yaml(exp_yaml, exp_id)

    return exp_yaml

    
if __name__ == "__main__":
    exp_yaml_start = 407
    template_id = 8
    val_replace_dict = {
        "VAL01": ["AS", "AR", "MR"],
        "VAL02": ["buet", "private"],
        "VAL03": [
            "resnet34", "resnet50", 
            "effnetb0", "effnetb1",
        ]
    }

    generate_exp_yaml(
        exp_yaml_start, 
        template_id, 
        val_replace_dict
    )
    
