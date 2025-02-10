import yaml

from prep_01_gs import (
    load_template,
    get_all_combinations,
    replace_value,
    save_exp_yaml
)

from prep_03_clf import insert_hps_result

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

        # Convert to str.
        exp_yaml = yaml.dump(exp_yaml)

        save_exp_yaml(exp_yaml, exp_id)

    return exp_yaml

    
if __name__ == "__main__":
    exp_yaml_start = 501
    template_id = 6
    val_replace_dict = {
        "VAL01": ["AS", "AR", "MR"],
        "VAL02": ["buet", "private", "syn"],
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
    
