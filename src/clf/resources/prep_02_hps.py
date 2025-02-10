
from prep_01_gs import generate_exp_yaml

    
if __name__ == "__main__":
    exp_yaml_start = 101
    template_id = 2
    val_replace_dict = {
        "VAL01": ["AS", "AR", "MR"],
    }

    generate_exp_yaml(
        exp_yaml_start, 
        template_id, 
        val_replace_dict
    )
    
