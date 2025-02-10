
from prep_03_clf import generate_exp_yaml

    
if __name__ == "__main__":
    exp_yaml_start = 207
    template_id = 9
    val_replace_dict = {
        "VAL01": ["gru", "lstm"],
    }

    generate_exp_yaml(
        exp_yaml_start, 
        template_id, 
        val_replace_dict
    )
    
