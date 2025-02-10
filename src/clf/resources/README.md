# Experiment setting yaml preparation.

1. `prep_01_gs.py`: Generate experiment yaml for learning rate grid search.

    Template: `template001.yaml`
    Outputs: `exp0001.yaml`

2. `prep_02_hps.py`: Generate experiment yaml for augmentation parameter search (hyper parameter search).

    Template: `template002.yaml`
    Output: `exp0101.yaml`, `exp0102.yaml`, `exp0103.yaml`

3. `prep_03_clf.py`: Generate experiment yaml for classification with ResNet18.

    Requires: Result of exp0101 or exp0102 or exp0103 for each diagnostic class respectively (Need to fill in `hps_result_dict` in L12).
    Template: `template003.yaml`
    Output: exp0201 - exp0209

4. `prep_04_synpt_eval.py`: Generate experiment yaml for evaluation of models trained with synthesized data using real-world data ("syn" setting).

    Requires: Result of models trained with synthesized data.
    Template: `template003.yaml`
    Output: exp0301 - exp0306

5. `prep_05_synpt.py`: Generate experiment yaml for finetuning models trained with synthesized data ("syn -> real" setting)
    
    Requires: Result of models trained with synthesized data.
    Template: `template004.yaml`
    Output: exp0401 - exp0406
