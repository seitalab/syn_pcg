import sys
from argparse import ArgumentParser

sys.path.append("..")
from utils import split_exp_targets
from train import ClassificationExperimentManager
from evaluate import ClassificationExperimentEvaluator

parser = ArgumentParser()

parser.add_argument(
    '--exp', 
    default="9001,9002"
)
parser.add_argument(
    '--device', 
    default="cuda:0"
)
parser.add_argument(
    '--eval', 
    action="store_true"
)    

args = parser.parse_args()
exp_ids = split_exp_targets(args.exp)

errors = []
for exp_id in exp_ids:
    if args.eval:
        executer = ClassificationExperimentEvaluator(
            int(exp_id), 
            args.device,
            debug=False
        )
    else:
        executer = ClassificationExperimentManager(
            int(exp_id), 
            args.device,
            debug=False
        )
    executer.main(single_run=False)

print("*"*80)
for e in errors:
    print(e)

