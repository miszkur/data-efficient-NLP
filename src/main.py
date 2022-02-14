import json
import argparse
from utils import fix_filename
from run_experiment import run_experiment


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", default="test.json", type="str")
    args = parser.parse_args()
    config_file = json.load(open(fix_filename(args.config_file), "r"))
    print(f"Config: {config_file}")
    run(config_file)

main()