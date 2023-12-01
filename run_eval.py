import subprocess
from pathlib import Path
from typing import List, Dict
import argparse
import wandb

ACCEL_CMDS = [
    "accelerate",
    "launch",
    "--config_file",
    "../llm-code-understanding/configs/accelerate/simple_inference.yaml",
]


def make_harness_cmd(model: str, tasks: List[str], batch_size: int, kwargs: Dict):
    out = [
        "main.py",
        "--model",
        model,
        "--tasks",
        ",".join(tasks),
        "--allow_code_execution",
        "--batch_size",
        str(batch_size),
    ]

    for k, v in kwargs.items():
        out.append(f"--{k}")
        out.append(str(v))

    return out


def launch_harness(model: str, tasks: List[str], batch_size: int, kwargs: Dict):
    cmd = make_harness_cmd(model, tasks, batch_size, kwargs)
    proc = subprocess.run(cmd, check=True)
    print(proc.stdout)
    return proc


def main(args):
    run_name = args.name
    model = args.model
    print(f"Running Evaluation Harness for {run_name} with '{model}'")
    debug = args.debug
    base_kwargs = {}
    if debug:
        print("Debug mode enabled")
        base_kwargs["limit"] = 10

    print("Running HumanEval and MBPP")
    launch_harness(
        model,
        ["human_eval", "mbpp"],
        10,
        {**base_kwargs, "temperature": 0.2, "n_samples": 50},
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, required=True)
    parser.add_argument("--model", type=str, default="codeparrot/codeparrot-small")
    parser.add_argument("--debug","-d", action="store_true")
    main(parser.parse_args())
