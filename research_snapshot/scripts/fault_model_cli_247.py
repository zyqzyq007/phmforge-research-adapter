"""Thin CLI for the source-bound fault classifier used by A247."""
import argparse
from pathlib import Path
import sys


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import real_fault_model  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)

    train = commands.add_parser("train")
    train.add_argument("--data", type=Path, required=True)
    train.add_argument("--out", type=Path, required=True)
    train.add_argument("--arch", default="mlp")
    train.add_argument("--epochs", type=int, required=True)

    predict = commands.add_parser("predict")
    predict.add_argument("--model", type=Path, required=True)
    predict.add_argument("--input", type=Path, required=True)
    predict.add_argument("--out", type=Path, required=True)

    args = parser.parse_args()
    if args.command == "train":
        real_fault_model.train(args.data, args.out, args.arch, args.epochs)
    else:
        real_fault_model.infer(args.model, args.input, args.out)


if __name__ == "__main__":
    main()
