"""Experiment utils"""
import json
from pathlib import Path
from argparse import Namespace, ArgumentParser
from datetime import datetime
from pygit2 import discover_repository, Repository


class ExperimentConfig:
    def __init__(self, name: str, args: Namespace):
        self.name = name
        self.args = args.__dict__
        self.commit_hash = current_commit_hash()
        self.timestamp = datetime.now()

    def save(self, dir_):
        with open(dir_ / "exp_conf.json", "w", encoding="utf-8") as f:
            f.write(self.__str__())

    def __str__(self):
        return json.dumps(self.__dict__, indent=4, default=str, ensure_ascii=False)


def current_commit_hash():
    repo = Repository(discover_repository(Path.cwd()))
    return str(repo.revparse_single("HEAD").id)


def _dummy_args():
    parser = ArgumentParser("Training with noisy labels on DINO Imagenet features")
    parser.add_argument("--epochs", default=100, type=int, help="Number of epochs of training.")
    parser.add_argument("--batch_size", default=128, type=int, help="Batch size")
    return parser.parse_args()


def test():
    conf = ExperimentConfig("aba", _dummy_args())
    print(conf)
    conf.save(Path.cwd())


if __name__ == "__main__":
    test()
