"""Experiment utils"""
import json
from pathlib import Path
from argparse import Namespace
from datetime import datetime
from pygit2 import discover_repository, Repository


class ExperimentConfig:
    def __init__(self, name: str, args: Namespace):
        self.name = name
        self.args = args.__dict__
        self.commit_hash = current_commit_hash()
        self.timestamp = datetime.now()

    def save(self):
        with open("exp_conf.json", "w", encoding="utf-8") as f:
            json.dump(self.__str__(), f, ensure_ascii=False, indent=4)

    def __str__(self):
        return json.dumps(self.__dict__, indent=4, default=str)


def current_commit_hash():
    repo = Repository(discover_repository(Path.cwd()))
    return str(repo.revparse_single("HEAD").id)


def test():
    conf = ExperimentConfig("aba", None)
    print(conf)
    # print(dir(current_commit_hash()))
    print(current_commit_hash())


if __name__ == "__main__":
    test()
