"""Experiment utils"""
from pathlib import Path
from argparse import Namespace
from datetime import datetime
from pygit2 import discover_repository, Repository


class ExperimentConfig:
    def __init__(self, name: str, args: Namespace):
        self.name = name
        self.args = args
        self.commit_hash = current_commit_hash()
        self.timestamp = datetime.now()


def current_commit_hash():
    repo = Repository(discover_repository(Path.cwd()))
    return repo.revparse("HEAD")


def test():
    pass


if __name__ == "__main__":
    test()
