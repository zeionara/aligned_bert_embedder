from typing import Tuple, Iterable

import yaml


def read_yaml(filename):
    with open(filename) as file:
        return yaml.load(file, Loader=yaml.FullLoader)


def read_lines(filename):
    with open(filename) as f:
        return [line.replace('\n', '') for line in f.readlines()]


def read_tokens(path: str) -> Iterable[Tuple[str]]:
    for sentence in read_lines(path):
        yield tuple(sentence.split(' '))
