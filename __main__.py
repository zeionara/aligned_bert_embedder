import os

import click

from .utils.etc import read_yaml, read_tokens
from . import AlignedBertEmbedder


@click.group()
def main():
    pass


@click.command()
@click.argument('config-path', type=str)
@click.argument('raw-text-path', type=str)
def embed(config_path: str, raw_text_path: str):
    # Some required setup which may actually happen far away from the module
    config = read_yaml(config_path)

    config['paths']['config'] = os.path.join(config['paths']['root'], config['paths']['config'])
    config['paths']['vocabulary'] = os.path.join(config['paths']['root'], config['paths']['vocabulary'])
    config['paths']['checkpoint'] = os.path.join(config['paths']['root'], config['paths']['checkpoint'])

    tokens = tuple(read_tokens(raw_text_path))

    # Embeddings generation from text representations of tokens grouped into sentences
    embeddings = tuple(AlignedBertEmbedder(config).embed(tokens))
    print('Results: ')
    print(embeddings)


if __name__ == '__main__':
    main.add_command(embed)
    main()
