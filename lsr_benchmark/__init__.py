__version__ = "0.0.1"
import click
import json
from pathlib import Path
from lsr_benchmark.corpus import materialize_corpus, materialize_truths, materialize_inputs


@click.group()
def main():
    pass

@main.command()
def foo():
    print("foo")

@main.command()
@click.argument('directory', type=Path)
def create_lsr_corpus(directory):
    config = json.loads((directory/"config.json").read_text())
    materialize_corpus(directory, config)
    materialize_inputs(directory, config)
    materialize_truths(directory, config)

if __name__ == '__main__':
    main()

