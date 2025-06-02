__version__ = "0.0.1"
import click
import json
from pathlib import Path
from lsr_benchmark.ir_datasets import ensure_corpus_is_extracted, build_dataset_from_local_cache
from lsr_benchmark.corpus import materialize_corpus, materialize_truths, materialize_inputs


def load(ir_datasets_id: str):
    ensure_corpus_is_extracted(ir_datasets_id)
    return build_dataset_from_local_cache(ir_datasets_id)

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

