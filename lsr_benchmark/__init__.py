__version__ = "0.0.1"
import click
import json
from pathlib import Path
from ir_datasets import registry
from lsr_benchmark.ir_datasets import ensure_corpus_is_extracted, build_dataset_from_local_cache, MAPPING_OF_DATASET_IDS
from lsr_benchmark.corpus import materialize_corpus, materialize_truths, materialize_inputs


def register_to_ir_datasets():
    for k in MAPPING_OF_DATASET_IDS.keys():
        irds_id = f"lsr-benchmark/{k}/segmented"
        if irds_id not in registry:
            registry.register(irds_id, build_dataset_from_local_cache(k, True))


def load(ir_datasets_id: str):
    ensure_corpus_is_extracted(ir_datasets_id)
    return build_dataset_from_local_cache(ir_datasets_id, False)

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

