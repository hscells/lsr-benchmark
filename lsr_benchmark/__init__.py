__version__ = "0.0.1"
import click
import json
import gzip
from pathlib import Path
from ir_datasets import registry
from lsr_benchmark.irds import ensure_corpus_is_extracted, build_dataset_from_local_cache, MAPPING_OF_DATASET_IDS
from lsr_benchmark.corpus import materialize_corpus, materialize_truths, materialize_inputs, materialize_raw_corpus, create_subsample


SUPPORTED_IR_DATASETS = MAPPING_OF_DATASET_IDS.keys()

def register_to_ir_datasets():
    for k in SUPPORTED_IR_DATASETS:
        irds_id = f"lsr-benchmark/{k}/segmented"
        if irds_id not in registry:
            registry.register(irds_id, build_dataset_from_local_cache(k, True))

        irds_id = f"lsr-benchmark/{k}"
        if irds_id not in registry:
            registry.register(irds_id, build_dataset_from_local_cache(k, False))


def load(ir_datasets_id: str):
    ensure_corpus_is_extracted(ir_datasets_id)
    return build_dataset_from_local_cache(ir_datasets_id, False)

@click.group()
def main():
    pass

@main.command()
def foo():
    print("foo")

def create_subsampled_corpus(directory, config):
    subsample = create_subsample(config["runs"], config["ir-datasets-id"], config["subsample_depth"], directory)
    target_directory = directory / "subsampled-corpus"

    target_directory.mkdir(exist_ok=True)
    with gzip.open(target_directory / "document-mapping.json.gz", "wt") as f:
        f.write(json.dumps({i:i for i in subsample}))

    materialize_raw_corpus(target_directory, subsample, config)
    materialize_inputs(target_directory, config)
    materialize_truths(target_directory, config)
    (target_directory / "document-mapping.json.gz").unlink()


@main.command()
@click.argument('directory', type=Path)
def create_lsr_corpus(directory):
    config = json.loads((directory/"config.json").read_text())
    create_subsampled_corpus(directory, config)
    #materialize_corpus(directory, config)
    #materialize_inputs(directory, config)
    #materialize_truths(directory, config)

if __name__ == '__main__':
    main()

