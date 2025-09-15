__version__ = "0.0.1"
import json
import gzip
from pathlib import Path
from ir_datasets import registry
from lsr_benchmark.irds import build_dataset, MAPPING_OF_DATASET_IDS
from lsr_benchmark.corpus import materialize_corpus, materialize_truths, materialize_inputs, materialize_queries, materialize_qrels, create_subsample
from click import group, argument

from tirex_tracker import tracking, ExportFormat

SUPPORTED_IR_DATASETS = MAPPING_OF_DATASET_IDS.keys()

from ._commands._evaluate import evaluate
import lsr_benchmark.utils
import os


def register_to_ir_datasets(dataset=None):
    if dataset and os.path.isdir(dataset):
        if dataset not in registry:
            ds = build_dataset(dataset, False)
            registry.register(dataset, ds)
            registry.register("lsr-benchmark/" + dataset, ds)
    elif dataset and dataset not in SUPPORTED_IR_DATASETS:
        raise ValueError(f"Can not register {dataset}.")
    else:
        for k in SUPPORTED_IR_DATASETS:
            irds_id = f"lsr-benchmark/{k}/segmented"
            if irds_id not in registry:
                registry.register(irds_id, build_dataset(k, True))

            irds_id = f"lsr-benchmark/{k}"
            if irds_id not in registry:
                registry.register(irds_id, build_dataset(k, False))


def load(ir_datasets_id: str):
    return build_dataset(ir_datasets_id, False)


@group()
def main():
    pass


def create_subsampled_corpus(directory, config):
    subsample = create_subsample(config["runs"], config["ir-datasets-id"], config["subsample_depth"], directory)
    target_directory = directory

    target_directory.mkdir(exist_ok=True)
    with tracking(export_file_path=Path(target_directory) / "dataset-metadata.yml", export_format=ExportFormat.IR_METADATA):
        materialize_corpus(target_directory, subsample, config)
        materialize_queries(target_directory, config)
        materialize_qrels(target_directory/"qrels.txt", config)


@main.command()
@argument('directory', type=Path)
def create_lsr_corpus(directory):
    config = json.loads((directory/"config.json").read_text())
    create_subsampled_corpus(directory, config)

main.command()(evaluate)

if __name__ == '__main__':
    main()
