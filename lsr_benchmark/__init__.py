__version__ = "0.0.1rc2"
import json
from pathlib import Path
from ir_datasets import registry
from lsr_benchmark.irds import build_dataset, MAPPING_OF_DATASET_IDS, ir_datasets_from_tira
from lsr_benchmark.corpus import materialize_corpus, materialize_queries, materialize_qrels
from click import group, argument
import lsr_benchmark.click

from tirex_tracker import tracking, ExportFormat

SUPPORTED_IR_DATASETS = MAPPING_OF_DATASET_IDS.keys()

from ._commands._evaluate import evaluate
from ._commands._retrieval import retrieval
from ._commands._download import download
import os


def register_to_ir_datasets(dataset=None):
    if dataset and os.path.isdir(dataset):
        if dataset not in registry:
            ds = build_dataset(dataset, False)
            registry.register(dataset, ds)
            registry.register("lsr-benchmark/" + dataset, ds)
    elif dataset and (dataset in ir_datasets_from_tira() or dataset in ir_datasets_from_tira(True)):
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
    target_directory = directory

    target_directory.mkdir(exist_ok=True)
    with tracking(export_file_path=Path(target_directory) / "dataset-metadata.yml", export_format=ExportFormat.IR_METADATA):
        materialize_corpus(target_directory, config)
        materialize_queries(target_directory, config)
        materialize_qrels(target_directory/"qrels.txt", config)


@main.command()
@argument('directory', type=Path)
def create_lsr_corpus(directory):
    config = json.loads((directory/"config.json").read_text())
    create_subsampled_corpus(directory, config)


@main.command()
def overview():
    overview = json.loads((Path(__file__).parent / "datasets" / "overview.json").read_text())
    df_dataset = []
    overall_size = 0
    overall_datasets = len(overview)
    overall_embeddings = set()
    def f(s):
        return str(int(s/(1024))) + " MB"

    import pandas as pd
    model_to_size = {}

    for dataset_id, stats in overview.items():
        overall_size += int(stats['dataset-size'])
        embeddings_for_dataset = 0
        for embedding, embedding_size in stats['embedding-sizes'].items():
            overall_size += int(embedding_size)
            embeddings_for_dataset += int(embedding_size)
            overall_embeddings.add(embedding)
            model_to_size[embedding] = int(embedding_size) + model_to_size.get(embedding, 0)

        df_dataset += [{"Dataset": dataset_id, "Text": f(int(stats['dataset-size'])), "Avg. Embeddings": f(embeddings_for_dataset/len(overall_embeddings))}]
    df_dataset = pd.DataFrame(df_dataset)
    df_dataset.index = ['']*len(df_dataset)
    
    df_embeddings = []
    for k, v in model_to_size.items():
        df_embeddings += [{"Model": k, "Size (avg)": f(v/len(overall_embeddings))}]


    df_embeddings = pd.DataFrame(df_embeddings)
    df_embeddings.index = ['']*len(df_embeddings)

    print(f"Overview of the lsr-benchmark:\n\n\t- {overall_datasets} Datasets with {len(overall_embeddings)} pre-computed embeddings ({f(overall_size)})\n\nDatasets:\n{df_dataset}\n\nEmbeddings:\n{df_embeddings}")

main.command()(download)
main.command()(evaluate)
main.command()(retrieval)

if __name__ == '__main__':
    main()
