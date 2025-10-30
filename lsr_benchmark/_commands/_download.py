import click
from tira.rest_api_client import Client
from pathlib import Path
from lsr_benchmark.datasets import all_embeddings, all_ir_datasets, IR_DATASET_TO_TIRA_DATASET
from shutil import copytree

@click.option(
    "--dataset",
    type=click.Choice(all_ir_datasets()),
    required=True,
)
@click.option(
    "--embedding",
    type=click.Choice(all_embeddings()),
    required=True,
)
@click.option(
    "-o", "--out",
    type=str,
    required=False,
    multiple=False,
    default=None,
    help="The output directory to write to.",
)
def download_embeddings(dataset, embedding, out):
    tira = Client()
    ret = tira.get_run_output(f'lsr-benchmark/lightning-ir/{embedding}', IR_DATASET_TO_TIRA_DATASET[dataset])
    if out is not None:
        copytree(ret, out)
        ret = out
    print(ret)
