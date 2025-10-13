import click
from lsr_benchmark.datasets import all_embeddings, all_datasets, all_ir_datasets, TIRA_DATASET_ID_TO_IR_DATASET_ID

@click.argument(
    "dataset",
    type=str,
    type=click.Choice([TIRA_DATASET_ID_TO_IR_DATASET_ID[i] for i in all_datasets()),
    nargs=1,
)
@click.argument(
    "embedding",
    type=str,
    type=click.Choice(all_embeddings()),
    nargs=1,
)
@click.option(
    "-o", "--out",
    type=str,
    required=False,
    multiple=False,
    default=None,
    help="The output directory to write to.",
)
def download(dataset, embedding, out):
    print("foo")
