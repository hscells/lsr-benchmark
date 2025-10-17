from lsr_benchmark.datasets import all_embeddings, all_datasets, IR_DATASET_TO_TIRA_DATASET, TIRA_DATASET_ID_TO_IR_DATASET_ID
from pathlib import Path
import os

def retrieve_command():
    import click
    class ClickParamTypeLsrDataset(click.ParamType):
        name = "dataset_or_dir"

        def convert(self, value, param, ctx):
            available_datasets = all_datasets()
            if value in available_datasets:
                return value
            
            if value in IR_DATASET_TO_TIRA_DATASET:
                return IR_DATASET_TO_TIRA_DATASET[value]

            if os.path.isdir(value):
                return os.path.abspath(value)

            msg = f"{value!r} is not a supported dataset " + \
            f"({', '.join(TIRA_DATASET_ID_TO_IR_DATASET_ID.get(i, i) for i in available_datasets)}) " + \
            "or a valid directory path"

            self.fail(msg, param, ctx)

    class ClickParamTypeLsrEmbedding(click.ParamType):
        name = "embedding_or_dir"

        def convert(self, value, param, ctx):
            if value:
                value = value.replace("/", "-")

            available_embeddings = all_embeddings()
            if value in available_embeddings:
                return "lightning-ir/" + value

            if os.path.isdir(value):
                return os.path.abspath(value)

            msg = f"{value!r} is not a supported embedding " + \
            f"({', '.join(available_embeddings)}) " + \
            "or a valid directory path"

            self.fail(msg, param, ctx)

    """A decorator that wraps a Click command with standard retrieval options."""
    def decorator(func):
        func = click.option(
            "--dataset",
            type=ClickParamTypeLsrDataset(),
            required=True,
            help="The dataset id or a local directory."
        )(func)

        func = click.option(
            "--output",
            required=True,
            type=Path,
            help="The directory where the output should be stored."
        )(func)

        func = click.option(
            "--embedding",
            type=ClickParamTypeLsrEmbedding(),
            required=False,
            default="naver/splade-v3",
            help="The embedding model."
        )(func)

        func = click.option(
            "--k",
            type=int,
            required=False,
            default=10,
            help="Number of results to return per each query."
        )(func)

        # Wrap as a command
        func = click.command()(func)
        return func
    return decorator
