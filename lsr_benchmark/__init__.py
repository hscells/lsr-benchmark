import click
import json
from pathlib import Path
from lsr_benchmark.corpus.corpus_subsampling import create_subsample
from trectools import TrecRun
import ir_datasets

@click.command()
@click.argument('directory', type=Path)
def main(directory):
    config = json.loads((directory/"config.json").read_text())
    subsample = create_subsample(config["runs"], config["ir-datasets-id"], config["subsample_depth"], directory)
    docs_store = ir_datasets.load(config["ir-datasets-id"]).docs_store()

    for doc in subsample:
        docs_store.get(doc)

    print(directory)

if __name__ == '__main__':
    main()

