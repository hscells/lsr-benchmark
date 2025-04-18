import click
import json
from pathlib import Path
from lsr_benchmark.corpus.corpus_subsampling import create_subsample
from lsr_benchmark.corpus import load_docs
from trectools import TrecRun
from tqdm import tqdm
import ir_datasets


@click.command()
@click.argument('directory', type=Path)
def main(directory):
    config = json.loads((directory/"config.json").read_text())
    subsample = create_subsample(config["runs"], config["ir-datasets-id"], config["subsample_depth"], directory)
    docs = load_docs(config["ir-datasets-id"], subsample)

    print(directory)

if __name__ == '__main__':
    main()

