import click
import json
from pathlib import Path
from lsr_benchmark.corpus.corpus_subsampling import create_subsample
from trectools import TrecRun

@click.command()
@click.argument('directory', type=Path)
def main(directory):
    config = json.loads((directory/"config.json").read_text())
    subsample = create_subsample(config["runs"], config["subsample_depth"])

    print(directory)

if __name__ == '__main__':
    main()

